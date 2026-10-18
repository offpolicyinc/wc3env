"""Many games of a checkpoint against Warcraft's AI at once (wc3env.vector), with what each game says about how
it went: one JSON line per game, printed as each one ends (so a run cut short keeps the games it finished).

    python -m training.play.evaluate runs/cloning/bc6w/last.pt --games 12 [--wine] [--difficulty 0] > eval.jsonl
    python -m training.play.evaluate --table eval.jsonl ...      # the table of finished games, per checkpoint

Many checkpoints at once on GCP: python -m training.cloud.gcp eval.

Per game: the result (victory, defeat, or the time limit), its length, gold and lumber per minute, the largest food
in use (the army and workers it kept), units and heroes killed, buildings razed and built, and the decisions that
found no legal choice, by head. Whether it attacks: orders aimed at an enemy unit or building (right-click, attack,
attack-move: pros fight mostly with right-clicks), enemy buildings ever seen, damage done to them, and the most of
its army (no workers) ever within NEAR of one. Races cycle through every pair; plans come from the plan bank, as in
self-play. With `record`, each game also carries its steps as data set arrays (training/play/diagnose.py).

Whether it follows its plan, over the whole game (not step by step): each game's plan, and a second plan of the same
matchup that the model never sees (the control), are both compared at the end with what the player did, as the
rewards compare them (training/selfplay/reward.py): the build order's edit distance (its first 20 builds), and the
Hamming distances of its unit types and of its research levels. With `empty_half`, every other game is played with
the empty plan and still scored against the plan it would have had: a model that uses its plan is closer to it when
given it than when not, and closer to it than to the control.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator

import numpy as np
import torch

from wc3env.binary import OWN, STRUCTURE
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, local_workers, remote_workers, wine_workers

from ..policy.agent import Agents, Seat
from ..policy.features import PAIRS, STEP_MS, Static, env_race, worker_types
from ..policy.model import load
from ..policy.native import hook_decision, hook_spec
from ..policy.plan import PlanBank
from ..selfplay.actor import GATHER, Recorder
from ..selfplay.reward import CHANNELS, TIME_LIMIT, Shaping, distance, win

NEAR = 1200.0  # game units: a unit this close to an enemy building is at the enemy's base
AT_ENEMY = ("smart", "attack", "attack_move")


def evaluate(checkpoint, games: int, difficulty: int, minutes: float, temperature: float, wine: bool,
             plans: str | None, map_name: str, seed: int = 0, label: str = "", record: bool = False,
             opponent: str = "ai", pair: tuple | None = None, device: str = "cpu", remote: int = 0, replays: str = "", empty_half: bool = False, handicap: float = 1.0) -> Iterator[dict]:  # fmt: skip
    """Plays the games, yielding each one's record as it ends; a game whose worker failed is result "failed". With
    `record`, a record's "data" holds the game's steps as data set arrays (not JSON: callers drop it to print).
    The checkpoint plays slot 0 in even games and slot 1 in odd ones (Warcraft treats slot 0, the local player,
    differently: self-play's actor.Game.slot), the opponent the other; the record's "slot" says which.
    `opponent` "self": the checkpoint plays the other slot too, instead of Warcraft's AI; `handicap`: the AI's hit points as
    a share of their own (league.HANDICAPS). `pair`: every game's (own race, opponent's race) instead of every pair
    in turn. `remote`: the games' workers connect in on this port from elsewhere (wc3env.vector --connect, e.g. Modal
    sandboxes: training/cloud/evaluate.py) instead of starting here. `replays`: a folder, as the games' workers see
    it, for each game's native replay (game-<k>.w3g; the record's "replay" names it; its difficulty and handicap are
    saved beside it, so it plays back as it was played).

    Games step on their own, as in self-play: those whose observations are in are decided together, the rest keep
    stepping. Each record says how long one decision took (model_ms) and a game's step, sent to back (step_ms): means
    so far."""
    rng = np.random.default_rng(seed)
    agents = Agents(load(checkpoint, device), temperature, seed)
    bank = PlanBank(plans) if plans and os.path.exists(plans) else None
    races = [tuple(pair) if pair else PAIRS[(seed + g) % len(PAIRS)] for g in range(games)]
    us = [0] * games  # the checkpoint's slot in each game (0, as in self-play: actor.Game.slot); the opponent's is 1 - it
    configs = [
        # recording against the AI: as self-play's ladder games, the AI plays an observed agent slot, so the critic
        # gets the opponent's scalars and the record carries its values
        GameConfig(map=map_name, players=tuple(sorted((PlayerConfig(u, env_race(a)), PlayerConfig(1 - u, env_race(b), control="computer" if opponent == "ai" and not record else "agent")), key=lambda p: p.slot)),
                   ai_agents=(1 - u,) if opponent == "ai" and record else (),
                   ai_difficulty=difficulty if opponent == "ai" else None, setup=MatchSetup(seed=int(rng.integers(2**31 - 1)), randomize_starts=True),
                   render=False, sound=False, step_ms=STEP_MS, observation="binary",
                   features={u: hook_spec(Static.load(map_name, (a, b))),
                             **({1 - u: hook_spec(Static.load(map_name, (b, a)))} if opponent == "self" or record else {})})
        for (a, b), u in zip(races, us)
    ]  # fmt: skip
    targets = [bank.sample(a, b, rng) if bank else None for a, b in races]  # each game's plan
    controls = [_other(bank, a, b, t, rng) if bank else None for (a, b), t in zip(races, targets)]
    blind = [empty_half and g % 2 == 1 for g in range(games)]  # played with the empty plan, scored against its plan
    seats = [Seat(Static.load(map_name, r), None if b else t) for r, t, b in zip(races, targets, blind)]
    follow = [(Shaping(t), Shaping(c)) for t, c in zip(targets, controls)]  # what the player did, against each
    others = [Seat(Static.load(map_name, r[::-1]), bank.sample(r[1], r[0], rng) if bank else None) for r in races]
    stats = [{"checkpoint": label, "race": a, "opponent": b, "slot": u, "food_peak": 0, "failed": Counter(), "attack_orders": 0,
              "enemy_buildings_seen": 0, "building_damage": 0.0, "units_near_enemy": 0} for (a, b), u in zip(races, us)]  # fmt: skip
    seen = [{} for _ in races]  # per game: enemy building id -> its last hit points
    recorders = [Recorder(s.static, s.plan, None, 0) for s in seats] if record else None
    workers = list(worker_types())

    def watch(i: int, obs) -> None:
        """The attack measures from what the player sees this step."""
        u, s = obs.units, stats[i]
        mine = (u["flags"] & OWN) != 0
        enemy_buildings = u[~mine & ((u["flags"] & STRUCTURE) != 0) & (u["owner"] < 12)]
        for b in enemy_buildings:
            last = seen[i].get(int(b["unit_id"]))
            if last is not None and b["hp"] < last:
                s["building_damage"] += float(last - b["hp"])
            seen[i][int(b["unit_id"])] = float(b["hp"])
        s["enemy_buildings_seen"] = len(seen[i])
        army = u[mine & ((u["flags"] & STRUCTURE) == 0) & ~np.isin(u["type_id"], workers)]
        if len(enemy_buildings) and len(army):
            d2 = (army["x"][:, None] - enemy_buildings["x"][None]) ** 2 + (army["y"][:, None] - enemy_buildings["y"][None]) ** 2
            s["units_near_enemy"] = max(s["units_near_enemy"], int((d2.min(1) < NEAR**2).sum()))
    limit = int(minutes * 60 * 1000 / STEP_MS)
    timing = [0.0, 0.0, 0]  # seconds deciding, seconds games spent stepping (sent -> back, summed), decisions
    start = time.perf_counter()

    def replay_path(i: int) -> str | None:
        sep = "\\" if ":" in replays else "/"
        return f"{replays.rstrip(sep)}{sep}game-{i}.w3g" if replays else None

    def finish(i: int) -> dict:
        s = stats[i]
        s["failed"] = dict(s["failed"].most_common(4))
        s["plan"] = "empty" if blind[i] else "given" if targets[i] is not None else "none"
        s["follow"], s["control"] = (_distances(f) for f in follow[i])
        s["model_ms"] = round(1000 * timing[0] / max(timing[2], 1), 1)  # one decision of the games ready together
        s["step_ms"] = round(1000 * timing[1] / max(sum(steps.values()), 1), 1)  # a game's step, sent -> back
        if recorders is not None:
            s["data"] = recorders[i].chunk(0.0)
        return s

    def at_enemy(obs, action) -> bool:
        if action is None or action.type not in AT_ENEMY or not action.target:
            return False
        target = obs.units[obs.units["unit_id"] == action.target]
        return bool(len(target)) and not target["flags"][0] & OWN and int(target["owner"][0]) < 12

    launch = remote_workers(remote) if remote else wine_workers() if wine else local_workers
    with contextlib.redirect_stdout(io.StringIO()), VectorSession(configs, group_size=2, launch=launch) as session:
        started = session.reset()
        observations = {i: o[us[i]] for i, o in started}
        theirs = {i: o.get(1 - us[i]) for i, o in started}  # the opponent's, when observed (opponent "self", record)
        if handicap < 1:  # the AI's units as if they had this share of their hit points (league.HANDICAPS)
            for i in observations:
                session.debug(i, "handicap", player=1 - us[i], hp=float(handicap))
        live = set(observations)
        steps = dict.fromkeys(live, 0)
        ready, stepping = sorted(live), 0  # games whose observation is in (their decision not sent yet); games stepping
        sent_at: dict[int, float] = {}
        reported = 0  # decisions at the last progress line
        hops: dict[str, float] = {}  # ms summed over steps: w_step the game's step in its worker, h_round sent -> back
        while live:
            # as in self-play's actors: the games whose observations are in decide together and step on at once, while
            # the rest keep stepping; no game waits for a round
            ids, ready = ready, []
            if ids:
                began = time.perf_counter()
                decisions = agents.act([seats[i] for i in ids], [observations[i] for i in ids],
                                       [observations[i].features for i in ids],
                                       [theirs[i].features["scalars"] for i in ids] if recorders is not None else None)  # fmt: skip
                if opponent == "self":  # the other slot decided by the same model, as a seat of its own
                    answers = agents.act([others[i] for i in ids], [theirs[i] for i in ids], [theirs[i].features for i in ids])
                    other = dict(zip(ids, answers))
                timing[0] += time.perf_counter() - began
                timing[2] += 1
                for i, d in zip(ids, decisions):
                    if d.failed:
                        stats[i]["failed"][d.failed] += 1
                    stats[i]["attack_orders"] += at_enemy(observations[i], d.action)
                    if recorders is not None:
                        recorders[i].add(observations[i], d, np.zeros(1, np.float32))
                    u = us[i]
                    taken = {u: hook_decision(d.action)} if d.action is not None else {}
                    sent = {u: d.commands} if opponent == "ai" and recorders is None else {u: d.commands, 1 - u: []}  # (recording: the AI's agent slot gets an empty batch)
                    if opponent == "self":
                        sent[1 - u] = other[i].commands
                        if other[i].action is not None:
                            taken[1 - u] = hook_decision(other[i].action)
                    session.send(i, sent, taken or None, replay_path(i))
                    sent_at[i] = time.perf_counter()
                stepping += len(ids)
            first = True
            while stepping:  # the first game back, and any others back within GATHER
                try:
                    got = session.recv(timeout=None if first else GATHER)
                except RuntimeError as error:  # "game <i>: <what happened>": that game is left out
                    i = int(re.match(r"game (\d+)", str(error)).group(1))
                    stepping, first = stepping - 1, False
                    stats[i]["result"], stats[i]["error"] = "failed", str(error)[:300]
                    live.discard(i)
                    yield finish(i)
                    continue
                if got is None:
                    break
                i, obs, done, info = got
                stepping, first = stepping - 1, False
                timing[1] += time.perf_counter() - sent_at.pop(i)
                for part, ms in info.get("hops", {}).items():  # the way there and back, in parts (wc3env.vector)
                    if not part.startswith("_"):
                        hops[part] = hops.get(part, 0.0) + ms
                final = info["final_observations"][us[i]] if done else obs[us[i]]
                steps[i] += 1
                s = stats[i]
                s["food_peak"] = max(s["food_peak"], final.food_used)
                watch(i, final)
                paid = [f.step(final) for f in follow[i]]
                if recorders is not None:  # the step's reward as self-play pays it: the plan's shaping, and the win
                    over = done or steps[i] >= limit
                    paid[0][CHANNELS.index("win")] = win(final) + (TIME_LIMIT if over and not done else 0.0)
                    recorders[i].extra["reward"][-1] = paid[0]
                    recorders[i].extra["done"][-1] = over
                if done or steps[i] >= limit:
                    if replays:
                        s["replay"] = info.get("replay") if done else session.save_replay(i, replay_path(i))
                    score, played = final.score, final.game_time_seconds / 60
                    s.update(result=final.result or "time limit", minutes=round(played, 1),
                             gold_per_minute=round(score.get("gold_mined", 0) / max(played, 1e-6)),
                             lumber_per_minute=round(score.get("lumber_total", 0) / max(played, 1e-6)),
                             units_killed=score.get("units_killed", 0), heroes_killed=score.get("heroes_killed", 0),
                             structures_razed=score.get("structures_razed", 0),
                             structures_built=score.get("structures_built", 0), food_end=final.food_used)  # fmt: skip
                    live.discard(i)
                    yield finish(i)
                else:
                    observations[i], theirs[i] = obs[us[i]], obs.get(1 - us[i])
                    ready.append(i)
            if timing[2] >= reported + 100 and live:  # progress, past the games' silenced output
                done_steps, reported = sum(steps.values()), timing[2]
                print(f"game minute {done_steps / len(steps) * STEP_MS / 60000:.1f} on average, {len(live)} games: model "
                      f"{1000 * timing[0] / timing[2]:.0f} ms for {done_steps / timing[2]:.1f} games at a time, a game's step "
                      f"{1000 * timing[1] / max(done_steps, 1):.0f} ms there and back ("
                      + ", ".join(f"{k} {v / max(done_steps, 1):.0f}" for k, v in hops.items())
                      + f"), {time.perf_counter() - start:.0f} s so far",
                      file=sys.__stderr__, flush=True)  # fmt: skip


def _other(bank, race: str, opponent: str, plan: dict, rng) -> dict:
    """Another plan of the matchup than `plan`, for the control (the same one if the bank has no other)."""
    for _ in range(50):
        other = bank.sample(race, opponent, rng)
        if not all(np.array_equal(other[k], plan[k]) for k in plan):
            return other
    return other


def _distances(shaping) -> dict:
    """How far what the player did is from a Shaping's plan: the build order's edit distance (of the plan's opening
    length), and the Hamming distances of unit types and research levels (of the plan's counts)."""
    levels = {(c, k) for c, n in shaping.researched.items() for k in range(1, n + 1)}
    return {"build": distance(shaping.opening, shaping.built), "opening": len(shaping.opening),
            "units": len(shaping.units ^ shaping.made), "unit_types": len(shaping.units),
            "upgrades": len(shaping.levels ^ levels), "levels": len(shaping.levels)}  # fmt: skip


def following(rows: list[dict]) -> str:
    """Per checkpoint and plan (given, empty): the mean distances to the game's plan and to the control's (lower:
    closer). The plan is used when "given" is closer to its plan than "empty" is, and than to the control."""
    by = defaultdict(list)
    for r in rows:
        if r.get("result") != "failed" and r.get("follow"):
            by[(r.get("checkpoint", ""), r.get("plan", ""))].append(r)
    out = ["checkpoint                     plan   games   build (of)  control   units (of)  control   upgrades (of)  control"]
    for (name, plan), games in sorted(by.items()):
        mean = lambda which, k: sum(g[which][k] for g in games) / len(games)  # noqa: E731
        out.append(f"{name[-30:]:30s} {plan:6s} {len(games):5d} " + "  ".join(
            f"{mean('follow', k):6.1f} ({mean('follow', n):4.1f}) {mean('control', k):7.1f}"
            for k, n in (("build", "opening"), ("units", "unit_types"), ("upgrades", "levels"))))  # fmt: skip
    return "\n".join(out)


def summary(rows: list[dict]) -> str:
    """Per checkpoint: games, wins, defeats and time limits, then the means over its games that did not fail."""
    by = defaultdict(list)
    for r in rows:
        by[r.get("checkpoint", "")].append(r)
    out = ["checkpoint                     games wins defeats timeouts  minutes gold/min lumber/min food_peak kills razed "
           "attacks seen damage near failed"]
    for name, games in by.items():
        played = [g for g in games if g.get("result") != "failed"]
        n = max(len(played), 1)
        count = lambda r: sum(g.get("result") == r for g in played)  # noqa: E731
        mean = lambda k: sum(g.get(k, 0) for g in played) / n  # noqa: E731
        failed = sum(sum(g.get("failed", {}).values()) for g in played) / n
        out.append(f"{name[-30:]:30s} {len(played):5d} {count('victory'):4d} {count('defeat'):7d} {count('time limit'):8d} "
                   f"{mean('minutes'):8.1f} {mean('gold_per_minute'):8.0f} {mean('lumber_per_minute'):10.0f} {mean('food_peak'):9.0f} "
                   f"{mean('units_killed'):5.0f} {mean('structures_razed'):5.1f} {mean('attack_orders'):7.0f} "
                   f"{mean('enemy_buildings_seen'):4.1f} {mean('building_damage'):6.0f} {mean('units_near_enemy'):4.1f} {failed:6.1f}")  # fmt: skip
    for name, games in by.items():  # Warcraft treats slot 0 differently (evaluate's docstring): wins by slot
        slots = [[g for g in games if g.get("result") != "failed" and g.get("slot", 0) == k] for k in (0, 1)]
        out.append(f"{name[-30:]:30s} wins by slot: " + ", ".join(
            f"{k}: {sum(g.get('result') == 'victory' for g in played)} of {len(played)}" for k, played in enumerate(slots)))  # fmt: skip
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--table", nargs="+", metavar="JSONL", help="print the table of these files' games instead")
    p.add_argument("--label", default="", help="the checkpoint's name in each line (default: its path)")
    p.add_argument("--games", type=int, default=12, help="all at once")
    p.add_argument("--difficulty", type=int, default=0, help="the built-in AI's: 0 easy, 1 normal, 2 insane")
    p.add_argument("--opponent", choices=("ai", "self"), default="ai", help="self: the checkpoint plays both sides "
                   "(each line is the game of the side in its \"slot\")")
    p.add_argument("--minutes", type=float, default=30, help="each game's time limit, in game minutes")
    p.add_argument("--temperature", type=float, default=1.0, help="as the actors sample")
    p.add_argument("--plans", default="runs/plans.npz")
    p.add_argument("--map", default="Maps/FrozenThrone/(2)EchoIsles.w3x")
    p.add_argument("--wine", action="store_true", help="games under Wine (Linux)")
    p.add_argument("--threads", type=int, default=4, help="torch threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replays", default="", help="a folder for each game's replay, as the games see it")
    p.add_argument("--device", default="cpu", help="the model's: cpu or cuda")
    p.add_argument("--races", nargs=2, metavar=("OWN", "OPPONENT"), help="every game's races (default: every pair in turn)")
    p.add_argument("--handicap", type=float, default=1.0, help="player 1's hit points as a share of their own (league.HANDICAPS)")
    p.add_argument("--empty-half", action="store_true", help="every other game with the empty plan (scored against its plan)")
    args = p.parse_args()
    if args.table:
        rows = [json.loads(x) for f in args.table for x in open(f) if x.strip()]
        print(summary(rows))
        print(following(rows))
        return 0
    torch.set_num_threads(args.threads)
    out = sys.stdout  # the games' own output is silenced while they run; these lines are not
    for line in evaluate(args.checkpoint, args.games, args.difficulty, args.minutes, args.temperature, args.wine,
                         args.plans, args.map, args.seed, args.label or args.checkpoint, opponent=args.opponent,
                         pair=args.races, device=args.device, replays=args.replays,
                         empty_half=args.empty_half, handicap=args.handicap):  # fmt: skip
        print(json.dumps(line), file=out, flush=True)  # (no "data": record is off)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
