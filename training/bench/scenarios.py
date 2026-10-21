"""The scenario runner: players fight a scenario's opponent (training/selfplay/scenarios.py) on the same draws, so
they compare fight by fight. A player is a checkpoint (deciding for player 0 as training does: the local slot, its
units casting only what it orders) or a script: "solution" (the scenario's hand rule) or "baseline" (a script
without the skill: attack-move). The fixed evaluation is combat mode's fight at stage 0.85 (docs/training/combat.md):

    python -m training.bench.scenarios --wine --device cuda --handicap 0.85 --seed 1000 --fights 400 --games 8 \\
        --record runs/evals.jsonl --policy runs/selfplay/x/last.pt,runs/keep/l6la-450.pt
    python -m training.bench.scenarios --wine --scenario focus_fire --level 0 --fights 200 --games 4 \\
        --policy solution,baseline           # does the skill pay against attack-move? (curriculum.md's gate)
    python -m training.bench.scenarios --fights 2 --watch --policy build/bc12.pt   # a visible window, real time

Each player plays --games games at once, all stepped together (a checkpoint's decisions batched). The summary gives
each player's wins with a 95% interval and the share of its army's value it kept, and for two players the fights
only one won (a sign test). 2026-10-12, Warcraft's attack-move against itself in combat's fights: as two computer
slots 18 and 18 of 40; with player 0 the local (human) slot, 1 and 36 (only player 1's units cast on their own).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, local_workers, wine_workers

from ..policy.agent import Agents, Seat
from ..policy.features import STEP_MS, Static
from ..policy.model import load
from ..policy.native import hook_decision, hook_spec
from ..selfplay.modes import combat
from ..selfplay.scenarios import SCENARIOS

SCRIPTS = ("solution", "baseline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="fight", choices=sorted(SCENARIOS))
    p.add_argument("--level", type=int, default=0, help="the scenario's level (fight: --handicap instead)")
    p.add_argument(
        "--handicap",
        type=float,
        default=0,
        help="fight: the opponent's stage (combat.HANDICAPS: below 1 its "
        "handicap, past 1 its food in units as a multiple of player 0's; 0: none)",
    )
    p.add_argument(
        "--policy",
        default="baseline",
        help="the players, comma-separated: checkpoints, 'solution', 'baseline' (fight's baseline: attack-move)",
    )
    p.add_argument("--fights", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--games", type=int, default=1, help="games a player, stepped together")
    p.add_argument("--device", default="cpu", help="where checkpoints decide (cuda: only when the GPU is free)")
    p.add_argument("--human", action="store_true", help="scripts only: player 0 the local slot, as checkpoints play")
    p.add_argument("--watch", action="store_true", help="a visible window at normal speed, the camera on each fight")
    p.add_argument("--wine", action="store_true", help="the game under Wine, from Linux ($WINEPREFIX)")
    p.add_argument(
        "--record", default="", help="a JSON line of the results appended here (the run board: runs/evals.jsonl)"
    )
    a = p.parse_args()
    scenario = SCENARIOS[a.scenario]
    level = a.handicap if a.scenario == "fight" else a.level
    players = a.policy.split(",")
    if "solution" in players and scenario.solution is None:
        raise SystemExit(f"{a.scenario} has no solution")
    static = Static.load(combat.MAP, ("orc", "orc"))
    agents = {name: Agents(load(name, a.device)) for name in players if name not in SCRIPTS}
    k, n = a.games, len(players) * a.games  # game g: player g // k, on draw g % k of each round
    config = GameConfig(map=combat.MAP, players=(PlayerConfig(0, "orc"), PlayerConfig(1, "orc")),
                        setup=MatchSetup(seed=a.seed), computer_agents=() if a.human or agents else (0, 1), render=a.watch,
                        background_visible=a.watch, sound=False, step_ms=STEP_MS, observation="binary",
                        features={0: hook_spec(static)} if agents else None)  # fmt: skip
    rng, began = np.random.default_rng(a.seed), time.time()
    results = [{} for _ in players]  # player -> {fight: (winner, the share of its army's value it kept)}
    launch = wine_workers() if a.wine else local_workers
    with VectorSession([config] * n, speed=1.0 if a.watch else 2048, launch=launch) as games:
        obs = dict(games.reset())
        for first in range(0, a.fights, k):  # a round: k fights, each the same for every player
            for j in range(k):
                draw = rng.bit_generator.state
                for g in range(j, n, k):
                    if obs[g][0].game_time_seconds > combat.RESET_MINUTES * 60:  # the map again, as training does
                        obs[g] = games.reset_one(g)
                    rng.bit_generator.state = draw
                    ops = scenario().stage_ops(obs[g], rng, level)
                    games.stage(g, ops, {0: [], 1: []})
                    if a.watch and g == 0:  # the camera on the middle of both armies
                        at = [(o["x"], o["y"]) for op, o in ops if op == "spawn"]
                        camera = {"x": sum(x for x, _ in at) / len(at), "y": sum(y for _, y in at) / len(at)}
            live = {}  # game -> [start, last scripted order, seat (the model's memory), score, the fight's scripts,
            # player 0's army value at the start]
            for _ in range(n):
                g, obs[g], _, _ = games.recv()
                score = combat.Score(obs[g])
                live[g] = [obs[g][0].game_time_seconds, -1e9, Seat(static), score, scenario(), score.values[0]]
            if a.watch:
                games.debug(0, "camera", **camera)
            while live:  # every live game a step at a time, each checkpoint's decisions in one batch
                actions, decisions = {g: {0: [], 1: []} for g in live}, {}
                for g, fight in live.items():
                    now, script, player = obs[g][0].game_time_seconds, fight[4], players[g // k]
                    if now - fight[1] >= combat.ORDER_SECONDS:  # the opponent (and the baseline) re-aimed
                        fight[1] = now
                        actions[g][1] = script.opponent(obs[g][1], obs[g][0], level)
                        if player == "baseline":
                            actions[g][0] = script.baseline(obs[g][0], obs[g][1], level)
                    if player == "solution":  # every step: it re-issues only orders that changed
                        actions[g][0] = script.solution(obs[g][0], obs[g][1], level)
                for name, agent in agents.items():
                    mine = [g for g in live if players[g // k] == name]
                    if not mine:
                        continue
                    for g in mine:
                        combat.allow(obs[g][0].features)  # only the fight types, as in training
                    made = agent.act(
                        [live[g][2] for g in mine], [obs[g][0] for g in mine], [obs[g][0].features for g in mine]
                    )
                    for g, d in zip(mine, made):
                        actions[g][0] = d.commands
                        decisions[g] = {0: hook_decision(d.action)} if d.action is not None else None
                for g in live:
                    games.send(g, actions[g], decisions.get(g))
                for _ in range(len(live)):
                    g, obs[g], done, _ = games.recv()
                    now, score = obs[g][0].game_time_seconds, live[g][3]
                    _, winner = score.step(obs[g])
                    if done or winner is not None or now - live[g][0] >= combat.SECONDS:
                        results[g // k][first + g % k] = (winner, score.values[0] / max(live[g][5], 1e-9))
                        print(f"fight {first + g % k} game {g}: winner {winner} after {now - live[g][0]:.0f} game s",
                              flush=True)  # fmt: skip
                        del live[g]
    report(a, players, [[r[f] for f in sorted(r)] for r in results], began)


def report(a, players: list[str], results: list[list[tuple]], began: float) -> None:
    """Each player's wins with a 95% interval; for two, the fights only one of them won; --record's line."""
    rows = {}
    for name, r in zip(players, results):
        w = [x[0] for x in r]
        kept = float(np.mean([x[1] for x in r]))  # the share of its army's value left at the end (0 when it lost)
        rows[name] = {
            "won": w.count(0),
            "lost": w.count(1),
            "draws": w.count(None),
            "fights": len(w),
            "kept": round(kept, 4),
        }
        lo, hi = wilson(w.count(0), len(w))
        print(f"{name}: player 0 won {w.count(0)}, player 1 {w.count(1)}, draws {w.count(None)} of {len(w)}; "
              f"won {w.count(0) / len(w):.2f} [{lo:.2f}, {hi:.2f}]; kept {kept:.2f} of its army's value")  # fmt: skip
    if len(results) == 2:  # the same fights: only those one of them won tell them apart (a sign test on those)
        a_, b_ = ([x[0] for x in r] for r in results)
        first, second = (sum(x == 0 and y != 0 for x, y in pair) for pair in (zip(a_, b_), zip(b_, a_)))
        tail = sum(math.comb(first + second, i) for i in range(min(first, second) + 1)) / 2 ** (first + second)
        print(f"only {players[0]} won {first}, only {players[1]} {second}; sign test p {min(1.0, 2 * tail):.3f}")
    if a.record:
        with open(a.record, "a") as f:
            f.write(json.dumps({"ended": round(time.time(), 1), "scenario": a.scenario, "level": a.level,
                                "handicap": a.handicap, "seed": a.seed, "fights": a.fights, "results": rows}) + "\n")  # fmt: skip
    print(f"{time.time() - began:.0f} s")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of k successes in n."""
    centre, half = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


if __name__ == "__main__":
    main()
