"""Play a cloned policy against Warcraft's built-in AI: the test that counts (docs/training/5-trainer.md).

    python -m training.play.vs_ai runs/cloning/bc4w/last.pt --games 4 --difficulty 0 --replays runs/games
    python -m training.play.vs_ai runs/cloning/bc4w/last.pt --watch      # in a visible window
    python -m training.play.vs_ai runs/cloning/bc4w/last.pt --itself --watch   # against itself

Each game can be saved as a native replay (--replays): the agent's commands are player commands, so Warcraft's
replay viewer shows exactly what it did.

The agent sees each 250 ms step as in training (features and memory), keeps its memory and the attention window
of the current 64-step chunk, and decides every step (noop is one of the types): head by head, each head's
choices limited by the same masks as the data set (training/policy/masks.py), then the translator
turns the action into wc3env commands. Prints each game's result, length, actions by type, and how many
commands the game refused.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from ..policy.actions import Action
from ..policy.agent import Agents, Seat
from ..policy.features import STEP_MS, Static, env_race
from ..policy.model import Policy, load
from ..policy.native import hook_decision, hook_spec
from ..policy.plan import PLAN_HEROES


def _own_counts(obs) -> dict:
    from wc3env.binary import OWN, fourcc

    own = obs.units[(obs.units["flags"] & OWN) != 0]
    return dict(Counter(fourcc(int(t)) for t in own["type_id"]).most_common())


def _describe(obs, a: Action, commands: list[dict]) -> str:
    from wc3env.binary import fourcc

    def name(uid):
        hit = obs.units[obs.units["unit_id"] == uid]
        return fourcc(int(hit["type_id"][0])) if len(hit) else str(uid)

    arg = fourcc(a.argument) if a.argument >> 24 else str(a.argument or "")
    where = f" at ({a.x:.0f},{a.y:.0f})" if a.x is not None else f" on {name(a.target)}" if a.target else ""
    who = f" by {name(a.actor)}" + (f" +{a.size or 'all'}" if a.size != 1 else "") if a.actor else ""
    who += " (shift)" if a.queued else ""
    return (f"{obs.game_time_seconds:6.1f}s gold {obs.gold:4d} food {obs.food_used}/{obs.food_cap}: "
            f"{a.type} {arg}{who}{where} -> {len(commands)} commands")


def play(model: Policy, map_name: str, races: tuple[str, str], difficulty: int, minutes: float, seed: int,
         plan: dict | None, temperature: float, watch: bool = False, replay: Path | None = None,
         trace: Path | None = None, itself: bool = False) -> dict:
    from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

    config = GameConfig(
        map=map_name,
        players=(PlayerConfig(0, env_race(races[0])), PlayerConfig(1, env_race(races[1]), control="agent" if itself else "computer")),
        ai_difficulty=None if itself else difficulty,
        setup=MatchSetup(seed=seed, randomize_starts=True),
        render=watch,
        sound=False,
        window_mode="background",
        background_visible=watch,  # shown, but kept out of the desktop's input: an unfocused interactive window stalled the clock
        step_ms=STEP_MS,
        observation="binary",
        features={0: hook_spec(Static.load(map_name, races)),
                  **({1: hook_spec(Static.load(map_name, races[::-1]))} if itself else {})},
    )
    agents = Agents(model, temperature, seed)
    seat = Seat(Static.load(map_name, races), plan)
    other = Seat(Static.load(map_name, races[::-1]), None) if itself else None  # with `itself`: the same model plays slot 1
    lines: list[str] = []
    decided, failed = Counter(), Counter()
    refused, commands, result, obs = Counter(), 0, "", None
    with contextlib.redirect_stdout(io.StringIO()), GameSession(config) as session:
        session.reset()
        if not watch:  # as fast as the machine allows; watching needs real time
            session.debug("speed", factor=2048)
            session.debug("waitfloor", ms=1)
        orders: list[dict] = []
        taken = None  # the decision behind `orders`, for the hook's next features
        other_orders: list[dict] = []
        for _ in range(int(minutes * 60 * 1000 / STEP_MS)):
            sent = {0: orders, **({1: other_orders} if itself else {})}
            observations, done, info = session.step(sent, decisions=taken)
            obs = observations[0]
            for r in (info or {}).get("rejected", {}).get(0, []):
                refused[str(r.get("reason", r) if isinstance(r, dict) else r)[:60]] += 1
                lines.append(f"        refused: {r}")
            for pl in (info or {}).get("placements", {}).get(0, []):
                lines.append(f"        placed: {pl}")
            if obs.result or done:
                result = obs.result or "ended"
                break
            if itself:
                decision, theirs = agents.act([seat, other], [obs, observations[1]], [obs.features, observations[1].features])
                other_orders = theirs.commands
            else:
                decision = agents.act([seat], [obs], [obs.features])[0]
            orders = decision.commands
            taken = {0: hook_decision(decision.action)} if decision.action is not None else None
            if itself and theirs.action is not None:
                taken = {**(taken or {}), 1: hook_decision(theirs.action)}
            if decision.failed:
                failed[decision.failed] += 1
            if decision.action is not None:
                decided[decision.action.type] += 1
                lines.append(_describe(obs, decision.action, orders))
            commands += len(orders)
        if replay is not None:
            replay.parent.mkdir(parents=True, exist_ok=True)
            session.save_replay(replay)
    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "result": result or "time limit",
        "minutes": round(obs.game_time_seconds / 60, 1) if obs is not None else 0,
        "decisions": dict(decided.most_common()),
        "commands": commands,
        "refused": dict(refused.most_common(8)),
        "nothing legal": dict(failed.most_common(6)),
        "food": f"{obs.food_used}/{obs.food_cap}" if obs is not None else "",
        "unspent lumber": obs.lumber if obs is not None else 0,
        "gold": obs.gold if obs is not None else 0,
        "gold_mined": obs.score.get("gold_mined", 0) if obs is not None else 0,
        "own": _own_counts(obs) if obs is not None else {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--map", default="Maps/FrozenThrone/(2)EchoIsles.w3x")
    p.add_argument("--races", nargs=2, default=("orc", "human"), help="the agent's, then the AI's")
    p.add_argument("--difficulty", type=int, default=0, help="0 easy, 1 normal, 2 insane")
    p.add_argument("--minutes", type=float, default=30, help="training's games last up to 30 minutes too")
    p.add_argument("--temperature", type=float, default=0.8, help="below 1 favours the likelier choices (AlphaStar played at 0.8)")
    p.add_argument("--plan-from", help="a data set file whose plan the agent is given (none: the empty plan)")
    p.add_argument("--plans", help="a plan bank (training/cloud/plans.py): each game a won pro plan of the matchup")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--watch", action="store_true", help="show the game in a window")
    p.add_argument("--itself", action="store_true", help="the checkpoint plays both sides (no built-in AI)")
    p.add_argument("--replays", type=Path, help="save each game's replay here")
    p.add_argument("--trace", type=Path, help="write each decision (and what the game said) to this folder")
    args = p.parse_args()
    model = load(args.checkpoint)
    plan = None
    if args.plan_from:
        from ..train.loader import ReplayData

        plan = ReplayData(args.plan_from).plan
        print("plan heroes:", [int(h) for h in plan["heroes"][:PLAN_HEROES]])
    bank = None
    if args.plans:
        import numpy as np

        from ..policy.plan import PlanBank

        bank, rng = PlanBank(args.plans), np.random.default_rng(args.seed)
    for g in range(args.games):
        if bank is not None:
            plan = bank.sample(args.races[0], args.races[1], rng)
        replay = args.replays.absolute() / f"{Path(args.checkpoint).parent.name}-{args.races[0]}-vs-{args.races[1]}-{args.seed + g}.w3g" if args.replays else None
        r = play(model, args.map, tuple(args.races), args.difficulty, args.minutes, args.seed + g, plan, args.temperature,
                 args.watch, replay, args.trace / f"game-{args.seed + g}.txt" if args.trace else None, args.itself)
        if replay:
            r["replay"] = str(replay)
        print(f"game {g}: {r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
