"""What a checkpoint does in a fight: the share of its steps given to each order type (noop: none), and how often it
wins, against Warcraft's attack-move at one ladder stage (training/selfplay/modes/combat.py), on this machine's CPU.
2026-10-12: cb4's type head had fallen to ~0.11 nats (cb1's start: 0.59); does it now mostly give no orders?
Also each type's chance (the type head's softmax over the legal types, averaged over steps) and the share of steps it
was legal on: whether spells and items are ever tried (cb5: cast_unit 0.1% of steps).

    python -m training.bench.combat_orders --policy build/cb4-now.pt --fights 6 --handicap 0.95
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession

from ..policy.actions import TYPES
from ..policy.agent import Agents, Seat
from ..policy.features import STEP_MS, Static
from ..policy.model import load
from ..policy.native import hook_decision, hook_spec
from ..selfplay.modes import combat

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--fights", type=int, default=6)
    p.add_argument("--handicap", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="", help="a JSON file for the counts")
    a = p.parse_args()
    static = Static.load(combat.MAP, ("orc", "orc"))
    agents = Agents(load(a.policy))
    logits = []  # the type head's output at each step (one seat)
    agents.model.model.type_head.register_forward_hook(lambda m, i, o: logits.append(o.detach().float().reshape(-1, len(TYPES))[0]))
    chance, legal, failed = np.zeros(len(TYPES)), np.zeros(len(TYPES)), Counter()  # failed: chosen, then no legal choice
    config = GameConfig(map=str(ROOT / "build" / "maps" / "(2)FlatArena.w3x"), players=(PlayerConfig(0, "orc"), PlayerConfig(1, "orc")),
                        setup=MatchSetup(seed=a.seed), sound=False, step_ms=STEP_MS,
                        observation="binary", features={0: hook_spec(static)})  # fmt: skip
    rng, types, wins, began = np.random.default_rng(a.seed), Counter(), Counter(), time.time()
    with VectorSession([config], speed=2048) as games:
        ((_, obs),) = games.reset()
        for fight in range(a.fights):
            ops = combat.stage_ops(obs, rng, a.handicap if a.handicap != 1 else None)
            games.stage(0, ops, {0: [], 1: []})
            _, obs, _, _ = games.recv()
            start, last, seat, score = obs[0].game_time_seconds, -1e9, Seat(static), combat.Score(obs)
            while True:
                now = obs[0].game_time_seconds
                actions = {0: [], 1: []}
                if now - last >= combat.ORDER_SECONDS:  # the scripted side re-aimed, as in training
                    last, actions[1] = now, combat.attack_move(obs[1], obs[0])
                combat.allow(obs[0].features)
                d = agents.act([seat], [obs[0]], [obs[0].features])[0]
                mask = torch.as_tensor(np.asarray(d.masks["type"]).reshape(-1)[: len(TYPES)]).bool()
                chance += torch.softmax(logits[-1].masked_fill(~mask, float("-inf")), -1).numpy()
                legal += mask.numpy()
                if d.failed:
                    failed[d.failed] += 1
                types[TYPES[int(d.choices["type"])] if d.action is not None else "noop"] += 1
                actions[0] = d.commands
                games.send(0, actions, {0: hook_decision(d.action)} if d.action is not None else None)
                _, obs, done, _ = games.recv()
                _, winner = score.step(obs)
                if done or winner is not None or now - start >= combat.SECONDS:
                    wins[winner] += 1
                    noop = types["noop"] / max(sum(types.values()), 1)
                    print(f"fight {fight}: winner {winner} after {now - start:.0f} game s; so far no order on {noop:.0%} of "
                          f"{sum(types.values())} steps", flush=True)
                    break
    steps = sum(types.values())
    shares = {t: round(n / steps, 3) for t, n in types.most_common()}
    print(f"{a.policy}: won {wins[0]} of {a.fights}; {steps} steps; order types {shares}; {time.time() - began:.0f} s")
    chances = {t: round(float(c / steps), 4) for t, c in zip(TYPES, chance)}
    legals = {t: round(float(n / steps), 3) for t, n in zip(TYPES, legal)}
    print("type    chance (mean over steps)   legal on (share of steps)")
    for t in sorted(TYPES, key=lambda t: -chances[t]):
        print(f"{t:12s} {chances[t]:8.4f} {legals[t]:8.3f}")
    print(f"chosen but no legal choice for a head (recorded as no order): {dict(failed.most_common())}")
    if a.out:
        Path(a.out).write_text(json.dumps({"policy": a.policy, "won": wins[0], "fights": a.fights, "steps": steps, "types": shares,
                                           "chance": chances, "legal": legals, "failed": dict(failed)}))


if __name__ == "__main__":
    main()
