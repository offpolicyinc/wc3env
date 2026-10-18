"""Is a combat fight fair? Warcraft's attack-move against itself in staged fights (training/selfplay/modes/combat.py),
one local game, both sides scripted: each player should win about half. 2026-10-12, 40 fights without heroes: as two
computer slots 18 and 18 (4 draws); with player 0 the local (human) slot, 1 and 36 (only player 1's units cast on
their own). With heroes, 150 s fights: 16 and 22 (2 draws).

    python -m training.bench.combat_balance --fights 40
    python -m training.bench.combat_balance --fights 2 --watch      # in a visible window, at the game's own speed
    python -m training.bench.combat_balance --fights 2 --watch --policy build/bc12.pt   # player 0: a checkpoint (CPU)
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession

from ..policy.agent import Agents, Seat
from ..policy.features import STEP_MS, Static
from ..policy.model import load
from ..policy.native import hook_decision, hook_spec
from ..selfplay.modes import combat

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fights", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--human", action="store_true", help="player 0 the local (human) slot, as a whole game has it")
    p.add_argument("--watch", action="store_true", help="a visible window at normal speed, the camera on each fight")
    p.add_argument("--policy", default="", help="a checkpoint deciding for player 0, on this machine's CPU (sampled, as in "
                   "self-play); player 1 attack-moves")
    p.add_argument("--handicap", type=float, default=0, help="player 1 at this ladder stage (combat.HANDICAPS: below 1 "
                   "its handicap, past 1 its food in units as a multiple of player 0's)")
    a = p.parse_args()
    static = Static.load(combat.MAP, ("orc", "orc"))
    agents = Agents(load(a.policy)) if a.policy else None
    config = GameConfig(map=str(ROOT / "build" / "maps" / "(2)FlatArena.w3x"), players=(PlayerConfig(0, "orc"), PlayerConfig(1, "orc")),
                        setup=MatchSetup(seed=a.seed), computer_agents=() if a.human else (0, 1), render=a.watch, background_visible=a.watch, sound=False,
                        step_ms=STEP_MS, observation="binary", features={0: hook_spec(static)} if a.policy else None)  # fmt: skip
    rng, results, began = np.random.default_rng(a.seed), Counter(), time.time()
    with VectorSession([config], speed=1.0 if a.watch else 2048) as games:
        ((_, obs),) = games.reset()
        for fight in range(a.fights):
            ops = combat.stage_ops(obs, rng, a.handicap or None)
            games.stage(0, ops, {0: [], 1: []})
            _, obs, _, _ = games.recv()
            if a.watch:  # the camera on the middle of both armies
                at = [(o["x"], o["y"]) for op, o in ops if op == "spawn"]
                games.debug(0, "camera", x=sum(x for x, _ in at) / len(at), y=sum(y for _, y in at) / len(at))
            start, last, seat = obs[0].game_time_seconds, -1e9, Seat(static)  # a new seat: the model's memory empty
            score = combat.Score(obs)
            while True:
                now = obs[0].game_time_seconds
                actions, decisions = {0: [], 1: []}, None
                if now - last >= combat.ORDER_SECONDS:
                    last, actions = now, {q: combat.attack_move(obs[q], obs[1 - q]) for q in (0, 1)}
                if agents is not None:
                    combat.allow(obs[0].features)  # only the fight types, as in training
                    d = agents.act([seat], [obs[0]], [obs[0].features])[0]
                    actions[0] = d.commands
                    decisions = {0: hook_decision(d.action)} if d.action is not None else None
                games.send(0, actions, decisions)
                _, obs, done, _ = games.recv()
                _, winner = score.step(obs)
                if done or winner is not None or now - start >= combat.SECONDS:
                    results[winner] += 1
                    print(f"fight {fight}: winner {winner} after {now - start:.0f} game s", flush=True)
                    break
    print(f"player 0 won {results[0]}, player 1 {results[1]}, draws {results[None]}; {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
