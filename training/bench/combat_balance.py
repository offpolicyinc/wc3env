"""Is a combat fight fair? Warcraft's attack-move against itself in staged fights (training/selfplay/modes/combat.py),
one local game, both sides scripted: each player should win about half. 2026-10-12, 40 fights without heroes: as two
computer slots 18 and 18 (4 draws); with player 0 the local (human) slot, 1 and 36 (only player 1's units cast on
their own). With heroes, 150 s fights: 16 and 22 (2 draws).

    python -m training.bench.combat_balance --fights 40
    python -m training.bench.combat_balance --fights 2 --watch      # in a visible window, at the game's own speed
    python -m training.bench.combat_balance --fights 2 --watch --policy build/bc12.pt   # player 0: a checkpoint
    python -m training.bench.combat_balance --fights 100 --retreat 0.3 --handicap 0.85   # does pulling back hurt units pay?
    python -m training.bench.combat_balance --wine --fights 200 --handicap 0.85 --seed 1000 --policy cb9.pt,cb7.pt

A checkpoint plays player 0 as training does (the local slot: its units cast only what it orders). Several
checkpoints: a game each, every fight drawn the same in all of them (the same armies in the same places), so they
compare fight by fight; the summary gives each one's wins with a 95% interval and, for two, the fights only one won.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fights", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--human", action="store_true", help="player 0 the local (human) slot, as a whole game has it")
    p.add_argument("--watch", action="store_true", help="a visible window at normal speed, the camera on each fight")
    p.add_argument("--policy", default="", help="checkpoints (comma-separated) deciding for player 0, a game each, on "
                   "--device (sampled, as in self-play); player 1 attack-moves")
    p.add_argument("--device", default="cpu", help="where the checkpoints decide (cuda: only when the GPU is free)")
    p.add_argument("--handicap", type=float, default=0, help="player 1 at this ladder stage (combat.HANDICAPS: below 1 "
                   "its handicap, past 1 its food in units as a multiple of player 0's)")
    p.add_argument("--retreat", type=float, default=0, help="player 0 attack-moves too, but pulls each unit under this "
                   "share of its hit points back behind its army (the scripted check of curriculum.md's first skill)")
    p.add_argument("--wine", action="store_true", help="the game under Wine, from Linux ($WINEPREFIX)")
    a = p.parse_args()
    static = Static.load(combat.MAP, ("orc", "orc"))
    policies = a.policy.split(",") if a.policy else []
    agents = [Agents(load(path, a.device)) for path in policies]
    n = max(len(policies), 1)
    config = GameConfig(map=combat.MAP, players=(PlayerConfig(0, "orc"), PlayerConfig(1, "orc")),
                        setup=MatchSetup(seed=a.seed), computer_agents=() if a.human or agents else (0, 1), render=a.watch,
                        background_visible=a.watch, sound=False, step_ms=STEP_MS, observation="binary",
                        features={0: hook_spec(static)} if agents else None)  # fmt: skip
    rng, winners, began = np.random.default_rng(a.seed), [[] for _ in range(n)], time.time()
    launch = wine_workers() if a.wine else local_workers
    with VectorSession([config] * n, speed=1.0 if a.watch else 2048, launch=launch) as games:
        obs = dict(games.reset())
        for fight in range(a.fights):
            draw = rng.bit_generator.state
            for i in range(n):  # every game the same fight
                if obs[i][0].game_time_seconds > combat.RESET_MINUTES * 60:  # the map loaded again, as training does
                    obs[i] = games.reset_one(i)
                rng.bit_generator.state = draw
                ops = combat.stage_ops(obs[i], rng, a.handicap or None)
                games.stage(i, ops, {0: [], 1: []})
            fights, left = {}, n  # game -> [start, last order, seat, score]
            while left:
                i, obs[i], done, _ = games.recv()
                now = obs[i][0].game_time_seconds
                if i not in fights:  # the fight's first step; a new seat: the model's memory empty
                    fights[i] = [now, -1e9, Seat(static), combat.Score(obs[i])]
                    if a.watch and i == 0:  # the camera on the middle of both armies
                        at = [(o["x"], o["y"]) for op, o in ops if op == "spawn"]
                        games.debug(0, "camera", x=sum(x for x, _ in at) / len(at), y=sum(y for _, y in at) / len(at))
                else:
                    _, winner = fights[i][3].step(obs[i])
                    if done or winner is not None or now - fights[i][0] >= combat.SECONDS:
                        winners[i].append(winner)
                        left -= 1
                        print(f"fight {fight} game {i}: winner {winner} after {now - fights[i][0]:.0f} game s", flush=True)
                        continue
                actions, decisions = {0: [], 1: []}, None
                if now - fights[i][1] >= combat.ORDER_SECONDS:
                    fights[i][1], actions = now, {q: combat.attack_move(obs[i][q], obs[i][1 - q]) for q in (0, 1)}
                    if a.retreat:
                        actions[0] = retreat(obs[i][0], obs[i][1], a.retreat)
                if agents:
                    combat.allow(obs[i][0].features)  # only the fight types, as in training
                    d = agents[i].act([fights[i][2]], [obs[i][0]], [obs[i][0].features])[0]
                    actions[0] = d.commands
                    decisions = {0: hook_decision(d.action)} if d.action is not None else None
                games.send(i, actions, decisions)
    for name, w in zip(policies or ["attack-move"], winners):
        lo, hi = wilson(w.count(0), len(w))
        print(f"{name}: player 0 won {w.count(0)}, player 1 {w.count(1)}, draws {w.count(None)} of {len(w)}; "
              f"won {w.count(0) / len(w):.2f} [{lo:.2f}, {hi:.2f}]")
    if n == 2:  # the same fights: only those one of them won tell them apart (a sign test on those)
        first, second = (sum(x == 0 and y != 0 for x, y in pair) for pair in (zip(*winners), zip(*winners[::-1])))
        tail = sum(math.comb(first + second, k) for k in range(min(first, second) + 1)) / 2 ** (first + second)
        print(f"only {policies[0]} won {first}, only {policies[1]} {second}; sign test p {min(1.0, 2 * tail):.3f}")
    print(f"{time.time() - began:.0f} s")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of k successes in n."""
    centre, half = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def retreat(own, enemy, below: float, back: float = 800.0, near: float = 500.0) -> list[dict]:
    """Attack-move, except that each unit under `below` of its hit points with an enemy within `near` moves `back`
    behind its army's centre, away from the enemy's; once clear of enemies it attack-moves again."""
    orders, us, them = combat.attack_move(own, enemy), combat._fighters(own), combat._fighters(enemy)
    if not len(us) or not len(them):
        return orders
    centre = np.array([us["x"].mean(), us["y"].mean()])
    away = centre - np.array([them["x"].mean(), them["y"].mean()])
    to = centre + back * away / max(float(np.hypot(*away)), 1.0)
    gap = np.hypot(us["x"][:, None] - them["x"][None, :], us["y"][:, None] - them["y"][None, :]).min(axis=1)
    hurt = {int(i) for i, hp, top, d in zip(us["unit_id"], us["hp"], us["max_hp"], gap)
            if hp < below * max(top, 1) and d < near}
    move = {"x": float(to[0]), "y": float(to[1])}
    return [{"unit_id": o["unit_id"], "command": "move", "arguments": move} if o["unit_id"] in hurt else o for o in orders]


if __name__ == "__main__":
    main()
