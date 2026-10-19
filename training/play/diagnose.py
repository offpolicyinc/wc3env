"""How pros and a checkpoint play over the game, measured the same way on both: pro players from data set files, the
checkpoint's games against the built-in AI recorded in the same format. Per stretch of game time (0-5, 5-10, 10-15
minutes): its army, how often it orders units at the enemy, how much it moves, and how often its army is at the
enemy's base.

    python -m training.play.diagnose --pros runs/stage/dataset3 --files 200
    python -m training.play.diagnose --checkpoint runs/sp4-last.pt --games 6 --minutes 15 [--out numbers.json]

- army: its living units that are neither buildings nor workers, on average;
- at enemy: right-clicks, attacks and attack-moves aimed at an enemy unit or building, per minute (pros fight
  mostly with right-clicks: one pro game had 700 right-clicks and no attack order);
- attack-moves and moves: per minute;
- at base: the share of steps with at least 3 army units within NEAR of an enemy building (seen or remembered).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..policy.actions import TYPES
from ..policy.features import COLUMN, MAP, OWNERS, STEP_MS, type_ids, worker_types
from .evaluate import NEAR

STRETCHES = ((0, 5), (5, 10), (10, 15))
ME, ENEMY = OWNERS.index("me"), OWNERS.index("enemy")
AT_ENEMY = {TYPES.index(t) for t in ("smart", "attack", "attack_move")}
MOVE, ATTACK_MOVE = TYPES.index("move"), TYPES.index("attack_move")


def measure(d: dict) -> dict:
    """One player's game (data set arrays) -> per stretch: army, at_enemy/min, attack_moves/min, moves/min, at_base."""
    counts = d["unit_count"].astype(int)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    min_x, min_y, max_x, max_y = d["bounds"]
    cell = (max_x - min_x) / MAP
    near = (NEAR / cell) ** 2
    ids = type_ids()  # the data stores unit types as these embedding ids
    workers = [ids[w] for w in worker_types() if w in ids]
    structure = d["units"][:, COLUMN["structure"]] > 0.5
    dead = d["units"][:, COLUMN["dead"]] > 0.5
    owner, types, cells = d["unit_owner"], d["unit_type"], d["unit_cell"].astype(int)
    minutes = d["game_time"] / 60 if "game_time" in d else np.arange(len(counts)) * STEP_MS / 60000
    acc = defaultdict(lambda: defaultdict(float))
    for t in range(len(counts)):
        stretch = next((s for s in STRETCHES if s[0] <= minutes[t] < s[1]), None)
        if stretch is None:
            continue
        a, b = starts[t], starts[t] + counts[t]
        o, s, dd = owner[a:b], structure[a:b], dead[a:b]
        army = (o == ME) & ~s & ~dd & ~np.isin(types[a:b], workers)
        enemy_buildings = (o == ENEMY) & s
        r = acc[stretch]
        r["steps"] += 1
        r["army"] += army.sum()
        kind = int(d["type"][t])
        target = int(d["target"][t])
        if kind in AT_ENEMY and 0 <= target < counts[t] and o[target] == ENEMY:
            r["at_enemy"] += 1
        r["attack_moves"] += kind == ATTACK_MOVE
        r["moves"] += kind == MOVE
        if army.sum() >= 3 and enemy_buildings.any():
            ay, ax = np.divmod(cells[a:b][army], MAP)
            by, bx = np.divmod(cells[a:b][enemy_buildings], MAP)
            d2 = (ay[:, None] - by[None]) ** 2 + (ax[:, None] - bx[None]) ** 2
            r["at_base"] += (d2.min(1) < near).sum() >= 3
    out = {}
    for stretch, r in acc.items():
        per_minute = r["steps"] * STEP_MS / 60000
        out[f"{stretch[0]}-{stretch[1]}"] = {"army": r["army"] / r["steps"], "at_enemy": r["at_enemy"] / per_minute,
                                             "attack_moves": r["attack_moves"] / per_minute,
                                             "moves": r["moves"] / per_minute, "at_base": r["at_base"] / r["steps"]}  # fmt: skip
    return out


def pros(folder: Path, files: int, seed: int = 0) -> list[dict]:
    paths = sorted(folder.glob("*.npz"))
    random.Random(seed).shuffle(paths)
    out = []
    for p in paths[:files]:
        d = dict(np.load(p, allow_pickle=True))
        m = measure(d)
        m["won"] = int(d["result"]) > 0
        out.append(m)
    return out


def checkpoint_games(
    checkpoint: str, games: int, minutes: float, map_name: str, plans: str, wine: bool = False
) -> list[dict]:
    """The checkpoint's games against the easy built-in AI (training/play/evaluate.py, recorded), measured."""
    from .evaluate import evaluate

    return [measure(g["data"]) for g in evaluate(checkpoint, games, 0, minutes, 1.0, wine, plans, map_name, record=True)
            if "data" in g]  # fmt: skip


def table(name: str, games: list[dict]) -> str:
    lines = [f"{name}: {len(games)} players"]
    for s in (f"{a}-{b}" for a, b in STRETCHES):
        rows = [g[s] for g in games if s in g]
        if rows:
            mean = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            lines.append(f"  minutes {s:5s}  army {mean['army']:5.1f}  at enemy/min {mean['at_enemy']:5.1f}  "
                         f"attack-moves/min {mean['attack_moves']:4.1f}  moves/min {mean['moves']:5.1f}  "
                         f"army at enemy base {100 * mean['at_base']:4.1f}% of the time  ({len(rows)})")  # fmt: skip
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pros", type=Path, help="a data set folder")
    p.add_argument("--files", type=int, default=200)
    p.add_argument("--checkpoint")
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--map", default="Maps/FrozenThrone/(2)EchoIsles.w3x")
    p.add_argument("--plans", default="runs/plans.npz")
    p.add_argument("--wine", action="store_true", help="games under Wine (Linux)")
    p.add_argument("--out", type=Path, help="also write the per-player numbers here (JSON)")
    args = p.parse_args()
    numbers = {}
    if args.pros:
        numbers["pros"] = pros(args.pros, args.files)
        print(table("pros (all)", numbers["pros"]))
        print(table("pros (winners)", [g for g in numbers["pros"] if g["won"]]))
    if args.checkpoint:
        numbers["checkpoint"] = checkpoint_games(
            args.checkpoint, args.games, args.minutes, args.map, args.plans, args.wine
        )
        print(table(Path(args.checkpoint).name, numbers["checkpoint"]))
    if args.out:
        args.out.write_text(json.dumps(numbers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
