"""How often the players the clone learned from go after the enemy: pros (pro2) against Warcraft's insane AI
(bots3k), per player-game of a sample of each data set on the Modal volume:

- actions a minute (not noop), and the share of them that are point orders, unit-targeted orders, attack-moves;
- point orders into the enemy's half (nearer the enemy's start than one's own);
- orders aimed at an enemy unit, and how many of those aim at one in the fog (a remembered unit: seen before, not now);
- when the player's first own non-structure unit stands in the enemy's half (game minutes).

    python -m modal run training/bench/data_attacks.py
"""

from __future__ import annotations

import modal

app = modal.App("wc3-data-attacks", image=modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
                .add_local_dir("training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
                .add_local_dir("src/wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"]))  # fmt: skip
VOLUME = modal.Volume.from_name("wc3-dataset")
SAMPLE = 300


def geometry(d) -> tuple | None:
    """A player's game (data set arrays): (its base's cell [row, col], the farthest start's: the enemy's on a 2-player
    map), or None when it has no building at the start or the map no second start."""
    import numpy as np

    from training.policy.features import COLUMN, MAP, OWNERS

    count = d["unit_count"].astype(np.int64)
    first = slice(0, int(count[0]))
    owner, cell = d["unit_owner"][first], d["unit_cell"][first].astype(np.int64)
    own_buildings = (owner == OWNERS.index("me")) & (d["units"][first, COLUMN["structure"]] > 0.5)
    starts = np.argwhere(d["static_starts"] > 0).astype(float)
    if not own_buildings.any() or len(starts) < 2:
        return None
    home = np.array([cell[own_buildings] // MAP, cell[own_buildings] % MAP], float).mean(1)
    return home, starts[np.argmax(((starts - home) ** 2).sum(1))]


def in_enemy_half(cells, home, goal):
    """Which map cells (row * MAP + col) are nearer the enemy's start than one's own."""
    import numpy as np

    from training.policy.features import MAP

    p = np.stack([np.asarray(cells) // MAP, np.asarray(cells) % MAP], -1).astype(float)
    return ((p - goal) ** 2).sum(-1) < ((p - home) ** 2).sum(-1)


def one(path: str) -> dict | None:
    import numpy as np

    from training.policy.actions import TYPES
    from training.policy.features import COLUMN, OWNERS

    d = np.load(path)
    count = d["unit_count"].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(count)])
    owner, cell, units = d["unit_owner"], d["unit_cell"].astype(np.int64), d["units"]
    me, enemy = OWNERS.index("me"), OWNERS.index("enemy")
    places = geometry(d)
    if places is None:
        return None
    home, goal = places

    def enemy_half(c: np.ndarray) -> np.ndarray:
        return in_enemy_half(c, home, goal)

    kind, point, target = d["type"].astype(int), d["point"].astype(np.int64), d["target"].astype(int)
    acting = kind > 0
    minutes = float(d["game_time"][-1]) / 60
    points = acting & (point >= 0)
    out = {"minutes": minutes, "actions": int(acting.sum()), "points": int(points.sum()),
           "points_enemy_half": int(enemy_half(point[points]).sum()) if points.any() else 0,
           "attack_moves": int((kind == TYPES.index("attack_move")).sum())}  # fmt: skip
    aimed = fog = 0
    for t in np.flatnonzero(acting & (target >= 0)):
        row = offsets[t] + target[t]
        if owner[row] == enemy:
            aimed += 1
            fog += int(units[row, COLUMN["remembered"]] > 0.5)
    out["aimed_enemy"], out["aimed_fog"] = aimed, fog
    out["army_enemy_half_minute"] = None
    army = (owner == me) & (units[:, COLUMN["structure"]] < 0.5)
    inside = np.zeros(len(owner), bool)
    inside[army] = enemy_half(cell[army])
    step_of = np.repeat(np.arange(len(count)), count)
    steps = step_of[inside]
    if len(steps):
        out["army_enemy_half_minute"] = float(d["game_time"][steps[0]]) / 60
    return out


@app.function(volumes={"/dataset": VOLUME}, cpu=8, memory=16384, timeout=1800)
def survey(folder: str) -> list[dict]:
    import os
    import random
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(f for f in os.listdir(f"/dataset/{folder}") if f.endswith(".npz"))
    random.Random(0).shuffle(files)
    paths = [f"/dataset/{folder}/{f}" for f in files[:SAMPLE]]
    with ProcessPoolExecutor(8) as pool:
        return [r for r in pool.map(one, paths) if r is not None]


@app.local_entrypoint()
def main():
    import numpy as np

    for folder in ("pro2", "bots3k"):
        rows = survey.remote(folder)
        total = lambda k: sum(r[k] for r in rows)  # noqa: E731
        minutes = total("minutes")
        reached = [r["army_enemy_half_minute"] for r in rows if r["army_enemy_half_minute"] is not None]
        print(f"{folder}: {len(rows)} player-games, {minutes / len(rows):.1f} min each")
        print(f"  actions/min {total('actions') / minutes:.1f}; point orders {total('points') / total('actions'):.1%} of actions, "
              f"{total('points_enemy_half') / max(total('points'), 1):.1%} of them into the enemy's half; "
              f"attack-moves {total('attack_moves') / minutes:.2f}/min")
        print(f"  orders aimed at an enemy unit {total('aimed_enemy') / minutes:.2f}/min, "
              f"{total('aimed_fog') / max(total('aimed_enemy'), 1):.1%} of them at one in the fog")
        print(f"  own unit ever in the enemy's half: {len(reached)}/{len(rows)}; first at minute "
              f"median {np.median(reached) if reached else float('nan'):.1f}")
