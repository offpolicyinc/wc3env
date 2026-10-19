"""What the features are (docs/training/2-features.md), and the map's data.

The features themselves are computed in C (wc3hook/features.c: in the game hook for live play, as a library
through policy/native.py for recorded observations): unit rows, the remembered tree grid, scalars, the last
decisions. Numbers are Pluto's: fractions of a natural scale (hp of max, x of the map width), positions also as
sin/cos at 4 scales. Types, owners, orders, items, buffs and abilities are ids for embeddings (0: none). Rows: own
units, visible enemies, enemy structures remembered under fog, creeps, neutrals, ground items, and the visible
trees nearest our workers (targets for gathering and rallies); at most MAX_UNITS, in that order. `unit_id` says
which game object each row is, so the action heads can point at rows.

Here: their sizes and columns (COLUMNS), the embedding ids (generated into the C's tables by
tools/feature_tables.py), the map's data (Static), and build_map: the map grid from the rows, as the loader
draws it (the model draws the same on the GPU: model.draw_map).
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from wc3env.data import MAPS

from .actions import reference

MAX_UNITS = 256
MAP = 128
PATCH = 16  # the map is 8 x 8 patches (regions) of 16 x 16 cells
RACES = ("human", "orc", "undead", "nightelf")
PAIRS = [(a, b) for a in RACES for b in RACES]  # every matchup, (own race, opponent's)
STEP_MS = 250  # one decision per step: 4 a second


def env_race(race: str) -> str:
    """Our race name as wc3env spells it."""
    return race.replace("nightelf", "night_elf")


OWNERS = ("none", "me", "enemy", "creep", "neutral", "item", "tree")  # embedding ids 0..6
MAX_BUFFS, MAX_ABILITIES = 8, 5
HISTORY = 4  # decisions remembered for "recent" flags and the previous-actions scalars


@cache
def type_ids() -> dict[int, int]:
    """Unit, structure and item type (as its integer id) -> embedding id, 1-based."""
    ref = reference()
    codes = sorted(ref["units"]) + sorted(ref["items"])
    return {int.from_bytes(c.encode("latin-1"), "big"): i + 1 for i, c in enumerate(codes)}


@cache
def ability_ids() -> dict[int, int]:
    return {int.from_bytes(c.encode("latin-1"), "big"): i + 1 for i, c in enumerate(sorted(reference()["abilities"]))}


@cache
def buff_ids() -> dict[int, int]:
    return {int.from_bytes(c.encode("latin-1"), "big"): i + 1 for i, c in enumerate(sorted(reference()["buffs"]))}


@cache
def upgrade_ids() -> dict[int, int]:
    return {int.from_bytes(c.encode("latin-1"), "big"): i for i, c in enumerate(sorted(reference()["upgrades"]))}


@dataclass
class Static:
    """What a game's observations don't repeat: the map's playable bounds (min_x, min_y, max_x, max_y), the
    races (the observer's first) and, from the map's data (`Static.load`), MAP x MAP grids of its terrain
    (fraction of walkable and of buildable ground), the trees it starts with, its gold mines and start
    locations."""

    bounds: tuple[float, float, float, float]
    races: tuple[str, str]
    walkable: np.ndarray | None = None
    buildable: np.ndarray | None = None
    trees: np.ndarray | None = None
    mines: np.ndarray | None = None
    starts: np.ndarray | None = None

    @classmethod
    def load(cls, map_name: str, races: tuple[str, str]) -> Static:
        """From wc3env's data for a map (`python -m tools.prepare map`), named as a replay or the game names it."""
        data = json.loads((MAPS / f"{Path(map_name.replace(chr(92), '/')).stem}.json").read_text(encoding="utf-8"))
        b = data["bounds"]
        bounds = (b["min_x"], b["min_y"], b["max_x"], b["max_y"])
        p = data["pathing"]
        flags = np.frombuffer(zlib.decompress(base64.b64decode(p["flags"])), np.uint8).reshape(p["height"], p["width"])
        ys, xs = np.mgrid[0 : p["height"], 0 : p["width"]]
        cx, cy = p["origin"][0] + 32 * xs + 16, p["origin"][1] + 32 * ys + 16  # each pathing cell's middle
        inside = (cx >= bounds[0]) & (cx < bounds[2]) & (cy >= bounds[1]) & (cy < bounds[3])
        cells = _cell(cx[inside], cy[inside], bounds)
        total = np.zeros((MAP, MAP), np.float32)
        np.add.at(total, cells, 1.0)
        walkable, buildable = np.zeros((MAP, MAP), np.float32), np.zeros((MAP, MAP), np.float32)
        np.add.at(walkable, cells, (flags[inside] & 0x02) == 0)
        np.add.at(buildable, cells, (flags[inside] & 0x08) == 0)
        np.divide(walkable, total, out=walkable, where=total > 0)
        np.divide(buildable, total, out=buildable, where=total > 0)

        def points(xs, ys):
            grid = np.zeros((MAP, MAP), np.float32)
            if len(xs):
                np.add.at(grid, _cell(np.array(xs, float), np.array(ys, float), bounds), 1.0)
            return grid

        trees = points(data["trees"]["x"], data["trees"]["y"])
        mines = points([m["x"] for m in data["gold_mines"]], [m["y"] for m in data["gold_mines"]])
        starts = points([s["x"] for s in data["start_locations"]], [s["y"] for s in data["start_locations"]])
        return cls(bounds, races, walkable, buildable, trees, mines, starts)


@cache
def start_points(map_name: str) -> list[tuple[float, float]]:
    """The map's start locations, (x, y) in world units."""
    data = json.loads((MAPS / f"{Path(map_name.replace(chr(92), '/')).stem}.json").read_text(encoding="utf-8"))
    return [(float(s["x"]), float(s["y"])) for s in data["start_locations"]]


# the unit row's numbers, in order
COLUMNS = (
    ("x", "y", *(f"{f}_{axis}_{k}" for k in range(4) for axis in "xy" for f in ("sin", "cos")))
    + ("hp", "max_hp", "mana", "max_mana")
    + ("own", "structure", "hero", "inside", "remembered", "illusion", "dead")
    + ("last_seen", "level", "target_dx", "target_dy", "has_target", "progress", "queue_length")
    + ("armor", "damage_min", "damage_max", "attack_period", "move_speed", "facing_sin", "facing_cos", "resource")
    + ("xp", "skill_points", "strength", "agility", "intelligence")
    + tuple(f"{role}_{k}" for k in range(HISTORY) for role in ("was_actor", "was_target"))
    + ("attacked", "hp_lost")
    + tuple(f"ability_{f}_{k}" for k in range(MAX_ABILITIES) for f in ("level", "cooldown"))
)
COLUMN = {name: i for i, name in enumerate(COLUMNS)}
UNIT_FLOATS = len(COLUMNS)


@cache
def worker_types() -> frozenset[int]:
    return frozenset(
        int.from_bytes(c.encode("latin-1"), "big")
        for c, u in reference()["units"].items()
        if u.get("builds") and not u["structure"]
    )


def static_map(static: Static) -> np.ndarray:
    """[4, MAP, MAP] walkable, buildable, starts and mines: what model.draw_map takes besides the rows."""
    grids = (static.walkable, static.buildable, static.starts, static.mines)
    return np.stack([g if g is not None else np.zeros((MAP, MAP), np.float32) for g in grids]).astype(np.float32)


GOLD_MINE = int.from_bytes(b"ngol", "big")
# trees (remembered), gold mines, own units, own structures, enemy units, enemy structures (incl. remembered),
# creeps, last targets, walkable, buildable, start locations, the enemy's start locations
MAP_CHANNELS = 12
# a start location with none of the player's buildings within this many cells (either way) is the enemy's: on a
# 2-player map, the other base. Every player knows it (StarCraft's API gives AlphaStar the enemy's start
# locations); without it the model saw two identical start marks, and bc7 never once walked to the idle player's base
OWN_BASE = 6


def enemy_starts(starts: np.ndarray, own_structures: np.ndarray) -> np.ndarray:
    """The start-location grid [MAP, MAP] without the starts that have one of the player's buildings (a grid of
    counts) within OWN_BASE cells; all of them while it has no building at all."""
    if not own_structures.any():
        return starts
    out = starts.copy()
    for r, c in zip(*np.nonzero(starts)):  # a few start cells: only their own neighbourhoods are looked at
        # (a window over every cell of the map took 2.2 ms of an actor's 3.4 ms of features per step)
        if own_structures[max(r - OWN_BASE, 0) : r + OWN_BASE + 1, max(c - OWN_BASE, 0) : c + OWN_BASE + 1].any():
            out[r, c] = 0
    return out


def _cell(x, y, bounds):
    min_x, min_y, max_x, max_y = bounds
    cx = np.clip(((np.asarray(x) - min_x) / (max_x - min_x) * MAP).astype(int), 0, MAP - 1)
    cy = np.clip(((np.asarray(y) - min_y) / (max_y - min_y) * MAP).astype(int), 0, MAP - 1)
    return cy, cx


def build_map(n, unit_cell, unit_owner, unit_type, structure, trees, static: Static, recent_cells) -> np.ndarray:
    """The MAP_CHANNELS x MAP x MAP grid from a step's first n rows (their cells, owner and type ids and
    structure flags), the remembered trees, the map's data and the cells of recent decisions' points. The data
    loader rebuilds the grid from stored rows with this same function."""
    m = np.zeros((MAP_CHANNELS, MAP, MAP), np.float32)
    m[0] = trees
    if static.mines is not None:
        m[1] += static.mines
    owner, cell = np.asarray(unit_owner[:n]), np.asarray(unit_cell[:n])
    is_structure = np.asarray(structure[:n]) > 0
    channel = np.full(n, -1)
    channel[owner == OWNERS.index("me")] = 2
    channel[owner == OWNERS.index("enemy")] = 4
    channel[(channel >= 0) & is_structure] += 1  # own/enemy structures: 3 and 5
    channel[owner == OWNERS.index("creep")] = 6
    channel[np.asarray(unit_type[:n]) == type_ids().get(GOLD_MINE, -1)] = -1  # the map's mines are already there
    keep = channel >= 0
    if keep.any():
        np.add.at(m, (channel[keep], cell[keep] // MAP, cell[keep] % MAP), 1.0)
    for c in recent_cells:
        m[7, c // MAP, c % MAP] = 1.0
    m[:8] = np.minimum(m[:8], 4.0) / 4.0
    for k, grid in ((8, static.walkable), (9, static.buildable), (10, static.starts)):
        if grid is not None:
            m[k] = np.minimum(grid, 1.0)
    m[11] = enemy_starts(m[10], m[3])
    return m


def scalar_size() -> int:
    """The length of _scalars' vector: 9 numbers, both races one-hot, research levels, unit counts."""
    return 9 + 2 * len(RACES) + len(upgrade_ids()) + len(type_ids()) + 1
