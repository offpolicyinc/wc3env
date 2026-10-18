"""A game's plan: the strategy input the policy is given, as AlphaStar's z (docs/training/3-model.md).

    p = plan(types, arguments)       # a game's label columns (TYPES and argument_ids() indices, per step)
    p["heroes"]  [PLAN_HEROES]  int32: the heroes it made, first to third (argument indices, 0 for none)
    p["order"]   [PLAN_ORDER]   int32: its first builds, trainings and research, in order
    p["mix"]     [len(argument_ids()) + 1] float32: everything it made over the game, sqrt(count) / 4

In cloning each step is given its own game's plan, so the policy learns to play a plan; some games get the
empty plan (all zeros) so it also plays without one. When playing, the plan is sampled from a pro game of the
same matchup, or left empty.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .actions import TYPES, reference
from .masks import _id, argument_ids

PLAN_HEROES, PLAN_ORDER = 3, 20
MAKES = tuple(TYPES.index(t) for t in ("train", "build", "research"))
HIRES = MAKES + (TYPES.index("buy"),)  # a tavern hero is bought


@cache
def hero_arguments() -> frozenset[int]:
    ids = argument_ids()
    return frozenset(ids[_id(code)] for code, u in reference()["units"].items() if u.get("hero"))


def plan(types: np.ndarray, arguments: np.ndarray) -> dict:
    types, arguments = np.asarray(types), np.asarray(arguments)
    made = np.isin(types, MAKES) & (arguments > 0)
    hired = np.isin(types, HIRES) & np.isin(arguments, list(hero_arguments()))
    heroes = np.zeros(PLAN_HEROES, np.int32)
    first = list(dict.fromkeys(arguments[hired].tolist()))[:PLAN_HEROES]  # a revived or re-hired hero once
    heroes[: len(first)] = first
    order = np.zeros(PLAN_ORDER, np.int32)
    seq = arguments[made][:PLAN_ORDER]
    order[: len(seq)] = seq
    counts = np.bincount(arguments[made | hired], minlength=len(argument_ids()) + 1).astype(np.float32)
    counts[0] = 0
    return {"heroes": heroes, "order": order, "mix": np.sqrt(counts) / 4}


def empty() -> dict:
    return {
        "heroes": np.zeros(PLAN_HEROES, np.int32),
        "order": np.zeros(PLAN_ORDER, np.int32),
        "mix": np.zeros(len(argument_ids()) + 1, np.float32),
    }


class PlanBank:
    """The plans of the pro games (python -m training.cloud.plans writes the file), to give a player one from a
    game of its matchup, as AlphaStar sampled z from human games."""

    def __init__(self, path):
        with np.load(path) as d:
            self.heroes, self.order, self.mix = d["heroes"], d["order"], d["mix"].astype(np.float32)
            self.races, self.result = d["races"], d["result"]

    def __len__(self) -> int:
        return len(self.result)

    def sample(self, race: str, opponent: str, rng: np.random.Generator, wins: bool = True) -> dict:
        """A plan of a player of `race` against `opponent` (a "random" opponent matches any), from a game it won
        when there is one; the empty plan if the bank has no game of that race."""
        mine = self.races[:, 0] == race
        theirs = (self.races[:, 1] == opponent) | (self.races[:, 1] == "random")
        won = self.result == 1
        choices = (mine & theirs & won, mine & theirs, mine & won, mine) if wins else (mine & theirs, mine)
        for pick in choices:
            k = np.flatnonzero(pick)
            if len(k):
                i = int(rng.choice(k))
                return {"heroes": self.heroes[i], "order": self.order[i], "mix": self.mix[i]}
        return empty()


def bank_arrays(files) -> dict:
    """The bank's arrays from data set files (their labels give each player's plan)."""
    rows = []
    for f in files:
        with np.load(f) as d:
            p = plan(d["type"], d["argument"])
            rows.append((p, [str(r) for r in d["races"]], int(d["result"])))
    return {
        "heroes": np.stack([r[0]["heroes"] for r in rows]),
        "order": np.stack([r[0]["order"] for r in rows]),
        "mix": np.stack([r[0]["mix"] for r in rows]).astype(np.float16),
        "races": np.array([r[1] for r in rows]),
        "result": np.array([r[2] for r in rows], np.int8),
    }
