"""The arguments actions can carry, and the orders units cast (docs/training/1-actions.md#masks).

Which choices are legal, head by head, is computed in C next to the features (wc3hook/masks.c, through
policy/native.py): the type mask, each type's argument mask, each (type, argument)'s actor mask. The rule there is
to allow what might work: a wrongly allowed choice costs a step (the game refuses it), a wrongly forbidden one can
never be learned. So the masks check what a player can see is impossible (not enough gold, no building that makes
it, no skill point, an item not carried, a spell cooling down or out of mana) and leave the finer rules
(placement, target types, ranges) to the game. Replays check them: a pro's choice must never be masked out
(training/replays/check.py). What stays here is shared with that C (tools/feature_tables.py generates its tables
from these): the argument vocabulary and the orders.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .actions import reference

RESUME_HARVESTING = 852017
ORDER_IDS = range(851970, 852800)  # every order the engine has (its order ids are consecutive from 851970)


def _id(code: str) -> int:
    return int.from_bytes(code.encode("latin-1"), "big")


@cache
def argument_ids() -> dict[int, int]:
    """Every argument an action can carry -> its index (1-based; 0 is none): unit, structure and item types,
    upgrades, abilities (to learn) and orders (to cast)."""
    ref = reference()
    codes = sorted(ref["units"]) + sorted(ref["items"]) + sorted(ref["upgrades"]) + sorted(ref["abilities"])
    ids = [_id(c) for c in codes] + list(ORDER_IDS)
    return {v: i + 1 for i, v in enumerate(dict.fromkeys(ids))}


TYPE_ORDERS = Path(__file__).resolve().parents[1] / "replays" / "type_orders.json"


@cache
def type_orders() -> dict[str, frozenset[int]]:
    """Unit type -> orders it was seen casting in replays that no ability in the reference gives (the
    reference's order names come from game data that names some differently from the engine: Destroyer's
    `devourmagic`, a Shadow Priest's `creepheal`). Written by `python -m training.replays.check --type-orders`."""
    if not TYPE_ORDERS.is_file():
        return {}
    return {k: frozenset(v) for k, v in json.loads(TYPE_ORDERS.read_text()).items()}


@cache
def ability_orders() -> frozenset[int]:
    """Orders some ability in the reference gives; other orders (creep spells, return resources) are unknown."""
    out = set()
    for a in reference()["abilities"].values():
        for level in a["levels"].values():
            out.update(o["order_id"] for o in level.get("orders", []) if o.get("order_id") is not None)
    return frozenset(out)


