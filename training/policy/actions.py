"""The action space (docs/training/1-actions.md): one decision per step, a type and the heads it uses.

    Action(type="attack_move", actor=15420, size=8, x=1200.0, y=-340.0)
    Action(type="train", argument=fourcc_id("hpea"))

`translate(action, obs)` turns one into wc3env commands. A unit action commands the `size` own units nearest
the anchor that could carry it out (only the anchor's type when `same_type`): `select`. Production goes to the
least busy building that can make the type. The replay converter (training/replays/convert.py) picks the anchor,
same_type and size so that `select` gives back the units the player actually commanded, as closely as the
space allows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache

import numpy as np

from wc3env.binary import DEAD, INSIDE, OWN, STATES, STRUCTURE, fourcc
from wc3env.data import REFERENCE

TYPES = (
    "noop",
    "move",
    "attack_move",
    "smart",
    "attack",
    "stop",
    "hold",
    "train",
    "build",
    "research",
    "rally",
    "cast",
    "cast_point",
    "cast_unit",
    "learn",
    "use_item",
    "buy",
    "sell",
    "drop_item",
    "revive",
    "cancel",
)
HOLD_POSITION, SET_RALLY, CANCEL = 851993, 851980, 851976  # order ids
SIZES = (*range(1, 17), 0)  # an exact count up to 16; 0: every eligible unit
# the types a Shift-click can queue (pros build, then Shift-click the mine so the worker returns to it)
QUEUEABLE = ("move", "attack_move", "smart", "attack", "build", "cast", "cast_point", "cast_unit")
PRODUCTION = ("train", "research", "revive")  # the translator picks the building
SINGLE = ("build", "learn", "use_item", "buy", "sell", "drop_item")  # the anchor alone

# Which heads each type uses, the one rule for cloning, playing and RL (a head a type does not use is never
# sampled and never in a loss): the argument; where the action goes (AIMS: none, a point, a unit); the actor,
# unless the translator picks the building; same_type and size for group actions; Shift for QUEUEABLE ones.
AIMS = ("none", "point", "unit")
HAS_ARGUMENT = (
    *PRODUCTION,
    "build",
    "learn",
    "cast",
    "cast_point",
    "cast_unit",
    "use_item",
    "buy",
    "sell",
    "drop_item",
)
AIM_OF = {
    "noop": ("none",),
    **{t: ("point",) for t in ("move", "attack_move", "cast_point")},
    **{t: ("unit",) for t in ("smart", "attack", "cast_unit", "buy", "sell")},
    **{t: ("none",) for t in ("stop", "hold", "cancel", "cast", "train", "research", "revive", "learn")},
    **{t: ("point", "unit") for t in ("rally", "build", "drop_item")},
    "use_item": ("none", "point", "unit"),
}
GROUPED = tuple(t for t in TYPES if t not in ("noop", *PRODUCTION, *SINGLE))
MOVES = ("move", "attack_move", "smart", "attack", "stop", "hold")
ITEM_USES = ("use_item", "sell", "drop_item")
ANY_ARGUMENT = (*MOVES, "rally", "cancel", "buy")  # types whose actor does not depend on the argument


def head_rules() -> dict[str, np.ndarray]:
    """Per type (TYPES index): has_argument, has_actor, grouped, queueable [len(TYPES)] and aim [len(TYPES), 3]."""
    rules = {
        "has_argument": np.array([t in HAS_ARGUMENT for t in TYPES]),
        "has_actor": np.array([t != "noop" and t not in PRODUCTION for t in TYPES]),
        "grouped": np.array([t in GROUPED for t in TYPES]),
        "queueable": np.array([t in QUEUEABLE for t in TYPES]),
        "aim": np.array([[a in AIM_OF[t] for a in AIMS] for t in TYPES]),
    }
    return rules


@dataclass(frozen=True)
class Action:
    type: str
    argument: int = 0  # unit, structure, upgrade, ability or item type id, or an ability's order id
    actor: int = 0  # the anchor unit
    same_type: bool = False
    size: int = 1  # one of SIZES
    target: int = 0  # a unit, item or destructable id
    x: float | None = None
    y: float | None = None
    queued: bool = False  # Shift: after the units' current orders instead of replacing them


def fourcc_id(code: str) -> int:
    return int.from_bytes(code.encode("latin-1"), "big")


@cache
def reference() -> dict:
    return json.loads(REFERENCE.read_text())


def size_for(n: int) -> int:
    """n itself, or 0 (every eligible unit) beyond the largest count."""
    return next((s for s in SIZES if s and s >= n), 0)


def eligible(obs, anchor, same_type: bool) -> np.ndarray:
    """The own units an action anchored on `anchor` could command: living, on the map, of the anchor's kind
    (a structure's actions go to structures, a unit's to units), and of its type when same_type."""
    u = obs.units
    flags = u["flags"]
    ok = (flags & OWN != 0) & (flags & (DEAD | INSIDE) == 0)
    ok &= (flags & STRUCTURE != 0) == bool(anchor["flags"] & STRUCTURE)
    if same_type:
        ok &= u["type_id"] == anchor["type_id"]
    return u[ok]


def select(action: Action, obs) -> list[int]:
    """The units the translator commands for a unit action: the `size` eligible units nearest the anchor."""
    rows = obs.units[obs.units["unit_id"] == action.actor]
    if not len(rows):
        return []
    anchor = rows[0]
    if action.type in SINGLE:
        return [action.actor]
    units = eligible(obs, anchor, action.same_type)
    d = (units["x"] - anchor["x"]) ** 2 + (units["y"] - anchor["y"]) ** 2
    d[units["unit_id"] == action.actor] = -1  # the anchor first, even when another unit stands on it
    order = np.argsort(d, kind="stable")
    return units["unit_id"][order[: action.size or len(units)]].tolist()


def producer(obs, type_id: int, kind: str) -> int:
    """The least busy own building that can train, research or revive `type_id` (0 if none)."""
    code, units = fourcc(type_id), reference()["units"]
    best, best_queue = 0, None
    queue = obs.queue
    for u in obs.units:
        if not (u["flags"] & OWN and u["flags"] & STRUCTURE) or STATES[u["state"]] == "constructing":
            continue
        ref = units.get(fourcc(u["type_id"]), {})
        makes = {
            "train": ref.get("trains", []),
            "revive": ref.get("trains", []),
            "research": ref.get("researches", []) + ref.get("upgrades_to", []),
        }[kind]
        if code not in makes:
            continue
        n = int((queue["unit_id"] == u["unit_id"]).sum())
        if best_queue is None or n < best_queue:
            best, best_queue = int(u["unit_id"]), n
    return best


def translate(action: Action, obs) -> list[dict]:
    """The wc3env commands for an action; [] when nothing can carry it out."""
    t, a = action.type, action
    where = {"x": a.x, "y": a.y} if a.x is not None else {"target_id": a.target} if a.target else {}
    if t == "noop":
        return []
    if t in PRODUCTION:
        building = producer(obs, a.argument, t)
        if not building:
            return []
        if t == "revive":
            dead = obs.units[(obs.units["type_id"] == a.argument) & (obs.units["flags"] & DEAD != 0)]
            return [{"unit_id": building, "command": "revive", "arguments": {"target_id": int(dead["unit_id"][0])}}]
        return [{"unit_id": building, "command": t, "arguments": {"type_id": fourcc(a.argument)}}]
    per_unit = {
        "move": ("move", where),
        "attack_move": ("attack", where),
        "smart": ("smart", where),
        "attack": ("attack", where),
        "stop": ("stop", {}),
        "hold": ("cast", {"order": HOLD_POSITION}),
        "rally": ("cast", {"order": SET_RALLY, **where}),
        "cancel": ("cast", {"order": CANCEL}),
        "cast": ("cast", {"order": a.argument}),
        "cast_point": ("cast", {"order": a.argument, **where}),
        "cast_unit": ("cast", {"order": a.argument, **where}),
        "build": (
            "build",
            {"type_id": fourcc(a.argument), **where, **({"auto_place": True} if a.x is not None else {})},
        ),
        "learn": ("learn", {"ability_id": fourcc(a.argument)}),
    }
    if t in per_unit:
        command, arguments = per_unit[t]
        if a.queued and t in QUEUEABLE:
            arguments = {**arguments, "queued": True}
        return [{"unit_id": u, "command": command, "arguments": dict(arguments)} for u in select(a, obs)]
    if t in ITEM_USES:
        inv = obs.inventory
        slots = inv["slot"][(inv["unit_id"] == a.actor) & (inv["type_id"] == a.argument)]
        if not len(slots):
            return []
        command = "use_item" if t == "use_item" else "drop_item"
        return [{"unit_id": a.actor, "command": command, "arguments": {"slot": int(slots[0]), **where}}]
    if t == "buy":
        return [
            {
                "unit_id": a.actor,
                "command": "buy",
                "arguments": {"shop_id": a.target, "item_type_id": fourcc(a.argument)},
            }
        ]
    raise ValueError(f"unknown action type {t}")
