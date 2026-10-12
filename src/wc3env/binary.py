"""Binary observations: the JSON observation's facts as numpy record arrays, read from shared memory.

The DLL writes fixed-size records into the mapping `Local\\wc3hook-obs-<pid>` (wc3hook/obsbin.h), as
BWAPI's client reads its GameData: nothing is serialized or parsed. `step(ms, observe=[...])` and
`observe(player, format="binary")` reply with each observation's offset and size; `read` copies it
out before the next request overwrites the mapping. See docs/specs/observations.md#binary-observations.

Needs numpy (`pip install wc3env[binary]`).
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from .protocol import SCORE_FIELDS

MAGIC = 0x31424F57
VERSION = 3
MAP_BYTES = 16 << 20
TABLES = (
    "units",
    "abilities",
    "buffs",
    "queue",
    "inventory",
    "items",
    "destructables",
    "events",
    "heroes",
    "research",
    "orders",
)

_U4, _I4, _F4 = "<u4", "<i4", "<f4"
DTYPES = {
    "units": np.dtype(
        [
            ("unit_id", _U4),
            ("type_id", _U4),
            ("owner", _I4),
            ("x", _F4),
            ("y", _F4),
            ("hp", _F4),
            ("max_hp", _F4),
            ("mana", _F4),
            ("max_mana", _F4),
            ("flags", _U4),
            ("level", _I4),
            ("order_id", _U4),
            ("order_target", _U4),
            ("order_x", _F4),
            ("order_y", _F4),
            ("state", _U4),
            ("state_seconds", _F4),
            ("queue_seconds", _F4),
            ("armor", _F4),
            ("damage_min", _I4),
            ("damage_max", _I4),
            ("attack_period", _F4),
            ("move_speed", _F4),
            ("facing", _F4),
            ("resource", _I4),
        ]
    ),
    "abilities": np.dtype(
        [
            ("unit_id", _U4),
            ("ability_id", _U4),
            ("level", _I4),
            ("mana_cost", _I4),
            ("cooldown_seconds", _F4),
            ("cooldown_remaining", _F4),
        ]
    ),
    "buffs": np.dtype([("unit_id", _U4), ("buff_id", _U4)]),
    "queue": np.dtype([("unit_id", _U4), ("slot", _U4), ("type_id", _U4)]),
    "inventory": np.dtype([("unit_id", _U4), ("slot", _U4), ("type_id", _U4), ("charges", _I4)]),
    "items": np.dtype([("item_id", _U4), ("type_id", _U4), ("x", _F4), ("y", _F4)]),
    "destructables": np.dtype([("id", _U4), ("type_id", _U4), ("x", _F4), ("y", _F4), ("hp", _F4), ("flags", _U4)]),
    "events": np.dtype([("kind", _U4), ("unit_id", _U4), ("other_id", _U4), ("type_id", _U4), ("value", _I4)]),
    "heroes": np.dtype(
        [
            ("unit_id", _U4),
            ("xp", _I4),
            ("skill_points", _I4),
            ("strength", _I4),
            ("agility", _I4),
            ("intelligence", _I4),
        ]
    ),
    "research": np.dtype([("type_id", _U4), ("level", _I4)]),
    "orders": np.dtype(
        [
            ("unit_id", _U4),
            ("order_id", _U4),
            ("kind", _U4),
            ("target_id", _U4),
            ("item_type", _U4),
            ("x", _F4),
            ("y", _F4),
            ("origin", _U4),
            ("queued", _U4),
            ("time_ms", _U4),
        ]
    ),
}
# obsbin.h BinHeader: 292 bytes, one unpack per observation; the table entries follow the table count
HEADER = struct.Struct(f"<6I4i3If{len(SCORE_FIELDS)}iI{3 * len(TABLES)}I")
_TIME_OF_DAY = 13
_SCORE = slice(_TIME_OF_DAY + 1, _TIME_OF_DAY + 1 + len(SCORE_FIELDS))
_TABLE_ENTRIES = _SCORE.stop + 1

# units.flags
OWN, STRUCTURE, HERO, INSIDE, ILLUSION, DEAD = 1, 2, 4, 8, 16, 32
# destructables.flags
LUMBER, INVULNERABLE = 1, 2
# units.state
STATES = ("", "constructing", "upgrading")
RESULTS = ("", "victory", "defeat", "draw")
# orders.kind and orders.origin (who gave the order: the engine on its own, a player's command, a JASS script)
IMMEDIATE, POINT, TARGET = 0, 1, 2
ORIGINS = ("engine", "player", "script")
# events.kind: the engine event id, and the JSON event's kind
EVENT_KINDS = {
    18: "attacked",
    20: "death",
    26: "construct_start",
    27: "construct_cancel",
    28: "construct_finish",
    29: "upgrade_start",
    30: "upgrade_cancel",
    31: "upgrade_finish",
    32: "train_start",
    33: "train_cancel",
    34: "train_finish",
    35: "research_start",
    36: "research_cancel",
    37: "research_finish",
    41: "hero_level",
    42: "hero_learn",
    47: "summon",
    49: "item_pickup",
    50: "item_use",
    52: "damaged",
    272: "unit_sold",
    274: "item_sold",
    277: "spell_effect",
}


def fourcc(value: int) -> str:
    """A type, ability, buff or build-order id as its four characters ('hpea'); '' for 0."""
    return int(value).to_bytes(4, "big").decode("latin-1") if value else ""


@dataclass(frozen=True)
class BinaryObservation:
    """One player's observation. Arrays are copies: they stay valid after later steps."""

    player: int
    sequence: int
    game_time_seconds: float
    gold: int
    lumber: int
    food_used: int
    food_cap: int
    result: str
    events_lost: int
    orders_lost: int
    time_of_day: float  # 0..24; day from 6 to 18
    score: dict[str, int]
    units: np.ndarray
    abilities: np.ndarray
    buffs: np.ndarray
    queue: np.ndarray
    inventory: np.ndarray
    items: np.ndarray
    destructables: np.ndarray
    events: np.ndarray
    heroes: np.ndarray
    research: np.ndarray
    orders: np.ndarray  # orders given to the observer's units since its previous observation, oldest first
    raw: bytes = field(repr=False)  # the records the arrays view
    features_raw: bytes | None = field(default=None, repr=False)  # the hook's features block, if asked for

    def __reduce__(self):  # pickle as the bytes (VectorSession's pipes): cheaper than the arrays
        return _rebuild, (self.raw, self.features_raw)

    @cached_property
    def features(self) -> dict | None:
        """The model's features the hook computed for this observation (GameConfig.features), or None."""
        return parse_features(self.features_raw) if self.features_raw else None

    @cached_property
    def own_unit_ids(self) -> frozenset[int]:
        """The observer's living units, including those inside a mine, building or transport (flag INSIDE)."""
        flags = self.units["flags"]
        return frozenset(self.units["unit_id"][(flags & OWN != 0) & (flags & DEAD == 0)].tolist())


class SharedObservations:
    """The DLL's mapping for one game process."""

    def __init__(self, pid: int):
        self._map = mmap.mmap(-1, MAP_BYTES, tagname=f"Local\\wc3hook-obs-{pid}", access=mmap.ACCESS_READ)

    def read(self, offset: int, size: int, features: tuple[int, int] | None = None) -> BinaryObservation:
        """The observation at offset (one copy of it), with its features block at (offset, size) if given."""
        features_raw = self._map[features[0] : features[0] + features[1]] if features else None
        return _rebuild(self._map[offset : offset + size], features_raw)

    def close(self) -> None:
        self._map.close()


def _rebuild(raw: bytes, features_raw: bytes | None) -> BinaryObservation:
    obs = parse(raw)
    return replace(obs, features_raw=features_raw) if features_raw else obs


# The features block (wc3hook/features.h): the model's inputs the hook computes after an observation, for
# training/policy (unit rows as features.py makes them, the map's tree grid, scalars, and decide.tables' masks).
FEATURES_MAGIC, FEATURES_VERSION = 0x31544657, 1
FEAT_FLOATS, FEAT_MAP, FEAT_HISTORY, FEAT_KINDS, FEAT_N_ARGS, FEAT_MAX_UNITS = 69, 128, 4, 21, 2828, 256
_FEAT_HEADER = struct.Struct("<7I4i4i24s")
_FEAT_ARG_BYTES = (FEAT_N_ARGS + 7) // 8
_FEAT_ARG_STRIDE = (_FEAT_ARG_BYTES + 3) & ~3


def parse_features(data: bytes) -> dict:
    """A features block as arrays: unit rows (`units` [n, 69] and the ids features.py names, n = unit_count),
    `trees` [128, 128], `scalars`, `previous_type`, `recent`, and the masks: `type_mask`, `argument_types` with
    `argument_bits` [k, 354] (packed), `actor_keys` with `actor_rows` [K, 32] (packed)."""
    h = _FEAT_HEADER.unpack_from(data)
    magic, version, size, n, scalar_count, argument_n, actor_n = h[:7]
    if magic != FEATURES_MAGIC or version != FEATURES_VERSION or size != len(data):
        raise ValueError("features block does not match this wc3env version")
    out = {"unit_count": n, "previous_type": np.array(h[7:11], np.int32), "recent": np.array(h[11:15], np.int32),
           "type_mask": np.frombuffer(h[15][:FEAT_KINDS], np.uint8).astype(bool)}  # fmt: skip
    at = _FEAT_HEADER.size

    def take(name, dtype, shape):
        nonlocal at
        count = int(np.prod(shape))
        out[name] = np.frombuffer(data, dtype, count, at).reshape(shape)
        at += (count * np.dtype(dtype).itemsize + 3) & ~3

    take("units", np.float32, (n, FEAT_FLOATS))
    for name in ("unit_type", "unit_owner", "unit_order", "unit_queue"):
        take(name, np.int32, (n,))
    take("unit_inventory", np.int32, (n, 6))
    take("unit_buffs", np.int32, (n, 8))
    take("unit_abilities", np.int32, (n, 5))
    take("unit_id", np.uint32, (n,))
    take("unit_cell", np.int32, (n,))
    take("trees", np.uint8, (FEAT_MAP, FEAT_MAP))
    take("scalars", np.float32, (scalar_count,))
    take("argument_types", np.int32, (argument_n,))
    take("argument_bits", np.uint8, (argument_n, _FEAT_ARG_STRIDE))
    out["argument_bits"] = out["argument_bits"][:, :_FEAT_ARG_BYTES]
    take("actor_keys", np.int32, (actor_n,))
    take("actor_rows", np.uint8, (actor_n, FEAT_MAX_UNITS // 8))
    return out


def parse(data: bytes) -> BinaryObservation:
    """One observation as the DLL wrote it: the header, then each table's records."""
    if len(data) < HEADER.size:
        raise ValueError("binary observation is shorter than its header")
    h = HEADER.unpack_from(data)
    magic, version, size, player, sequence, time_ms, gold, lumber, food_used, food_cap, result, lost, orders_lost = h[
        :13
    ]
    if magic != MAGIC or version != VERSION or size != len(data):
        raise ValueError("binary observation header does not match this wc3env version")
    tables = {}
    for i, name in enumerate(TABLES):
        offset, count, record_size = h[_TABLE_ENTRIES + 3 * i : _TABLE_ENTRIES + 3 * i + 3]
        if record_size != DTYPES[name].itemsize:
            raise ValueError(f"binary observation table {name} does not match this wc3env version")
        tables[name] = np.frombuffer(data, DTYPES[name], count, offset)
    return BinaryObservation(
        player=player,
        sequence=sequence,
        game_time_seconds=time_ms / 1000,
        gold=gold,
        lumber=lumber,
        food_used=food_used,
        food_cap=food_cap,
        result=RESULTS[result],
        events_lost=lost,
        orders_lost=orders_lost,
        time_of_day=h[_TIME_OF_DAY],
        score=dict(zip(SCORE_FIELDS, h[_SCORE])),
        raw=data,
        **tables,
    )
