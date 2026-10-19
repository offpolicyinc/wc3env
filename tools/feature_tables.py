"""wc3hook/features_table.h from src/wc3env/data/reference.json: the lookups the hook's features and masks need
(docs/training/12-hook-features.md), in the order training's embeddings and argument indices use.

    python tools/feature_tables.py

Each table maps a type, ability, buff or upgrade id (its four characters as a big-endian integer) to its embedding
id: sorted by id for binary search, the embedding id being the code's place in training's sorted code list
(training/policy/features.py: type_ids, ability_ids, buff_ids, upgrade_ids).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from training.policy.actions import (  # noqa: E402  the action types, in order
    ANY_ARGUMENT,
    HAS_ARGUMENT,
    PRODUCTION,
    TYPES,
)
from training.policy.masks import ORDER_IDS, TYPE_ORDERS, argument_ids  # noqa: E402

REFERENCE = ROOT / "src" / "wc3env" / "data" / "reference.json"
OUT = ROOT / "wc3hook" / "features_table.h"


def _id(code: str) -> int:
    return int.from_bytes(code.encode("latin-1"), "big")


def _table(name: str, codes: list[str], first: int) -> str:
    pairs = sorted((_id(c), i + first) for i, c in enumerate(codes))
    rows = "\n".join(f"    {{{key}u, {value}}}," for key, value in pairs)
    return f"static const FeatId {name}[] = {{\n{rows}\n}};\n#define {name}_N {len(pairs)}\n"


def _cost(units: dict, code: str) -> tuple[int, int, int]:
    """masks._cost: gold, lumber, food; a town hall upgrade costs the difference."""
    u = units[code]
    gold, lumber = u.get("gold", 0), u.get("lumber", 0)
    for base, info in units.items():
        if code in info.get("upgrades_to", []):
            gold, lumber = gold - units[base].get("gold", 0), lumber - units[base].get("lumber", 0)
    return gold, lumber, u.get("food", 0) or 0


def _masks(ref: dict) -> str:
    """The tech tree for masks.py's checks; every list lives in FT_POOL as (offset, count). Requirements that are
    not unit types are left out: masks.py counts them as met."""
    units, upgrades, abilities = ref["units"], ref["upgrades"], ref["abilities"]
    pool: list[int] = []

    def put(values) -> str:
        start = len(pool)
        pool.extend(values)
        return "{%d, %d}" % (start, len(values))

    def codes(lst) -> list[int]:
        return [_id(c) for c in lst]

    def known(lst) -> list[int]:
        return [_id(c) for c in lst if c in units]

    groups: list[str] = []  # each hero's requires_by_count lists
    rows = []
    for code in sorted(units, key=_id):
        u = units[code]
        acting = bool(u["structure"] and (u.get("base_move_speed") or u.get("damage")))
        worker = bool(u.get("builds")) and not u["structure"]
        flags = (1 if u["structure"] else 0) | (2 if u.get("hero") else 0) | (4 if acting else 0) | (8 if worker else 0)
        gold, lumber, food = _cost(units, code)
        first_group = len(groups)
        for g in u.get("requires_by_count") or []:
            groups.append(put(known(g)))
        fields = [
            "%du" % _id(code),
            str(flags),
            str(gold),
            str(lumber),
            str(food),
            put(known(u.get("requires", []))),
            str(first_group),
            str(len(groups) - first_group),
            put(codes(u.get("trains", []))),
            put(codes(u.get("researches", []))),
            put(codes(u.get("upgrades_to", []))),
            put(codes(u.get("builds", []))),
            put(codes(u.get("potential_hero_abilities", []))),
        ]
        rows.append("    {" + ", ".join(fields) + "},")
    up_levels, up_rows = [], []
    for code in sorted(upgrades, key=_id):
        levels = upgrades[code]["levels"]
        up_rows.append("    {%du, %d, %d}," % (_id(code), len(up_levels), len(levels)))
        for lv in levels:
            up_levels.append(
                "    {%d, %d, %s}," % (lv.get("gold", 0), lv.get("lumber", 0), put(known(lv.get("requires", []))))
            )
    ab_rows, ab_levels, orders = [], [], []
    for code in sorted(abilities, key=_id):
        a = abilities[code]
        first = len(ab_levels)
        for key, level in a["levels"].items():
            given = [o for o in level.get("orders", []) if o.get("order_id") is not None]
            ab_levels.append("    {%d, %d, %d}," % (int(key), len(orders), len(given)))
            orders += ["    {%du, %d}," % (o["order_id"], 1 if o["kind"] == "cast" else 0) for o in given]
        skip = a.get("level_skip") or 2
        ab_rows.append("    {%du, %d, %d, %d, %d, %d}," % (_id(code), a.get("max_level", 3), a.get("required_level", 1),
                                                         skip, first, len(ab_levels) - first))  # fmt: skip
    type_orders = json.loads(TYPE_ORDERS.read_text()) if TYPE_ORDERS.is_file() else {}
    to_rows = [
        "    {%du, %s}," % (_id(c), put(sorted(v))) for c, v in sorted(type_orders.items(), key=lambda kv: _id(kv[0]))
    ]
    given_orders = {o["order_id"] for a in abilities.values() for lv in a["levels"].values()
                    for o in lv.get("orders", []) if o.get("order_id") is not None}  # fmt: skip
    unknown = [o for o in ORDER_IDS if o not in given_orders]
    args = sorted(argument_ids().items())
    kinds = []
    for t in TYPES:
        f = (
            (1 if t in HAS_ARGUMENT else 0)
            | (2 if t != "noop" and t not in PRODUCTION else 0)
            | (4 if t in ANY_ARGUMENT else 0)
        )
        kinds.append('    {"%s", %d},' % (t, f))
    return "\n".join(
        [
            "/* ---- masks: the tech tree (training/policy/masks.py) */",
            "typedef struct { int offset, count; } FtList; /* into FT_POOL */",
            "#define FT_STRUCTURE 1",
            "#define FT_HERO 2",
            "#define FT_ACTING 4 /* a structure that moves or attacks: it takes unit orders */",
            "#define FT_WORKER 8",
            "typedef struct { unsigned id; int flags, gold, lumber, food; FtList requires; int groups, groups_n;"
            " FtList trains, researches, upgrades_to, builds, hero_abilities; } FtUnit;",
            "typedef struct { int gold, lumber; FtList requires; } FtUpgradeLevel;",
            "typedef struct { unsigned id; int levels, levels_n; } FtUpgrade;",
            "typedef struct { int level, orders, orders_n; } FtAbilityLevel;",
            "typedef struct { unsigned id; int max_level, required_level, level_skip, levels, levels_n; } FtAbility;",
            "typedef struct { unsigned order; int cast; } FtOrder;",
            "typedef struct { unsigned id; FtList orders; } FtTypeOrders;",
            "typedef struct { const char *name; int flags; } FtKind; /* 1 takes an argument, 2 has an actor, 4 any argument */",
            "static const int FT_POOL[] = {" + ", ".join(map(str, pool)) + "};",
            "static const FtList FT_GROUPS[] = {" + ", ".join(groups) + "};",
            "static const FtUnit FT_UNITS[] = {",
            *rows,
            "};",
            "#define FT_UNITS_N %d" % len(rows),
            "static const FtUpgradeLevel FT_UPGRADE_LEVELS[] = {",
            *up_levels,
            "};",
            "static const FtUpgrade FT_UPGRADE_INFO[] = {",
            *up_rows,
            "};",
            "#define FT_UPGRADE_INFO_N %d" % len(up_rows),
            "static const FtAbilityLevel FT_ABILITY_LEVELS[] = {",
            *ab_levels,
            "};",
            "static const FtOrder FT_ORDERS[] = {",
            *orders,
            "};",
            "static const FtAbility FT_ABILITY_INFO[] = {",
            *ab_rows,
            "};",
            "#define FT_ABILITY_INFO_N %d" % len(ab_rows),
            "static const FtTypeOrders FT_TYPE_ORDERS[] = {",
            *to_rows,
            "};",
            "#define FT_TYPE_ORDERS_N %d" % len(to_rows),
            "#define FT_ORDER_FIRST %d" % ORDER_IDS.start,
            "#define FT_ORDER_COUNT %d" % len(ORDER_IDS),
            "/* orders no ability in the reference gives */",
            "static const unsigned FT_UNKNOWN_ORDERS[] = {" + ", ".join("%du" % o for o in unknown) + "};",
            "#define FT_UNKNOWN_ORDERS_N %d" % len(unknown),
            "/* argument id -> argument index (masks.argument_ids), sorted by id */",
            "static const FeatId FT_ARGUMENTS[] = {",
            *["    {%du, %d}," % (k, v) for k, v in args],
            "};",
            "#define FT_ARGUMENTS_N %d" % len(args),
            "#define FT_N_ARGS %d" % (len(args) + 1),
            "static const FtKind FT_KINDS[] = {",
            *kinds,
            "};",
            "#define FT_KINDS_N %d" % len(kinds),
        ]
    )


def main() -> None:
    ref = json.loads(REFERENCE.read_text())
    units, items = sorted(ref["units"]), sorted(ref["items"])
    workers = sorted(_id(c) for c, u in ref["units"].items() if u.get("builds") and not u["structure"])
    parts = [
        "/* generated by tools/feature_tables.py from src/wc3env/data/reference.json: id -> embedding id, sorted by id */",
        "#pragma once",
        "typedef struct { unsigned id; int embedding; } FeatId;",
        _table("FT_TYPES", units + items, 1),
        _table("FT_ABILITIES", sorted(ref["abilities"]), 1),
        _table("FT_BUFFS", sorted(ref["buffs"]), 1),
        _table("FT_UPGRADES", sorted(ref["upgrades"]), 0),
        "/* units that build structures (workers) */",
        "static const unsigned FT_WORKERS[] = {" + ", ".join(f"{w}u" for w in workers) + "};",
        f"#define FT_WORKERS_N {len(workers)}",
        _masks(ref),
        "",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(units) + len(items)} types, {len(ref['abilities'])} abilities, "
        f"{len(ref['buffs'])} buffs, {len(ref['upgrades'])} upgrades, {len(workers)} workers"
    )


if __name__ == "__main__":
    main()
