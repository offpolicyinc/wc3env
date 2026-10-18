"""Replay commands to actions (training/policy/actions.py), for imitation.

A replay played with observations gives, each step, the orders the player's commands gave its units (the
observation's `orders` rows with origin `player`): one row per unit a command reached. One command is one
action: its rows share the game time, order, target and point. The command's units give the anchor (the one
nearest their middle, or another of them whose nearest units are exactly the command's), same_type and size; its order gives the type and argument:

    move / attack / smart / stop / hold / setrally / cancel     the basic orders
    a unit, structure or upgrade type                           train, build, research (an upgrade or a
                                                                town hall tier)
    a hero ability's id                                         learn
    revive (the altar's order on its dead hero)                 revive
    an ability's order                                          cast, cast_point, cast_unit
    useslot1-6, dropitem                                        use_item, drop_item, sell (onto a shop)

Shop purchases are orders to the shop's (neutral) owner, so they come from the item_sold / unit_sold events
that name the player's buyer.

A command that changed nothing is dropped: the units were already doing exactly that (repeat clicks), or
production that never started (a full queue, not enough gold). Players also spam-click with a building
selected (half the rallies of a game come a median 150 ms after the last): a burst of rally clicks on the same
buildings, each within RALLY_BURST_MS of the one before, is one action, the last click's. What is dropped is reported, not silently
lost: `convert` returns every command with its action, and the reason when it is dropped or has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np

from wc3env.binary import EVENT_KINDS, IMMEDIATE, ORIGINS, OWN, POINT, STRUCTURE, TARGET, fourcc

from ..policy.actions import PRODUCTION, QUEUEABLE, SINGLE, Action, producer, reference, select, size_for

STOP, SETRALLY, ATTACK, ATTACK_GROUND, MOVE, PATROL, HOLD = 851972, 851980, 851983, 851984, 851986, 851990, 851993
SMART, CANCEL, DROP_ITEM, REVIVE, AI_MOVE = 851971, 851976, 852001, 852039, 851988
MOVE_SLOT, USE_SLOT = range(852002, 852008), range(852008, 852014)


@dataclass
class Command:
    """Every row of one player command: the units it reached, in order."""

    time_ms: int
    order_id: int
    kind: int
    target: int
    x: float
    y: float
    queued: bool
    item_type: int  # the item a give or drop moves
    units: list[int] = field(default_factory=list)


@dataclass
class Converted:
    command: Command | None  # None for a shop purchase (from an event)
    action: Action | None
    reason: str = ""  # why there is no action, or "" when there is
    selected: list[int] = field(default_factory=list)  # what select(action) gives back for a unit action


ARMY_ORDERS = (MOVE, AI_MOVE, ATTACK)


def commands(obs, structures: set[int], origin: str = "player") -> list[Command]:
    """The player's commands in an observation's orders, oldest first. A command to structures and units at once
    (right-click with a town hall and its workers selected) is two: a rally and an order.

    Warcraft's AI (`script`) moves its army through its captains, whose orders the hook sees as the engine's: they
    are taken too when a move or attack reaches several units at once or goes to a point. A single unit attacking a
    unit on the engine's say is its own reaction to an enemy nearby (auto-acquire), not a command, and is left out
    (in an AI game's 14 minutes: 380 army orders from the captains, against about 10 from the script itself)."""
    out: dict[tuple, Command] = {}
    captains: set[tuple] = set()
    for r in obs.orders:
        source = ORIGINS[r["origin"]]
        captain = origin == "script" and source == "engine" and int(r["order_id"]) in ARMY_ORDERS
        if source != origin and not captain:
            continue
        kind, unit = int(r["kind"]), int(r["unit_id"])
        x, y = (float(r["x"]), float(r["y"])) if kind != IMMEDIATE else (0.0, 0.0)
        key = (int(r["time_ms"]), int(r["order_id"]), kind, int(r["target_id"]), round(x), round(y), int(r["queued"]))
        key += (int(r["item_type"]), unit in structures)
        if key not in out:
            out[key] = Command(key[0], key[1], kind, key[3], x, y, bool(key[6]), key[7])
            if captain:
                captains.add(key)
        out[key].units.append(unit)
    return [c for key, c in out.items() if key not in captains or len(c.units) > 1 or c.kind == POINT]


RALLY_BURST_MS = 2000


def convert(before, after, origin: str = "player", rallies: dict | None = None) -> list[Converted]:
    """The commands during one step: `before` is the observation at the step's start, `after` the one at its end
    (whose orders and events are what happened during the step). `origin` is whose orders: `player` for a
    player's commands (replays), `script` for Warcraft's own AI, which orders its units one by one; orders
    given at the same moment with the same order, target and point are grouped into one command either way.
    `rallies`, kept by the caller across a player's steps, holds each building group's latest rally until
    RALLY_BURST_MS pass without another: a newer click replaces it, so a burst gives one rally, late by up to
    RALLY_BURST_MS (without `rallies`, every rally click is its own action)."""
    units = {int(u["unit_id"]): u for u in after.units}
    units.update({int(u["unit_id"]): u for u in before.units})  # positions and orders as the step began
    structures = {i for i, u in units.items() if u["flags"] & STRUCTURE}
    out = [_command(c, before, after, units, origin) for c in commands(after, structures, origin)]
    if rallies is not None:
        out = _rally_bursts(out, rallies, int(after.game_time_seconds * 1000))
    for e in after.events:
        kind = EVENT_KINDS.get(int(e["kind"]))
        buyer = units.get(int(e["other_id"]))
        if kind in ("item_sold", "unit_sold") and buyer is not None and buyer["flags"] & OWN:
            action = Action("buy", argument=int(e["type_id"]), actor=int(e["other_id"]), target=int(e["unit_id"]))
            out.append(Converted(None, action, selected=[action.actor]))
    return out


def _command(c: Command, before, after, units: dict, origin: str = "player") -> Converted:
    rows = [units[u] for u in c.units if u in units]
    if not rows:
        return Converted(c, None, "units not observed")
    xs, ys = np.array([float(r["x"]) for r in rows]), np.array([float(r["y"]) for r in rows])
    anchor = rows[int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))]
    structure = bool(anchor["flags"] & STRUCTURE)
    point = dict(x=c.x, y=c.y) if c.kind == POINT else {}
    target = dict(target=c.target) if c.kind == TARGET else {}
    kind, argument, reason = _type(c, anchor, structure, before, units, origin)
    if kind is None:
        return Converted(c, None, reason)
    if kind in PRODUCTION:  # the translator picks the least busy building
        action = Action(kind, argument=argument)
        if not producer(before, argument, kind):
            return Converted(c, action, "no building for it in the step's first observation")
        return Converted(c, action, "" if _started(kind, argument, after) else "never started")
    same_type = all(int(r["type_id"]) == int(anchor["type_id"]) for r in rows)
    single = kind in SINGLE
    action = Action(
        kind,
        argument=argument,
        actor=int(anchor["unit_id"]),
        same_type=same_type,
        size=1 if single else size_for(len(rows)),
        queued=c.queued and kind in QUEUEABLE,
        **(point if kind not in ("smart", "attack", "cast_unit", "sell") else {}),
        **target,
    )
    selected = select(action, before)
    if not selected:  # e.g. a unit trained during the step and commanded at once: the policy could not see it
        return Converted(c, action, "actor not in the step's first observation")
    if not single and set(selected) != set(c.units):  # another of the commanded units may anchor it exactly
        order = np.argsort((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        for r in (rows[k] for k in order[1:]):
            other = replace(action, actor=int(r["unit_id"]))
            if set(select(other, before)) == set(c.units):
                action, selected = other, select(other, before)
                break
    if kind == "learn" and not _event(after, "hero_learn", argument):
        return Converted(c, action, "never learned", selected)
    if not _changed(c, rows):
        return Converted(c, action, "already doing it", selected)
    return Converted(c, action, "", selected)


def _rally_bursts(out: list[Converted], rallies: dict, now_ms: int) -> list[Converted]:
    kept = []
    for c in out:
        if c.action is None or c.action.type != "rally" or c.reason:
            kept.append(c)
            continue
        key = tuple(sorted(c.command.units))
        held = rallies.get(key)
        if held is not None and c.command.time_ms - held.command.time_ms > RALLY_BURST_MS:
            kept.append(held)  # a burst that ended before this click
        rallies[key] = c  # a newer click of the same burst replaces the held one
    for key, held in list(rallies.items()):
        if now_ms - held.command.time_ms > RALLY_BURST_MS:
            kept.append(held)
            del rallies[key]
    return kept


def _type(
    c: Command, anchor, structure: bool, before, units: dict, origin: str = "player"
) -> tuple[str | None, int, str]:
    """The action type and argument of a command, or (None, 0, why not)."""
    o, k = c.order_id, c.kind
    if o == STOP:
        return "stop", 0, ""
    if o == HOLD:
        return "hold", 0, ""
    if o == CANCEL:
        return "cancel", 0, ""
    if o == SETRALLY or (o == SMART and structure):
        return "rally", 0, ""
    if o == SMART:
        return ("smart" if k == TARGET else "move"), 0, ""
    if o in (MOVE, AI_MOVE):  # the AI's own move
        return ("smart" if k == TARGET else "move"), 0, ""  # a move onto a unit follows it, as right-click does
    if o == ATTACK:
        return ("attack" if k == TARGET else "attack_move"), 0, ""
    if o == REVIVE:  # the altar's target order on its dead hero
        hero = before.units[before.units["unit_id"] == c.target]
        return ("revive", int(hero["type_id"][0]), "") if len(hero) else (None, 0, "revive of an unseen hero")
    if o in USE_SLOT:
        item = _slot_item(before, int(anchor["unit_id"]), o - USE_SLOT.start)
        return ("use_item", item, "") if item else (None, 0, "empty slot")
    if o == DROP_ITEM:  # onto a unit or the ground; onto someone else's building, a shop buys it
        target = units.get(c.target) if k == TARGET else None
        shop = target is not None and target["flags"] & STRUCTURE and not target["flags"] & OWN
        return ("sell" if shop else "drop_item"), c.item_type, ""
    if o in MOVE_SLOT:
        return None, 0, "moveslot (left out)"
    if o in (PATROL, ATTACK_GROUND):
        return None, 0, "patrol or attackground (left out)"
    if o >> 24:
        code = fourcc(o)
        ref = reference()
        if code in ref["units"]:
            unit = ref["units"][code]
            if unit["structure"] and not structure:
                return "build", o, ""
            if structure and code in ref["units"].get(fourcc(anchor["type_id"]), {}).get("upgrades_to", []):
                return "research", o, ""
            return "train", o, ""
        if code in ref["upgrades"]:
            return "research", o, ""
        if code in ref["abilities"]:
            return "learn", o, ""
        if code in ref["items"]:  # a sale from the player's own shop: the item_sold event names the buyer
            return None, 0, "own shop's sale (a buy, from item_sold)"
        return None, 0, f"unknown type {code}"
    # every other order is an ability's (reference.json lists most; creep abilities and the return-resources
    # button are orders too). Warcraft's AI also gives itself orders no player could (851974, every unit every
    # 16 s): an AI order counts only if an ability gives it or a player was seen giving it.
    if origin == "script" and o not in _issuable():
        return None, 0, "AI-internal order"
    return {IMMEDIATE: "cast", POINT: "cast_point", TARGET: "cast_unit"}[k], o, ""


@cache
def _issuable() -> frozenset[int]:
    """Orders a player can give beyond the basic ones: any ability's, and those pros gave (type_orders.json)."""
    from ..policy.masks import RESUME_HARVESTING, ability_orders, type_orders

    return frozenset(ability_orders() | {o for orders in type_orders().values() for o in orders} | {RESUME_HARVESTING})


def _slot_item(obs, unit_id: int, slot: int) -> int:
    inv = obs.inventory
    hit = inv[(inv["unit_id"] == unit_id) & (inv["slot"] == slot)]
    return int(hit["type_id"][0]) if len(hit) else 0


def _event(obs, kind: str, type_id: int) -> bool:
    ids = [k for k, name in EVENT_KINDS.items() if name == kind]
    e = obs.events
    return bool((np.isin(e["kind"], ids) & (e["type_id"] == type_id)).any())


def _started(kind: str, type_id: int, after) -> bool:
    if kind == "train":
        return _event(after, "train_start", type_id)
    if kind == "research":
        return _event(after, "research_start", type_id) or any(_event(after, "upgrade_start", t) for t in upgraded_from(type_id))
    return True  # a revive has no start event


@cache
def upgraded_from(type_id: int) -> tuple[int, ...]:
    """The structure types that upgrade into this one (Keep: the Town Hall). A building's upgrade_start event names
    it as it was (wc3hook/events.c), its upgrade_finish as it became: matched against what it becomes, every hall and
    tower upgrade was dropped as never started (0 of 675 pro replays' plans had one)."""
    code = fourcc(type_id)
    return tuple(int.from_bytes(c.encode("latin-1"), "big") for c, u in reference()["units"].items() if code in u.get("upgrades_to", []))


def _changed(c: Command, rows) -> bool:
    """False when every unit was already carrying out this exact order as the step began (a repeat click)."""
    if c.queued:
        return True  # added to the queue, behind the current order
    for r in rows:
        current = int(r["order_id"])
        if c.order_id == SMART and c.kind == TARGET:
            same = int(r["order_target"]) == c.target  # a right-click becomes harvest, attack, follow...
        elif c.kind == TARGET:
            same = current == c.order_id and int(r["order_target"]) == c.target
        elif c.kind == POINT:
            same = current == c.order_id and abs(float(r["order_x"]) - c.x) < 1 and abs(float(r["order_y"]) - c.y) < 1
        else:
            same = current == c.order_id or (c.order_id == STOP and current == 0)
        if not same:
            return True
    return False
