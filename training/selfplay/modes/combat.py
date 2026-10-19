"""Fights (--mode combat): two mirrored Orc armies on the flat arena, a fight every 1-2 game minutes, for debugging the
RL loop in minutes instead of hours (docs/training/combat.md).

Each fight draws a new army: three of the four Orc heroes, each at a random level with a random legal skill build and
random items, and a food budget filled with random unit types. Both players get the same one, placed as each other's
mirror image about a random point of the open arena, facing along a random direction. The town halls stay (Warcraft
ends a melee game for a player with no buildings); the melee start's workers are removed; both players own
every Orc upgrade and have no resources. Every fight begins on a freshly loaded map, so its clock (one of the model's
inputs) starts where a whole game's does; it is staged in the game's worker with the step that starts it
(VectorSession.stage).

The learner is player 0, the local player, whose units cast only what it orders; the opponent is player 1, a computer
slot without an AI (wc3env loads every agent slot but the local player's so), whose units cast their spells on their
own, as Warcraft's computer players' do: a competent opponent the learner has to out-cast. Warcraft's attack-move
against itself won 36 of 40 fights as player 1, 1 as player 0, so the even stage is far harder than an even fight.
(cb1-cb5 had both players computer slots, 18-18: the learner's spells were Warcraft's, not its own.)

A fight ends when one side's army value (each unit's gold + lumber, a hero's times 1 + HERO_WORTH per level above 1,
times the square root of its share of its hit points) falls below DECIDED of the other's and stays below for HOLD_SECONDS (one burst of
damage does not end a fight that could still turn): the stronger side won, +1 and -1. At SECONDS it is a loss for
both (so running away never pays). Every step pays the army value traded: (the enemy's value lost - one's own) / the
army's starting value, in the critic's "dealt" and "taken" channels; summed over a fight that is the final margin
(potential-based shaping, Ng et al. 1999: it does not change which policy is best, and cannot be farmed), at most 1,
as large as the win; so it never fades (Combat.shaping).

The ladder's opponent is Warcraft's attack-move (one rung, "attack-move"): every ORDER_SECONDS each of its units is
told to attack-move to the centre of the other army, and picks its own targets on the way. Its stages (HANDICAPS)
start weak and end stronger than the learner: below 1 its units take every hit scaled by 1/h (the hook's handicap);
past 1 it is not handicapped but outnumbers the learner, h times the food in units (the same heroes; the extra units
drawn from the army's own types, behind and beside its rows). More units, not tougher ones: the policy sees them (a
hidden damage scale on either side would teach it a Warcraft that is not the real one). Beating an equal army is the
first milestone; beating a bigger one takes better fighting than Warcraft's.

    python -m tools.prepare.arena      # builds build/maps/(2)FlatArena.w3x and its map data, once
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import numpy as np

from wc3env.binary import DEAD, HERO, ILLUSION, OWN, STRUCTURE, fourcc
from wc3env.data import MAPS

from ...policy.actions import TYPES, reference
from ..reward import CHANNELS, TIME_LIMIT
from . import Episode, Step

# the ladder's stages: below 1 the opponent's hit points as a share of its own, past 1 its units' food as a multiple of
# the learner's. 0.35 first: bc12 won 5 of 5 fights at 0.05 (2026-10-12), and a stage won FAST_WINS of its first
# FAST_GAMES is passed at once (league.py)
# 0.85 to 1 in steps of 0.05 (cb1: 0.85 -> 1 took its ladder wins from ~73% to ~44%), above 1 in steps of 0.1 to 2
# (2026-10-12: 1 -> 1.25 was a quarter more army at once); 0.6 added for cb6's start (the learner without automatic casting)
HANDICAPS = (0.35, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
# the order types a fight allows (the rest masked before each decision): no economy, no shop, no altar. Sell, drop,
# revive and the like were legal on most steps (revive on 85% with no altar on the map). Learn: a hero levelling up
# mid-fight gets a skill point (skills up to its staged level are learned at staging)
FIGHT_TYPES = ("noop", "move", "attack_move", "smart", "attack", "stop", "hold", "cast", "cast_point", "cast_unit", "use_item", "learn")
FIGHT_MASK = np.array([t in FIGHT_TYPES for t in TYPES])
MAP = "Maps/FrozenThrone/(2)FlatArena.w3x"  # under the game's folder (training/cloud/selfplay.py puts it there)
DECIDED = 0.3  # a side whose army value falls below this share of the other's (for HOLD_SECONDS) has lost
HOLD_SECONDS = 2.0
SECONDS = 150.0  # game seconds a fight may last (with heroes, 90 cut off 14 of 40 Warcraft-vs-Warcraft fights; decided ones took 32-88)
ORDER_SECONDS = 3.0  # how often the scripted side is re-aimed
FOOD = (8, 40)  # an army's food besides its heroes, drawn uniformly (a Grunt is 3, a Tauren 5)
HEROES = ("Obla", "Ofar", "Otch", "Oshd")  # an army has three of them, each once
LEVELS = (1, 10)  # a hero's level, drawn uniformly for each
# a hero's items: one to heal, then more drawn from the rest, ITEMS in all (shop consumables, permanent items as
# creeps drop them; no town portals, sanctuary, invisibility or summons)
HEAL = ("phea", "hslv", "pghe")
POOL = ("pman", "plcl", "shea", "spro", "sreg", "pnvl", "pgma", "pres", "wneg", "will", "pdiv", "wcyc", "rde1", "rde2",
        "rst1", "rag1", "rin1", "gcel", "bspd", "cnob", "penr", "rlif", "evtl", "prvt", "oli2", "ofir", "oven", "belv",
        "bgst", "ciri", "rwiz", "afac", "ajen", "clfm", "hval", "hcun", "lhst", "mcou")  # fmt: skip
ITEMS = (3, 6)
HERO_WORTH = 0.25  # a hero's army value: its price times 1 + this per level above 1 (a guess: a level 5 Blademaster
# ~1,050, five Grunts' worth)
# the unit types an army is drawn from, and how deep behind the front line each stands (melee first, siege last)
ROWS = {"ogru": 0, "orai": 0, "otau": 0, **dict.fromkeys(HEROES, 150), "ohun": 250, "okod": 250, "owyv": 250, "otbr": 250, "oshm": 400,
        "odoc": 400, "ospw": 400, "ocat": 550}  # fmt: skip
SPACING, JITTER = 90.0, 50.0  # sideways gap between units in a row; each unit nudged up to this far
GAP = (250.0, 600.0)  # each army's front line stands this far from the fight's centre
MARGIN = 1800.0  # the centre stays this far inside the playable area
HALLS = 2600.0  # and this far from either start location (the town halls)
FULL_MANA = 100000.0  # above any unit's maximum: the game caps it


@cache
def arena() -> tuple[tuple, tuple]:
    """The arena's playable bounds (min_x, min_y, max_x, max_y) and its start locations ((x, y) each)."""
    data = json.loads((MAPS / "(2)FlatArena.json").read_text(encoding="utf-8"))
    b = data["bounds"]
    return (b["min_x"], b["min_y"], b["max_x"], b["max_y"]), tuple((s["x"], s["y"]) for s in data["start_locations"])


@cache
def _food(type_id: str) -> int:
    return int(reference()["units"][type_id]["food"])


@cache
def _price(type_id: int) -> float:
    u = reference()["units"].get(fourcc(int(type_id)), {})
    return float(u.get("gold", 0) + u.get("lumber", 0))


@cache
def _upgrades() -> tuple:
    """Every Orc upgrade at its last level."""
    return tuple((k, len(v["levels"])) for k, v in reference()["upgrades"].items() if v.get("race") == "orc")


def skills(rng: np.random.Generator, hero: str, level: int) -> list[str]:
    """A random legal skill build for a hero of `level`, one skill a level: a basic skill's k-th level needs hero level
    2k - 1 (at most 3), the ultimate hero level 6 (taken then)."""
    abilities = reference()["abilities"]
    options = reference()["units"][hero]["potential_hero_abilities"]
    taken: dict[str, int] = {}
    for at in range(1, level + 1):
        ok = [a for a in options if taken.get(a, 0) < abilities[a]["max_level"]
              and at >= abilities[a]["required_level"] + 2 * taken.get(a, 0)]  # fmt: skip
        ultimate = [a for a in ok if abilities[a]["required_level"] >= 6]
        pick = ultimate[0] if ultimate else ok[int(rng.integers(len(ok)))]
        taken[pick] = taken.get(pick, 0) + 1
        yield pick


def army(rng: np.random.Generator) -> list[tuple[str, dict]]:
    """A random army, [(type, what its spawn also gets)]: three heroes (level, skills, items), then unit types drawn
    uniformly until the food budget is spent."""
    out = []
    for hero in rng.permutation(HEROES)[:3]:
        level = int(rng.integers(LEVELS[0], LEVELS[1] + 1))
        items = [str(rng.choice(HEAL)), *(str(i) for i in rng.choice(POOL, int(rng.integers(*ITEMS, endpoint=True)) - 1, replace=False))]
        out.append((str(hero), {"level": level, "learn": list(skills(rng, str(hero), level)), "items": items}))
    budget, types = int(rng.integers(FOOD[0], FOOD[1] + 1)), [t for t in ROWS if t not in HEROES]
    while fits := [t for t in types if _food(t) <= budget]:
        pick = fits[int(rng.integers(len(fits)))]
        out.append((pick, {}))
        budget -= _food(pick)
    return out


def reinforcements(rng: np.random.Generator, units: list[tuple[str, dict]], share: float) -> list[tuple[str, dict]]:
    """Units for an army `share` (> 1) times the food of `units`' non-heroes, on average: more of its own types, drawn
    uniformly, and for the food left over that no unit fits, one more with that share of its food as its chance (a
    tenth more of an 8-food army is less than a Grunt: it comes 0.8 / 3 of the time)."""
    types = [t for t, _ in units if t not in HEROES]
    budget, out = (share - 1) * sum(_food(t) for t in types), []
    while fits := [t for t in types if _food(t) <= budget]:
        pick = fits[int(rng.integers(len(fits)))]
        out.append((pick, {}))
        budget -= _food(pick)
    if budget > 0 and types:
        pick = types[int(rng.integers(len(types)))]
        if rng.random() < budget / _food(pick):
            out.append((pick, {}))
    return out


def placements(rng: np.random.Generator, units: list[tuple[str, dict]], extra_units: list = (), extra_player: int = 1) -> list:
    """[(type, extra, player, x, y)] for both armies, player 1's player 0's mirror image about a random centre; then
    `extra_player`'s `extra_units` (reinforcements), each in its type's row, past the ends of the mirrored one."""
    (min_x, min_y, max_x, max_y), starts = arena()
    while True:
        centre = (rng.uniform(min_x + MARGIN, max_x - MARGIN), rng.uniform(min_y + MARGIN, max_y - MARGIN))
        if all(math.dist(centre, s) > HALLS for s in starts):
            break
    angle, gap = rng.uniform(0, 2 * math.pi), rng.uniform(*GAP)
    along, across = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
    rows: dict[float, list] = {}
    for unit in units:
        rows.setdefault(ROWS[unit[0]], []).append(unit)
    more: dict[float, list] = {}
    for unit in extra_units:
        more.setdefault(ROWS[unit[0]], []).append(unit)

    def offset(depth: float, side: float) -> tuple[float, float]:
        back = gap + depth + rng.uniform(-JITTER, JITTER)  # from the centre, away from the enemy
        side += rng.uniform(-JITTER, JITTER)
        return -back * along[0] + side * across[0], -back * along[1] + side * across[1]

    out = []
    for depth, types in rows.items():
        for k, (t, extra) in enumerate(types):
            dx, dy = offset(depth, (k - (len(types) - 1) / 2) * SPACING)
            out.append((t, extra, 0, centre[0] + dx, centre[1] + dy))
            out.append((t, extra, 1, centre[0] - dx, centre[1] - dy))  # the mirror image: same distances, facing back
    side = 1 if extra_player == 0 else -1  # player 0's army at the centre + its offsets, player 1's at the centre -
    for depth, types in more.items():
        half = len(rows.get(depth, ())) / 2
        for k, (t, extra) in enumerate(types):  # alternately right and left of the row, outwards
            dx, dy = offset(depth, (1 if k % 2 == 0 else -1) * (half + k // 2 + 0.5) * SPACING)
            out.append((t, extra, extra_player, centre[0] + side * dx, centre[1] + side * dy))
    return out


GOLD_PER_MANA = 4 / 3  # the shop's price of mana: a Potion of Mana 200 gold for 150, a Greater 400 for 300


@cache
def _summoned() -> dict[int, list[tuple[int, int, int]]]:
    """Each unit type a spell makes and that costs nothing (wards, wolves, traps) -> [(ability, level, units per cast)],
    from the abilities' data: its UnitID, the count from "Summons 2 ..." (else 1)."""
    out: dict[int, list] = {}
    for code, ability in reference()["abilities"].items():
        for at, level in ability["levels"].items():
            unit = level.get("effects", {}).get("UnitID")
            if unit and unit in reference()["units"] and not _price(int.from_bytes(unit.encode("latin-1"), "big")):
                n = re.search(r"[Ss]ummons (\d+)", level.get("description", ""))
                entry = (int.from_bytes(code.encode("latin-1"), "big"), int(at), int(n.group(1)) if n else 1)
                out.setdefault(int.from_bytes(unit.encode("latin-1"), "big"), []).append(entry)
    return out


@cache
def _illusions() -> list[tuple[int, int, int]]:
    """The abilities that make illusions of their caster (Mirror Image) -> [(ability, level, illusions per cast)]."""
    out = []
    for code, ability in reference()["abilities"].items():
        for at, level in ability["levels"].items():
            n = re.search(r"creating (\d+) illusions?", level.get("description", ""))
            if n:
                out.append((int.from_bytes(code.encode("latin-1"), "big"), int(at), int(n.group(1))))
    return out


def mana_costs(observations: dict, into: dict) -> dict:
    """Each ability's mana cost and level as the players' own units list it (ability id -> (mana, level)), added to
    `into`: a summon keeps its worth after its caster dies."""
    for obs in observations.values():
        a = obs.abilities[obs.abilities["mana_cost"] > 0]
        for ability, mana, level in zip(a["ability_id"].tolist(), a["mana_cost"].tolist(), a["level"].tolist()):
            into[int(ability)] = (float(mana), int(level))
    return into


def _cast_worth(type_id: int, illusion: bool, mana: dict) -> float:
    """What a unit made by a spell is worth: its spell's mana at the shop's price, shared by the units one cast
    makes at the level cast (0 when the spell's cost was never seen)."""
    makers = _illusions() if illusion else _summoned().get(int(type_id), [])
    known = [(mana[a][0], n, mana[a][1] == at) for a, at, n in makers if a in mana]
    costs = [m / n for m, n, same in known if same] or [m / n for m, n, _ in known]  # else at another level (cast before a level-up)
    return max(costs) * GOLD_PER_MANA if costs else 0.0


def _fighters(obs) -> np.ndarray:
    """The observer's living units that are not buildings."""
    u = obs.units
    return u[((u["flags"] & OWN) != 0) & ((u["flags"] & (DEAD | STRUCTURE)) == 0)]


def value(obs, mana: dict | None = None) -> float:
    """The observer's army value: each unit's gold + lumber (a hero's times 1 + HERO_WORTH per level above 1) times the
    square root of its share of its hit points (Lanchester: a group's strength goes as its hit points times its damage,
    and a unit deals full damage until it dies, so two Grunts at half health are worth 1.41 Grunts, not 1; with the
    share itself, as until 2026-10-12, spreading damage paid as much as killing, and focus fire nothing). A unit a
    spell made (a ward, a wolf, a Mirror Image illusion) is worth its spell's mana at the shop's price (mana_costs;
    until 2026-10-13 a ward or wolf 0, so killing one paid nothing, and an illusion a whole hero)."""
    u = _fighters(obs)
    mana = mana or {}
    illusion = (u["flags"] & ILLUSION) != 0
    price = np.array([_cast_worth(t, True, mana) if i else _price(t) or _cast_worth(t, False, mana)
                      for t, i in zip(u["type_id"], illusion)])  # fmt: skip
    worth = price * np.where((u["flags"] & HERO) & ~illusion, 1 + HERO_WORTH * (u["level"] - 1), 1)
    return float((worth * np.sqrt(np.clip(u["hp"] / np.maximum(u["max_hp"], 1), 0, 1))).sum())


def stage_ops(observations: dict, rng: np.random.Generator, handicap: float | None, scripted: int = 1) -> list:
    """The debug ops staging the next fight in this game (wc3env.vector's VectorSession.stage): the last fight's units
    (and the melee start's workers) removed, the upgrades, no resources, both armies (heroes at their level with their
    skills learned, every unit at full mana), and the scripted side's (slot `scripted`) stage (HANDICAPS; None: no
    scripted side): below 1 its handicap, past 1 its reinforcements."""
    gone = {int(i) for obs in observations.values() for i in _fighters(obs)["unit_id"]}
    ops = [("remove", {"unit_id": i}) for i in sorted(gone)]
    for player in (0, 1):
        ops.append(("resources", {"player": player, "gold": 0, "lumber": 0}))
        ops += [("research", {"player": player, "type_id": t, "level": n}) for t, n in _upgrades()]
    units = army(rng)
    extra_units = reinforcements(rng, units, handicap) if handicap is not None and handicap > 1 else []
    ops += [("spawn", {"type_id": t, "player": p, "x": x, "y": y, "n": 1, **extra, "mana": FULL_MANA})
            for t, extra, p, x, y in placements(rng, units, extra_units, scripted)]  # fmt: skip
    if handicap is not None:  # 1: off (it stays on for units made later)
        ops.append(("handicap", {"player": scripted, "hp": float(min(handicap, 1.0))}))
    return ops


def attack_move(own, enemy) -> list[dict]:
    """The scripted side's order: every unit of `own` attack-moves to the centre of `enemy`'s army (its observation)."""
    them, us = _fighters(enemy), _fighters(own)
    if not len(them) or not len(us):
        return []
    at = {"x": float(them["x"].mean()), "y": float(them["y"].mean())}
    return [{"unit_id": int(i), "command": "attack", "arguments": dict(at)} for i in us["unit_id"]]


class Score:
    """A fight's army values step by step: what each side lost, and who has won."""

    def __init__(self, observations: dict):
        self.mana = mana_costs(observations, {})  # each spell's mana cost, as the fight's casters list them
        self.values = {p: value(observations[p], self.mana) for p in (0, 1)}
        self.start = max(sum(self.values.values()) / 2, 1.0)  # an army's starting value (the two's mean: one fight's
        # constant, so the trade still sums to the final margin when the ladder's opponent has more)
        self.leader, self.since = None, observations[0].game_time_seconds

    def step(self, observations: dict) -> tuple[dict, int | None]:
        """Each player's army value lost since the last step, as a share of the starting value (healing: negative);
        and the winner, once one side has been below DECIDED of the other's for HOLD_SECONDS (None: still fighting)."""
        mana_costs(observations, self.mana)
        now = {p: value(observations[p], self.mana) for p in (0, 1)}
        lost = {p: (self.values[p] - now[p]) / self.start for p in (0, 1)}
        self.values = now
        leader = 1 if now[0] < DECIDED * now[1] else 0 if now[1] < DECIDED * now[0] else None
        t = observations[0].game_time_seconds
        if leader != self.leader:
            self.leader, self.since = leader, t
        return lost, leader if leader is not None and t - self.since >= HOLD_SECONDS else None


def allow(block: dict) -> None:
    """A decision's feature block with only FIGHT_TYPES legal (in place: sampling and the recorded masks both read it)."""
    m = block["type_mask"]
    block["type_mask"] = m & np.pad(FIGHT_MASK, (0, max(len(m) - len(FIGHT_MASK), 0)))[: len(m)]


class Combat:
    map = MAP
    ladder, handicaps = ("attack-move",), HANDICAPS
    allow = staticmethod(allow)  # the actor applies it to every decision's features
    types = FIGHT_MASK  # the order types a fight can use (the learner's exploration floor holds only these)

    def pairs(self, races) -> list[tuple[str, str]]:
        return [("orc", "orc")]

    def plan(self, actor, race: str, other: str) -> None:
        return None  # a fight has no build to follow

    def config(self, game, rung: str) -> dict:
        return {}  # the learner the local player, casting only what it orders; the opponent's units on their own

    def episode(self, actor, game) -> Episode:
        return Fight(actor, game)

    def shaping(self, fade: float) -> float:
        return 1.0  # the army value traded: nothing to farm, so it never fades


class Fight(Episode):
    def __init__(self, actor, game):
        self.actor, self.game = actor, game
        self.learning = [slot for slot, p in game.policy.items() if p == "latest"]
        self.began, self.last_order, self.score = 0.0, -1e9, None

    def start(self, i: int, obs: dict, session) -> None:
        handicap = self.actor.handicap(self.game) if self.game.seatless else None
        rng = self.game.rng if self.game.rng is not None else self.actor.rng  # an evaluation round's own draws
        session.stage(i, stage_ops(obs, rng, handicap, self.game.other), {slot: [] for slot in obs})

    def staged(self, obs: dict) -> None:
        self.began, self.score = obs[0].game_time_seconds, Score(obs)

    def commands(self, obs: dict) -> dict:
        now = obs[0].game_time_seconds
        if not self.game.seatless or now - self.last_order < ORDER_SECONDS:
            return {}
        self.last_order = now
        us, them = self.game.slot, self.game.other
        return {them: attack_move(obs[them], obs[us])}

    def step(self, final: dict, done: bool, steps: int) -> Step:
        lost, winner = self.score.step(final)
        winner = {"victory": 0, "defeat": 1}.get(final[0].result) if done else winner
        now = final[0].game_time_seconds
        too_long = winner is None and not done and now - self.began >= SECONDS
        rewards = {}
        for slot in self.learning:
            r = np.zeros(len(CHANNELS), np.float32)
            r[CHANNELS.index("win")] = TIME_LIMIT if too_long else 0.0 if winner is None else 1.0 if winner == slot else -1.0
            r[CHANNELS.index("dealt")], r[CHANNELS.index("taken")] = lost[1 - slot], -lost[slot]
            rewards[slot] = r
        outcome = "time limit" if too_long else "draw" if winner is None else "victory" if winner == self.game.slot else "defeat"
        return Step(rewards, done or too_long or winner is not None, outcome, (now - self.began) / 60,
                    reload=not done)  # fmt: skip
