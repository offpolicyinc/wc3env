"""Scenarios (drills): small fights that each need one skill, on combat mode's arena (docs/training/todo/curriculum.md:
its design section and drill catalog). A scenario swaps four parts of combat mode's fight and keeps the rest (the
network, the actions, the reward: the win and the army value traded):

    stage_ops(obs, rng, level, scripted)    the debug ops that stage a fight at this level (VectorSession.stage)
    opponent(own, enemy, level)             the scripted opponent's commands (its observation, then the learner's)
    solution(own, enemy, level)             the hand rule for the learner's side: the bar a level is passed at
    baseline(own, enemy, level)             a script without the skill: the solution must beat it, or the skill does
                                            not pay here and the scenario is not built (the gate: E40-E48)

Commands are re-issued only when they change: an attack order given again every step would restart the attack. A
script leaves its own heroes on attack-move: a direct order keeps a computer slot's hero from casting (E43).
Drills that failed their gate (regroup, protect_hero, kite, surround) are in rsi/log.md E46-E48, not here.
"""

from __future__ import annotations

import numpy as np

from wc3env.binary import HERO, fourcc

from ..policy.actions import reference
from .modes import combat
from .modes.combat import FULL_MANA, _fighters, _summoned, attack_move, base_ops, placements, skills

MELEE_REACH = 250.0  # how far a melee unit looks for a target (a few steps); a ranged one its range and RANGED_EXTRA
RANGED_EXTRA = 150.0


class Orders:
    """A script's commands, sent only for units whose order changed since the last time (or that have none)."""

    def __init__(self):
        self.given: dict[int, tuple] = {}

    def __call__(self, wanted: dict[int, dict]) -> list[dict]:
        out = []
        for unit, command in wanted.items():
            key = (command["command"], tuple(sorted(command["arguments"].items())))
            if self.given.get(unit) != key:
                self.given[unit] = key
                out.append({"unit_id": unit, **command})
        return out


def staged(observations: dict, rng: np.random.Generator, army: list, scripted: int, theirs: list | None = None) -> list:
    """The ops of a mirrored fight of `army` ([(type, its spawn's extras)]) on combat's arena; with `theirs`, the
    scripted side's own army in place of the mirror."""
    sides = (army, theirs) if scripted == 1 else (theirs, army)
    other = None if theirs is None else sides[1]
    return base_ops(observations) + [
        ("spawn", {"type_id": t, "player": p, "x": x, "y": y, "n": 1, **extra, "mana": FULL_MANA})
        for t, extra, p, x, y in placements(rng, sides[0] if theirs is not None else army, (), scripted, other)
    ]


def hero(code: str, level: int, learn: list[str]) -> tuple[str, dict]:
    """A hero at `level` with these skills learned (a fixed build: the drill is about one of them)."""
    return code, {"level": level, "learn": learn, "items": []}


def unit_stats(type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each unit's (damage per second, attack range) from the game's tables (damage: [low, high] a hit)."""
    units = reference()["units"]
    rows = [units.get(fourcc(int(t)), {}) for t in type_ids]
    dps = np.array(
        [np.mean(r.get("damage") or [0]) / max(r.get("base_attack_period") or 1, 0.1) for r in rows], np.float32
    )
    return dps, np.array([r.get("base_attack_range") or 100 for r in rows], np.float32)


def reaches(us) -> np.ndarray:
    """How far each unit looks for a target: MELEE_REACH, or a ranged unit's range and RANGED_EXTRA."""
    _, reach = unit_stats(us["type_id"])
    return np.where(reach > 150, reach + RANGED_EXTRA, MELEE_REACH)


def gaps(us, them) -> np.ndarray:
    """[ours, theirs] distances."""
    return np.hypot(us["x"][:, None] - them["x"][None, :], us["y"][:, None] - them["y"][None, :])


def each_attacks(us, them, reach, score: np.ndarray, orders: Orders, skip=None) -> list[dict]:
    """Each of `us` attacks the enemy with the highest `score` within its `reach` (a number or one a unit; an enemy
    scored -inf is no candidate; none in reach: the nearest); units where `skip` is True are left alone."""
    gap, reach = gaps(us, them), np.broadcast_to(reach, len(us))
    wanted = {}
    for k, unit in enumerate(us["unit_id"]):
        if skip is not None and skip[k]:
            continue
        near = (gap[k] <= reach[k]) & np.isfinite(score)
        pick = np.argmax(np.where(near, score, -np.inf)) if near.any() else np.argmin(gap[k])
        wanted[int(unit)] = {"command": "attack", "arguments": {"target_id": int(them["unit_id"][pick])}}
    return orders(wanted)


def heroes_attack_move(us, them, orders: Orders, which=None) -> list[dict]:
    """Own heroes (or the units where `which`) attack-move to the enemy's centre, once (left to cast on their own
    after: a direct order keeps a computer slot's unit from casting, E43)."""
    which = (us["flags"] & HERO) != 0 if which is None else which
    at = {"x": float(them["x"].mean()), "y": float(them["y"].mean())}
    return orders({int(u): {"command": "attack", "arguments": at} for u in us["unit_id"][which]
                   if int(u) not in orders.given})  # fmt: skip


class Drill:
    """A drill's common parts: attack-move as its opponent and baseline (combat.attack_move), a mirrored army per
    level (`army`), and the solution's orders (`orders`)."""

    LEVELS: tuple = ()
    ORDER_SECONDS = combat.ORDER_SECONDS  # how often the opponent (and the baseline) is re-aimed

    def __init__(self):
        self.orders = Orders()

    def opponent(self, own, enemy, level) -> list[dict]:
        return attack_move(own, enemy)

    def baseline(self, own, enemy, level) -> list[dict]:
        return attack_move(own, enemy)

    def army(self, rng: np.random.Generator, level: int) -> list:
        """[(type, its spawn's extras)] for each side."""
        return [(t, {}) for t in self.LEVELS[level]]

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: int, scripted: int = 1) -> list:
        return staged(observations, rng, self.army(rng, level), scripted)

    def solution(self, own, enemy, level: int) -> list[dict]:
        us, them = _fighters(own), _fighters(enemy)
        return self.solve(own, us, them, level) if len(us) and len(them) else []


def worth(them) -> np.ndarray:
    """Each enemy's worth as a target: summons first (KillSummons: much damage, few hits), then the most damage per
    second for its hit points left (ValueFocus)."""
    dps, _ = unit_stats(them["type_id"])
    return np.isin(them["type_id"], list(_summoned())) * 1e6 + dps / np.maximum(them["hp"], 1)


class Fight(Drill):
    """Combat mode's fight (modes/combat.py): random armies with heroes; the level is the opponent's stage (a
    handicap, combat.HANDICAPS; 0: none). Solution: the drills' rules that passed their gates, together (ideas.md
    P7): the heroes attack-move (casting on their own), each other unit attacks the enemy of most worth within its
    reach, and a unit being hit pulls back (Retreat)."""

    name = "fight"

    def __init__(self):
        super().__init__()
        self.retreat, self.aimed = Retreat(), -1e9

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: float, scripted: int = 1) -> list:
        return combat.stage_ops(observations, rng, level or None, scripted)

    def solve(self, own, us, them, level: float) -> list[dict]:
        mine = ((us["flags"] & HERO) != 0) | (us["max_mana"] > 0)  # heroes and casters: their casts are the AI's
        back, out = self.retreat(own.game_time_seconds, us, them, self.orders, mine)
        skip = mine | back
        now = own.game_time_seconds
        if now - self.aimed >= combat.ORDER_SECONDS:  # heroes and casters re-aimed, as attack-move's are
            self.aimed = now
            for u in us["unit_id"][mine]:
                self.orders.given.pop(int(u), None)
        return (
            heroes_attack_move(us, them, self.orders, mine)
            + out
            + each_attacks(us, them, reaches(us), worth(them), self.orders, skip)
        )


class FocusFire(Drill):
    """Equal armies of melee and ranged units with no spells: the side that kills units sooner wins, since a hurt unit
    deals full damage until it dies. Solution: each unit attacks the enemy with the least hit points within REACH of
    it (none: the nearest), so the army's hits land on few targets. Baseline: attack-move, where each unit picks its
    own (Warcraft's: roughly whoever attacked it or is nearest)."""

    name = "focus_fire"
    LEVELS = (
        ("ogru",) * 4,  # 4 v 4 Grunts
        ("ogru", "ogru", "ogru", "ohun", "ohun", "ohun"),  # Grunts and Headhunters
        ("ogru",) * 4 + ("ohun",) * 3 + ("orai",) * 2 + ("otau",),  # mixed, 10 a side
    )
    REACH = 700.0  # game units: a Headhunter's range and a little; a Grunt's targets are those it reaches soon

    def solve(self, own, us, them, level: int) -> list[dict]:
        return each_attacks(us, them, self.REACH, -them["hp"], self.orders)


class ValueFocus(FocusFire):
    """FocusFire's armies; solution: each unit attacks, within its reach (reaches), the enemy whose damage per
    second is largest for its hit points left (Headhunters and casters before Grunts and Tauren: the most of the
    enemy's damage removed per hit). FocusFire's lowest-hp rule failed its mixed level (E40)."""

    name = "value_focus"

    def solve(self, own, us, them, level: int) -> list[dict]:
        dps, _ = unit_stats(them["type_id"])
        return each_attacks(us, them, reaches(us), dps / np.maximum(them["hp"], 1), self.orders)


class HeroSnipe(Drill):
    """A hero with a few units a side: a hero is worth several units and carries the army's spells and auras, so the
    side that kills the other's hero first should win. Solution: each unit attacks the enemy hero within REACH of it
    (the one with the least hp; none: the nearest enemy), so every hit goes on the hero until it dies. Baseline:
    attack-move, which hits whatever is near (the front line, the hero behind it last)."""

    name = "hero_snipe"
    LEVELS = (
        (("Ofar", 3), ("ogru",) * 4),  # a level 3 Far Seer and 4 Grunts
        (("Ofar", 5), ("ogru", "ogru", "ogru", "ohun", "ohun", "ohun")),  # a level 5 Far Seer, Grunts, Headhunters
    )
    REACH = 500.0  # game units: a Headhunter's range and a little; a Grunt goes for a hero only once it is close

    def army(self, rng: np.random.Generator, level: int) -> list:
        (code, at), units = self.LEVELS[level]
        return [hero(code, at, list(skills(rng, code, at))), *((t, {}) for t in units)]  # a random skill build

    def solve(self, own, us, them, level: int) -> list[dict]:
        mine, theirs = (us["flags"] & HERO) != 0, (them["flags"] & HERO) != 0
        score = np.where(theirs, -them["hp"], -np.inf)  # heroes only; none in reach: the nearest enemy
        return heroes_attack_move(us, them, self.orders) + each_attacks(us, them, self.REACH, score, self.orders, mine)


class BurstHealer(HeroSnipe):
    """HeroSnipe against a Shadow Hunter, whose Healing Wave outheals hits that land one by one (E43: the snipe rule
    0.08). Solution: the units focus the other enemies until BURST of them have the hero within reach, then all of
    them hit it at once."""

    name = "burst_healer"
    LEVELS = ((("Oshd", 5), ("ogru",) * 3 + ("ohun",) * 3),)
    LEARN = ["AOhw", "AOhx", "AOhw", "AOhx", "AOhw"]  # Healing Wave 3, Hex 2
    BURST = 4

    def army(self, rng: np.random.Generator, level: int) -> list:
        (code, at), units = self.LEVELS[level]
        return [hero(code, at, self.LEARN), *((t, {}) for t in units)]

    def solve(self, own, us, them, level: int) -> list[dict]:
        mine, theirs = (us["flags"] & HERO) != 0, (them["flags"] & HERO) != 0
        reach, gap = reaches(us), gaps(us, them)
        on = (gap[:, theirs] <= reach[:, None]).any(1) & ~mine if theirs.any() else np.zeros(len(us), bool)
        score = np.where(theirs, 1e6, -gap.mean(0)) if on.sum() >= self.BURST else np.where(theirs, -1e9, -them["hp"])
        return heroes_attack_move(us, them, self.orders) + each_attacks(us, them, reach, score, self.orders, mine)


class KillSummons(Drill):
    """A summoning hero with Grunts and Headhunters a side: wards and wolves deal much and die in a few hits. Solution:
    each unit attacks the nearest summon within its reach, then the lowest-hp enemy within it (focus fire), then
    the nearest. The heroes summon on their own: gated as two computer slots."""

    name = "kill_summons"
    LEVELS = (
        (hero("Ofar", 5, ["AOsf", "AOcl", "AOsf", "AOcl", "AOsf"]), ("ogru",) * 3 + ("ohun",) * 3),  # Feral Spirit
        (hero("Oshd", 5, ["AOsw", "AOhx", "AOsw", "AOhx", "AOsw"]), ("ogru",) * 3 + ("ohun",) * 3),  # Serpent Ward
    )

    def army(self, rng: np.random.Generator, level: int) -> list:
        lead, units = self.LEVELS[level]
        return [lead, *((t, {}) for t in units)]

    def solve(self, own, us, them, level: int) -> list[dict]:
        summon = np.isin(them["type_id"], list(_summoned()))
        score = np.where(summon, 1e6 - them["hp"], -them["hp"])  # summons first, then the lowest hp
        mine = (us["flags"] & HERO) != 0
        return heroes_attack_move(us, them, self.orders) + each_attacks(us, them, reaches(us), score, self.orders, mine)


class Dive(Drill):
    """Grunts in front, Headhunters and a Shaman behind, and Raiders, a side: the Raiders that go around the front
    line kill the ranged units and the caster, whom attack-move reaches last. Solution: the rest attack-move; each
    Raider first rides to a point WIDE to the side of the enemy's backline, then attacks the nearest ranged enemy."""

    name = "dive"
    LEVELS = (("ogru",) * 3 + ("orai",) * 2 + ("ohun",) * 3 + ("oshm",),)
    WIDE, ARRIVED = 500.0, 250.0

    def __init__(self):
        super().__init__()
        self.last, self.rode = -1e9, set()

    def solve(self, own, us, them, level: int) -> list[dict]:
        _, their_reach = unit_stats(them["type_id"])
        raider = np.array([fourcc(int(t)) == "orai" for t in us["type_id"]])
        ranged = their_reach > 150
        back = np.array([them["x"][ranged].mean(), them["y"][ranged].mean()]) if ranged.any() else None
        out, mine = [], np.array([us["x"].mean(), us["y"].mean()])
        for k in np.flatnonzero(raider):
            unit = int(us["unit_id"][k])
            if back is not None and unit not in self.rode:
                way = back - mine
                side = np.array([-way[1], way[0]]) / max(np.linalg.norm(way), 1e-6)
                to = back + side * self.WIDE * (1 if k % 2 else -1)
                if np.hypot(us["x"][k] - to[0], us["y"][k] - to[1]) > self.ARRIVED:
                    out += self.orders({unit: {"command": "move", "arguments": {"x": float(to[0]), "y": float(to[1])}}})
                    continue
                self.rode.add(unit)
            gap = np.hypot(them["x"] - us["x"][k], them["y"] - us["y"][k])
            pick = np.argmin(np.where(ranged, gap, gap + 1e6))
            out += self.orders({unit: {"command": "attack", "arguments": {"target_id": int(them["unit_id"][pick])}}})
        now = own.game_time_seconds
        if now - self.last >= combat.ORDER_SECONDS:  # the rest attack-move, re-aimed
            self.last = now
            at = {"x": float(them["x"].mean()), "y": float(them["y"].mean())}
            out += [{"unit_id": int(u), "command": "attack", "arguments": dict(at)} for u in us["unit_id"][~raider]]
        return out


class Retreat:
    """Pulling back the unit being hit: a unit below LOW of its hit points that lost more than FAST of them in the
    last second steps BACK away from the enemy, and rejoins after REJOIN seconds (PullBack's rule, E47)."""

    LOW, FAST, BACK, REJOIN = 0.4, 0.15, 300.0, 2.0

    def __init__(self):
        self.seen, self.out = {}, {}  # unit -> [(time, hp)] over the last second; unit -> when it stepped back

    def __call__(self, now: float, us, them, orders: Orders, keep=None) -> tuple[np.ndarray, list[dict]]:
        """(which of `us` are out, the orders that send them back); units where `keep` is True stay in."""
        back, out = np.zeros(len(us), bool), []
        theirs = np.array([them["x"].mean(), them["y"].mean()])
        for k, unit in enumerate(int(u) for u in us["unit_id"]):
            seen = self.seen.setdefault(unit, [])
            seen.append((now, float(us["hp"][k])))
            while now - seen[0][0] > 1.0:
                seen.pop(0)
            top = max(us["max_hp"][k], 1)
            if keep is not None and keep[k]:
                continue
            if unit in self.out and now - self.out[unit] < self.REJOIN:
                back[k] = True
            elif us["hp"][k] / top < self.LOW and (seen[0][1] - us["hp"][k]) / top > self.FAST:
                self.out[unit], back[k] = now, True
                at = np.array([us["x"][k], us["y"][k]])
                to = at + (at - theirs) / max(np.linalg.norm(at - theirs), 1e-6) * self.BACK
                out += orders({unit: {"command": "move", "arguments": {"x": float(to[0]), "y": float(to[1])}}})
        return back, out


class PullBack(Drill):
    """Grunts and Headhunters a side at mixed hit points (the same on both sides): a unit pulled back as it is hit
    makes its attacker walk and pick again, and it fights on later. Solution: focus fire (the lowest hp within reach)
    and Retreat. Level 1 is level 0 without the retreat (focus fire alone, on the same fights): what the pull-back
    adds (E47: 0.76 against 0.69)."""

    name = "pull_back"
    UNITS = ("ogru",) * 3 + ("ohun",) * 3
    LEVELS = (UNITS, UNITS)
    HP = (0.45, 1.0)  # each unit's starting share of its hit points, drawn uniformly

    def __init__(self):
        super().__init__()
        self.retreat = Retreat()

    def army(self, rng: np.random.Generator, level: int) -> list:
        units = reference()["units"]
        return [(t, {"hp": float(units[t]["hp"] * rng.uniform(*self.HP))}) for t in self.LEVELS[level]]

    def solve(self, own, us, them, level: int) -> list[dict]:
        back, out = self.retreat(own.game_time_seconds, us, them, self.orders, np.full(len(us), level == 1))
        return out + each_attacks(us, them, reaches(us), -them["hp"], self.orders, back)


class Surround(Drill):
    """Raiders against a hero that fights until it is hurt, then runs, re-aimed every step: a hero about as fast is
    not caught by a chase (it stops to strike, the hero runs on), so it has to be ringed before it runs. The
    opponent: the hero attacks the nearest Raider until below FLEE of its hit points, then runs to the most open of
    WAYS points STRIDE away (kept inside the arena), and fights again only when cornered (an attacker within OPEN of
    every way). Solution: each Raider takes its slot of a ring RING around the hero, evenly spaced, and attacks once
    within SLOT of it. Baseline: attack-move (all from the front). Varied: 4-6 Raiders, the hero any of combat's at
    level 1-3, without skills (E48 parked a version whose hero ran from the start: no Raider got ahead of it)."""

    name = "surround"
    LEVELS = ((4, 6),)  # Raiders: from, to
    ORDER_SECONDS = 0.25  # every step
    FLEE, WAYS, STRIDE, OPEN, EDGE = 0.6, 16, 300.0, 160.0, 600.0
    RING, SLOT = 110.0, 90.0

    def __init__(self):
        super().__init__()
        self.flight = Orders()

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: int, scripted: int = 1) -> list:
        raiders = [("orai", {})] * int(rng.integers(self.LEVELS[level][0], self.LEVELS[level][1] + 1))
        code = combat.HEROES[rng.integers(len(combat.HEROES))]
        return staged(observations, rng, raiders, scripted, [hero(code, int(rng.integers(1, 4)), [])])

    def opponent(self, own, enemy, level) -> list[dict]:
        heroes, them = _fighters(own), _fighters(enemy)
        if not len(heroes) or not len(them):
            return []
        h, gap = heroes[0], np.hypot(them["x"] - heroes[0]["x"], them["y"] - heroes[0]["y"])
        fight = {"command": "attack", "arguments": {"target_id": int(them["unit_id"][np.argmin(gap)])}}
        if h["hp"] >= self.FLEE * h["max_hp"]:
            return self.flight({int(h["unit_id"]): fight})
        (min_x, min_y, max_x, max_y), _ = combat.arena()
        angles = np.linspace(0, 2 * np.pi, self.WAYS, endpoint=False)
        to = np.stack([h["x"] + self.STRIDE * np.cos(angles), h["y"] + self.STRIDE * np.sin(angles)], 1)
        inside = (to[:, 0] > min_x + self.EDGE) & (to[:, 0] < max_x - self.EDGE)
        inside &= (to[:, 1] > min_y + self.EDGE) & (to[:, 1] < max_y - self.EDGE)
        room = np.hypot(to[:, None, 0] - them["x"][None], to[:, None, 1] - them["y"][None]).min(1)
        room = np.where(inside, room, -np.inf)
        if room.max() < self.OPEN:  # cornered
            return self.flight({int(h["unit_id"]): fight})
        way = to[np.argmax(room)]
        return self.flight(
            {int(h["unit_id"]): {"command": "move", "arguments": {"x": float(way[0]), "y": float(way[1])}}}
        )

    def solve(self, own, us, them, level: int) -> list[dict]:
        h = them[0]
        start = np.arctan2(us["y"].mean() - h["y"], us["x"].mean() - h["x"])  # the ring begins on the Raiders' side
        angles = start + np.arange(len(us)) * 2 * np.pi / len(us)
        slots = np.stack([h["x"] + self.RING * np.cos(angles), h["y"] + self.RING * np.sin(angles)], 1)
        # each Raider to a slot: in the order of their angles around the hero, so none crosses another's way
        order = np.argsort((np.arctan2(us["y"] - h["y"], us["x"] - h["x"]) - start) % (2 * np.pi))
        wanted = {}
        for slot, k in zip(slots, order):
            unit = int(us["unit_id"][k])
            if np.hypot(us["x"][k] - slot[0], us["y"][k] - slot[1]) <= self.SLOT:
                wanted[unit] = {"command": "attack", "arguments": {"target_id": int(h["unit_id"])}}
            else:
                wanted[unit] = {"command": "move", "arguments": {"x": float(slot[0]), "y": float(slot[1])}}
        return self.orders(wanted)


SCENARIOS = {
    s.name: s for s in (Fight, FocusFire, ValueFocus, HeroSnipe, BurstHealer, KillSummons, Dive, PullBack, Surround)
}
