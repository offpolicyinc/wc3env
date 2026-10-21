"""Scenarios: small fights that each need one skill, on combat mode's arena (docs/training/todo/curriculum.md, its
design section). A scenario swaps four parts of combat mode's fight and keeps the rest (the network, the actions, the
reward: the win and the army value traded):

    stage_ops(obs, rng, level, scripted)    the debug ops that stage a fight at this level (VectorSession.stage)
    opponent(own, enemy, level)             the scripted opponent's commands (its observation, then the learner's)
    solution(own, enemy, level)             the hand rule for the learner's side: the bar a level is passed at
    baseline(own, enemy, level)             a script without the skill: the solution must beat it, or the skill does
                                            not pay here and the scenario is not built (E9's pull-back did not)

Commands are re-issued only when they change: an attack order given again every step would restart the attack.
"""

from __future__ import annotations

import numpy as np

from wc3env.binary import HERO

from .modes import combat
from .modes.combat import FULL_MANA, _fighters, attack_move, base_ops, placements, skills


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


class AttackMove:
    """The opponent and the baseline every scenario so far uses: Warcraft's attack-move (combat.attack_move)."""

    def opponent(self, own, enemy, level) -> list[dict]:
        return attack_move(own, enemy)

    def baseline(self, own, enemy, level) -> list[dict]:
        return attack_move(own, enemy)


class Fight(AttackMove):
    """Combat mode's fight (modes/combat.py): random armies with heroes; the level is the opponent's stage (a
    handicap, combat.HANDICAPS; 0: none). No hand rule beats attack-move here yet: no solution."""

    name = "fight"
    solution = None

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: float, scripted: int = 1) -> list:
        return combat.stage_ops(observations, rng, level or None, scripted)


class FocusFire(AttackMove):
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

    def __init__(self):
        self.orders = Orders()

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: int, scripted: int = 1) -> list:
        units = [(t, {}) for t in self.LEVELS[level]]
        return base_ops(observations) + [
            ("spawn", {"type_id": t, "player": p, "x": x, "y": y, "n": 1, **extra, "mana": FULL_MANA})
            for t, extra, p, x, y in placements(rng, units, (), scripted)
        ]

    def solution(self, own, enemy, level: int) -> list[dict]:
        us, them = _fighters(own), _fighters(enemy)
        if not len(us) or not len(them):
            return []
        gap = np.hypot(us["x"][:, None] - them["x"][None, :], us["y"][:, None] - them["y"][None, :])
        wanted = {}
        for k, unit in enumerate(us["unit_id"]):
            near = gap[k] <= self.REACH
            pick = np.argmin(np.where(near, them["hp"], np.inf)) if near.any() else np.argmin(gap[k])
            wanted[int(unit)] = {"command": "attack", "arguments": {"target_id": int(them["unit_id"][pick])}}
        return self.orders(wanted)


class HeroSnipe(AttackMove):
    """A hero with a few units a side: a hero is worth several units and carries the army's spells and auras, so the
    side that kills the other's hero first should win. Solution: each unit attacks the enemy hero within REACH of it
    (none: the nearest enemy), so every hit goes on the hero until it dies; its own hero attack-moves. Baseline: attack-move, which hits whatever
    is near (the front line, the hero behind it last)."""

    name = "hero_snipe"
    LEVELS = (
        (("Ofar", 3), ("ogru",) * 4),  # a level 3 Far Seer and 4 Grunts
        (("Ofar", 5), ("ogru", "ogru", "ogru", "ohun", "ohun", "ohun")),  # a level 5 Far Seer, Grunts, Headhunters
    )
    REACH = 500.0  # game units: a Headhunter's range and a little; a Grunt goes for a hero only once it is close

    def __init__(self):
        self.orders = Orders()

    def stage_ops(self, observations: dict, rng: np.random.Generator, level: int, scripted: int = 1) -> list:
        (hero, at), units = self.LEVELS[level]
        army = [(hero, {"level": at, "learn": list(skills(rng, hero, at)), "items": []}), *((t, {}) for t in units)]
        return base_ops(observations) + [
            ("spawn", {"type_id": t, "player": p, "x": x, "y": y, "n": 1, **extra, "mana": FULL_MANA})
            for t, extra, p, x, y in placements(rng, army, (), scripted)
        ]

    def solution(self, own, enemy, level: int) -> list[dict]:
        us, them = _fighters(own), _fighters(enemy)
        if not len(us) or not len(them):
            return []
        gap = np.hypot(us["x"][:, None] - them["x"][None, :], us["y"][:, None] - them["y"][None, :])
        heroes, centre = (them["flags"] & HERO) != 0, {"x": float(them["x"].mean()), "y": float(them["y"].mean())}
        wanted = {}
        for k, unit in enumerate(us["unit_id"]):
            if us["flags"][k] & HERO:  # its own hero attack-moves once, as the baseline's: an attack order on a unit
                if int(unit) not in self.orders.given:  # would keep it from casting on its own (a computer slot's)
                    wanted[int(unit)] = {"command": "attack", "arguments": centre}
                continue
            near = heroes & (gap[k] <= self.REACH)
            pick = np.argmin(np.where(near, them["hp"], np.inf)) if near.any() else np.argmin(gap[k])
            wanted[int(unit)] = {"command": "attack", "arguments": {"target_id": int(them["unit_id"][pick])}}
        return self.orders(wanted)


SCENARIOS = {s.name: s for s in (Fight, FocusFire, HeroSnipe)}
