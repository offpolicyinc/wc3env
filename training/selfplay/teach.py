"""A drill's solution as a teacher (on-policy distillation, docs/training/rsi/ideas.md P5): a policy in the model's
own action space, one order a step (Teacher.label: one unit's, of those whose current order is not the script's,
at random; a noop when every unit follows it), whose labels on the model's steps the learner imitates beside PPO
(learner --teach, the actors' --teachers). Its own play is training/bench/scenarios.py's "teacher" player, from the
model's slot (no casts on their own): a drill is taught only where it beats the model there (focus fire 0.98; the
hero drills 0.00, the model ~0.75). Labels that were the script's orders in other forms hurt (rsi log E52-E54):
without the noops (an order on every labelled step: the policy attacked on 81% of steps), and the script's targets
for the policy's own group orders (a group sent after the weakest enemy across the line).
"""

from __future__ import annotations

import numpy as np

from wc3env.binary import HERO

from ..policy.actions import TYPES, Action
from ..policy.native import chosen_masks
from ..replays.convert import ATTACK, MOVE
from .scenarios import Orders

NEAR = 100.0  # a move is followed when its point is this close to the script's


class Wanted(Orders):
    """Every order the script wants now, changed or not (Orders sends only the changes)."""

    def __call__(self, wanted: dict[int, dict]) -> list[dict]:
        return [{"unit_id": unit, **command} for unit, command in wanted.items()]


def action(command: dict) -> Action | None:
    """A script's command for one unit as the model's action (that unit alone); None: one the model has no type for."""
    unit, kind, a = command["unit_id"], command["command"], command["arguments"]
    if kind == "attack" and "target_id" in a:
        return Action("attack", actor=unit, size=1, target=int(a["target_id"]))
    if kind in ("attack", "move") and "x" in a:
        return Action("attack_move" if kind == "attack" else "move", actor=unit, size=1, x=a["x"], y=a["y"])
    return None


def follows(unit, command: dict) -> bool:
    """Whether a unit's current order (an observation's units row) is already the command's. An attack-move is
    followed by any attack order (one that found a target is still one), and a hero's or caster's by any order (its
    casts: a computer slot's, actor --computer)."""
    kind, a, order = command["command"], command["arguments"], int(unit["order_id"])
    if kind == "attack" and "target_id" in a:
        return order == ATTACK and int(unit["order_target"]) == int(a["target_id"])
    if kind == "attack":
        return order == ATTACK or ((bool(unit["flags"] & HERO) or unit["max_mana"] > 0) and order != 0)
    if kind == "move":
        return order == MOVE and np.hypot(unit["order_x"] - a["x"], unit["order_y"] - a["y"]) <= NEAR
    return True


def legal(label: Action, f: dict) -> bool:
    """Whether the model could give a label at a step (its feature block): its units in the model's rows, its type
    allowed there and its actor by that type; else it is not taught (an impossible label's loss swamps the rest)."""
    rows = list(f["unit_id"][: int(f["unit_count"])])
    if label.actor not in rows or (label.target and label.target not in rows):  # (target 0: none)
        return False
    kind = TYPES.index(label.type)
    return bool(f["type_mask"][kind]) and bool(chosen_masks(f, kind, 0)["actor"][rows.index(label.actor)])


class Teacher:
    """One game's teacher: its own copy of the drill (a script may keep state) and the level."""

    def __init__(self, drill, level, rng: np.random.Generator):
        self.drill, self.level, self.rng = drill, level, rng
        self.drill.orders = Wanted()

    def label(self, own, enemy) -> Action | None:
        """The next order (None: a noop): one unit's, of those whose current order is not the script's, at random,
        from the learning side's and the enemy's observations."""
        units = {int(u["unit_id"]): u for u in own.units}
        todo = []
        for command in self.drill.solution(own, enemy, self.level):
            unit = units.get(int(command["unit_id"]))
            if unit is not None and not follows(unit, command) and (a := action(command)) is not None:
                todo.append(a)
        return todo[self.rng.integers(len(todo))] if todo else None
