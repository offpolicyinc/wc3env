"""The reward, every term: docs/training/rewards.md (what each pays, how much, its cap, and when it fades).

Win/loss plus shaping, each in a channel of its own (CHANNELS: one critic output each), recorded per step and
player in reward units (a win is 1); the learner weighs each channel's advantages by weights(shaping) and adds them.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from wc3env.binary import DEAD, EVENT_KINDS, ILLUSION, STRUCTURE, fourcc

from ..policy.actions import reference
from ..policy.masks import argument_ids
from ..policy.model import CRITIC

CHANNELS = CRITIC  # ("win", "build", "units", "upgrades", "dealt", "taken"): the critic's outputs
# no reward for hero experience or creeps (dropped 2026-10-11): bc11 already creeps and levels, and both pay without
# ever meeting the enemy; whatever of them helps to win, the win pays for. No reward for scouting either (dropped
# 2026-10-11): sp35's learner at update 1,000 ran units into the enemy's base by minute 2.5 in 31 of 32 games and
# lost ~46 units a game; seeing the base pays only if it helps to win
FADE = {"win": "always", "build": "shaping", "units": "shaping", "upgrades": "shaping", "dealt": "fight",
        "taken": "fight"}  # fmt: skip
FIGHT_FLOOR = 0.1  # dealt and taken fade to this share of FIGHT (2 -> 0.2 per 1,000): a win then outweighs a base
TERMINAL = {"victory": 1.0, "defeat": -1.0}
TIME_LIMIT = TERMINAL["defeat"]  # the game stopped by the clock: a loss for whoever is playing
# while it never wins, the shaping outweighs the win/loss (+-1) several times over: sp33's, ten times smaller, was
# under the noise of the win channel's critic, and the policy drifted without gaining any of it
# per step of edit distance closed between the pro's opening and the player's (AlphaStar's build-order reward):
# ~13 items, ~6.5 for the whole opening; an item out of place or extra costs, so repeating one is no way to earn
BUILD = 0.5
OPENING = 20  # the player's first this many builds are compared (AlphaStar's 20)
# per step of Hamming distance closed between what the pro made over its game and what the player has made
# (AlphaStar's built-units and upgrades rewards, weighted 6 to the build order's 4): a type the pro made pays the
# first time the player makes one, a type it did not costs as much; each research level counts as its own item
UNITS = 0.75  # unit, building and hero types (workers left out): ~17 in an Orc pro's game, ~13
UPGRADES = 0.75  # research levels: 1-5 in an Orc pro's game
FIGHT = 2.0  # per 1,000 of value destroyed while shaping is 1 (a grunt 0.4), buildings only as they die; FIGHT_FLOOR of it once winning
HIT = 5.0  # seconds: an enemy building's death counts while the player's units damaged it this recently
DECAY = 0.6  # per 10 game minutes
STAGES = (
    480,
    960,
    1440,
)  # units and upgrades: x0.5 after 8 game minutes, x0.5 again after 16, nothing after 24 (AlphaStar's)
WORKERS = frozenset(("hpea", "opeo", "ewsp", "uaco"))
SUPPLY = frozenset(("hhou", "otrb", "uzig", "emow"))  # food buildings: left out of build orders, as AlphaStar's
DEATH = next(k for k, name in EVENT_KINDS.items() if name == "death")
DAMAGED = next(
    k for k, name in EVENT_KINDS.items() if name == "damaged"
)  # unit_id the victim, other_id the source, value hundredths
UPGRADE_FINISH = next(k for k, name in EVENT_KINDS.items() if name == "upgrade_finish")
FINISHES = frozenset(
    k
    for k, name in EVENT_KINDS.items()
    if name in ("construct_finish", "train_finish", "research_finish", "upgrade_finish")
)


@cache
def cost(type_id: int) -> float:
    """What a unit or building costs, gold plus lumber (creeps have prices too; summons and critters 0)."""
    u = reference()["units"].get(fourcc(int(type_id)), {})
    return float(u.get("gold", 0) + u.get("lumber", 0))


def _health(obs, owners) -> dict:
    """Unit id -> (hp, max hp, cost, a building) of `obs`'s living units owned by `owners`, illusions left out."""
    u = obs.units
    keep = np.isin(u["owner"], list(owners)) & ((u["flags"] & (DEAD | ILLUSION)) == 0) & (u["max_hp"] > 0)
    rows = zip(u["unit_id"][keep], u["hp"][keep], u["max_hp"][keep], u["type_id"][keep], u["flags"][keep])
    return {int(i): (float(h), float(m), cost(int(t)), bool(f & STRUCTURE)) for i, h, m, t, f in rows}


def stage(seconds: float) -> float:
    """The units and upgrades rewards' weight at this game time (STAGES)."""
    return 0.5 ** sum(seconds >= t for t in STAGES[:2]) if seconds < STAGES[2] else 0.0


def distance(a: list, b: list) -> int:
    """Levenshtein distance: the fewest single insertions, deletions and substitutions turning a into b."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def _dealt(before: dict, done: dict, died: set, hit: dict, now: float) -> float:
    """The value of what one side's units did to the units in `before` (a _health() map of the other side) this
    step: `done` their damage to each (from damaged events, so nobody else's: not creeps'), at most the hp it had;
    a building's whole cost only when it dies, after they damaged it within HIT seconds (`hit`: when they last did).
    Dealt: the player's units on the enemy's; taken: the enemy's on the player's."""
    total = 0.0
    for i, (hp, top, price, building) in before.items():
        if building:
            if i in died and now - hit.get(i, -HIT) < HIT:
                total += price
        elif i in done:
            total += price * min(done[i], hp) / top
    return total


class Shaping:
    """One learning player's shaping over a game: what it has done towards its plan, and its last health maps."""

    def __init__(self, plan: dict | None):
        ids = {i: v for v, i in argument_ids().items()}
        research = reference()["upgrades"]
        # the plan's whole game: type -> how many the pro made (plan.mix is sqrt(count) / 4)
        mix = (plan or {}).get("mix")
        counts = (
            {}
            if mix is None
            else {fourcc(ids[int(i)]): round(float(4 * mix[i]) ** 2) for i in np.flatnonzero(mix) if int(i) in ids}
        )
        self.planned = bool(counts)  # the empty plan: nothing to compare with, so units and upgrades pay nothing
        self.units = {
            c for c, n in counts.items() if c and n > 0 and c not in WORKERS and c not in research
        }  # the pro's types
        self.levels = {
            (c, k) for c, n in counts.items() if c in research for k in range(1, n + 1)
        }  # and research levels
        order = [fourcc(ids.get(int(a), 0)) for a in (plan or {}).get("order", []) if a > 0]
        self.opening = [c for c in order if c and c not in WORKERS and c not in SUPPLY]  # the pro's
        self.built: list[str] = []  # the player's, the same way (its first OPENING)
        self.gap = len(self.opening)  # their edit distance now
        self.hit: dict[int, float] = {}  # enemy unit -> when one of the player's units last damaged it
        self.struck: dict[int, float] = {}  # own unit -> when one of the enemy's units last damaged it
        self.enemies: set[int] = set()  # the enemy player's units seen so far (whose hits count in taken)
        self.made: set[str] = set()  # the player's types so far
        self.researched: dict[str, int] = {}  # and research levels
        self.health: dict | None = None  # the last step's _health maps: own, enemy's

    def step(self, after) -> np.ndarray:
        """This step's shaping channels (CHANNELS, "win" left 0) for the player observing `after`."""
        out = np.zeros(len(CHANNELS), np.float32)
        decay = DECAY ** (after.game_time_seconds / 600)
        me = after.player
        enemy = 1 - me
        own = after.own_unit_ids
        for e in after.events:
            kind, unit = int(e["kind"]), int(e["unit_id"])
            if unit not in own:
                continue
            code = fourcc(int(e["type_id"]))
            if kind == UPGRADE_FINISH and code not in self.units:  # if it names the building as it was, as a start
                into = reference()["units"].get(code, {}).get("upgrades_to", [])
                code = into[0] if len(into) == 1 else code
            if kind not in FINISHES:
                continue
            if self.opening and code not in WORKERS and code not in SUPPLY and len(self.built) < OPENING:
                self.built.append(code)
                gap = distance(self.opening, self.built)
                out[CHANNELS.index("build")] += BUILD * (self.gap - gap) * decay
                self.gap = gap
            if not self.planned or code in WORKERS:
                continue
            if code in reference()["upgrades"]:
                level = self.researched[code] = self.researched.get(code, 0) + 1
                out[CHANNELS.index("upgrades")] += (
                    UPGRADES * (1 if (code, level) in self.levels else -1) * stage(after.game_time_seconds)
                )
            elif code not in self.made:
                self.made.add(code)
                out[CHANNELS.index("units")] += (
                    UNITS * (1 if code in self.units else -1) * stage(after.game_time_seconds)
                )
        died = {int(e["unit_id"]) for e in after.events if int(e["kind"]) == DEATH}
        now = after.game_time_seconds
        mine = own | set(self.health["own"]) if self.health is not None else own  # with those that died this step
        health = {"own": _health(after, (me,)), "enemy": _health(after, (enemy,))}
        self.enemies |= set(health["enemy"])
        done: dict[int, float] = {}  # unit -> the damage the player's own units did to it this step
        suffered: dict[int, float] = {}  # own unit -> the damage the enemy's units did to it this step
        for e in after.events:
            if int(e["kind"]) != DAMAGED:
                continue
            i, source, v = int(e["unit_id"]), int(e["other_id"]), int(e["value"]) / 100
            if source in mine:
                done[i] = done.get(i, 0.0) + v
                self.hit[i] = now
            elif source in self.enemies and i in mine:
                suffered[i] = suffered.get(i, 0.0) + v
                self.struck[i] = now
        if self.health is not None:
            out[CHANNELS.index("dealt")] = FIGHT * _dealt(self.health["enemy"], done, died, self.hit, now) / 1000
            out[CHANNELS.index("taken")] = -FIGHT * _dealt(self.health["own"], suffered, died, self.struck, now) / 1000
        self.health = health
        return out


def win(after) -> float:
    """The win channel: `after` is the player's observation at the step's end."""
    return TERMINAL.get(after.result, 0.0)


def weights(shaping: float, scale: float = 1.0) -> np.ndarray:
    """Each channel's weight now (FADE), for a shaping weight in [0, 1]; every channel but the win's times `scale`."""
    fight = FIGHT_FLOOR + (1 - FIGHT_FLOOR) * shaping
    by = {"always": 1.0, "shaping": scale * shaping, "fight": scale * fight}
    return np.array([by[FADE[c]] for c in CHANNELS], np.float32)
