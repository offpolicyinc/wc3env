"""Who the learning policy plays (docs/training/8-decisions.md): itself, frozen snapshots of itself, and a ladder of
Warcraft's AI.

The ladder (ladder games: the first --ai-games of each actor) and its stages are the mode's (training/selfplay/modes).
Whole games: Warcraft's AI at easy, normal and insane ("ai0", "ai1", "ai2"), each rung handicapped first (HANDICAPS):
the AI's units take every hit scaled by 1/h, as if they had h of their hit points (the hook's handicap,
wc3hook/stage.c), so the learner meets a real opponent that builds, defends and attacks, and can still beat it.
Winning PROMOTE of a block of WINDOW games (blocks one after another, not a window rolling every game: checked
after every game, a rolling one is soon passed by a lucky stretch) moves to the next stage, a stronger AI, or FAST_WINS of a stage's first
FAST_GAMES (a stage far too easy is not played 25 times); after the last stage the next win streak moves up a rung,
which starts at its first stage again. A time limit is a loss. Win/loss stays the reward: the ladder, not hand-made terms, makes wins reachable (sp1-sp4 never
won against the easy AI, and their shaping was farmed).

Self-play games pick their opponent per game: the current policy half the time, otherwise a snapshot chosen by
prioritised fictitious self-play (AlphaStar's PFSP, cut down): weight (1 - win rate against it)^2, so the snapshots
that still beat the learner get played most. Only the newest SNAPSHOTS are kept.

Games start short (START_MINUTES: many games, so many wins and losses to learn from, and no time to stall) and get
MINUTES_STEP longer, up to MAX_MINUTES, each time EXTEND_GAMES ladder games were won EXTEND of the time; never shorter.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import MISSING, dataclass, field, fields

import numpy as np

SELF = "self"
LADDER = ("ai0", "ai1", "ai2")
# an AI rung's stages: its units' hit points as a share of their own (1/h on every hit taken), the last one full
# 0.05 and 0.1 added 2026-10-11: bc12 won 2 of 16 twenty-minute games at 0.05, none at 0.2 (sp35's learner never won there)
HANDICAPS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
# 80 of each 100 (2026-10-12; 70% of the last 50 before, then 20 of the last 25, checked every game: cb1 moved on at about
# 70%, cb3 from 0.95 at 60-70%; a stage sees hundreds of games, so a rolling window had hundreds of chances. Then 40 of
# each 50: cb5 left 0.95 on one block of 41 while its blocks ran 22-35, ~60%)
WINDOW, PROMOTE = 100, 0.8
EXTEND_GAMES = 50  # ladder games between time-limit extensions
FAST_GAMES, FAST_WINS = 20, 18
LATEST_SHARE = 0.5
# snapshots kept, the newest (each one in a batch is one more pass of the model in inference; the oldest are the ones
# the learner beats most, which PFSP plays least)
SNAPSHOTS = 4
START_MINUTES, MINUTES_STEP, MAX_MINUTES, EXTEND = 15.0, 5.0, 30.0, 0.5
# ladder games count in the order they began, each once every game begun before it has ended: a win ends a game
# early and a loss runs to the clock, so counted as they finished, the first games after a change were all wins (sp9:
# 20 of 20 at update 90, 6% of 50 at update 190, on the same stage; sp8 moved on that way too early). Each actor says
# when its oldest running ladder game began; a finished game counts once every live actor's oldest began after it.
# An actor silent for SILENT seconds is not waited for (it stopped, crashed, or its run ended)
SILENT = 120.0


@dataclass
class Snapshot:
    name: str
    update: int
    games: int = 0
    learner_wins: float = 0.0  # a self-play game at the time limit counts by who was ahead, a tie half

    def win_rate(self) -> float:
        return (self.learner_wins + 1) / (self.games + 2)  # a prior of one win and one loss


@dataclass
class League:
    snapshots: list[Snapshot] = field(default_factory=list)
    ladder: tuple = LADDER
    handicaps: tuple = HANDICAPS  # each rung's stages (the mode's)
    level: int = 0  # the ladder's rung the learner is on
    results: deque = field(default_factory=lambda: deque(maxlen=WINDOW))  # its last games there: 1 won, 0 not
    minutes: float = START_MINUTES  # every game's time limit
    games: int = 0  # ladder games on this rung
    extended_at: int = 0  # `games` when the time limit last grew
    stage: int = 0  # the rung's handicap stage (an index of `handicaps`)
    hold: bool = False  # never move on (a fixed opponent: measuring a policy, not training it up a ladder)
    finished: list = field(default_factory=list)  # (when it began, score) of ladder games here not yet counted
    oldest: dict = field(default_factory=dict)  # actor -> (when its oldest running ladder game began or None, when it said)

    def __setstate__(self, state: dict) -> None:
        """A league saved before a field existed gets that field's default (resuming an older run)."""
        state.pop("in_flight", None)  # the queue the actors' reports replaced
        for f in fields(self):
            if f.name not in state:
                state[f.name] = f.default_factory() if f.default_factory is not MISSING else f.default
        self.__dict__.update(state)

    @property
    def rung(self) -> str:
        return self.ladder[self.level]

    @property
    def handicap(self) -> float:
        """The ladder opponent's hit points as a share of their own (1: none; above 1, a stronger opponent)."""
        return self.handicaps[min(self.stage, len(self.handicaps) - 1)]  # (a league saved with a longer list: its last)

    def win_rate(self) -> float:
        """The learner's win rate on its rung (0 before 10 games there)."""
        return float(np.mean(self.results)) if len(self.results) >= 10 else 0.0

    def shaping(self) -> float:
        """The shaping rewards' weight (reward.py): 1 while the learner never wins on its rung, 0 from 50% wins."""
        return max(0.0, 1.0 - self.win_rate() / 0.5)

    def add(self, name: str, update: int) -> None:
        """A new snapshot joins; past SNAPSHOTS, the oldest leaves."""
        self.snapshots = [*self.snapshots, Snapshot(name, update)][-SNAPSHOTS:]

    def report(self, actor: int, ended: list, oldest: float | None, now: float | None = None) -> None:
        """An actor's report: games that ended, each (opponent, score, when it began, the (rung, stage) it began
        under; None for self-play), and when its oldest running ladder game began (None: none running)."""
        now = time.time() if now is None else now
        for opponent, score, began, stage in ended:
            if stage is None:
                self.record(opponent, score)
            elif tuple(stage) == (self.rung, self.stage):  # begun before the learner moved on: not counted
                self.finished.append((began, score))
        self.oldest[actor] = (oldest, now)
        live = [o for o, said in self.oldest.values() if o is not None and now - said < SILENT]
        upto = min(live, default=now)  # every game begun before this has ended
        ready = sorted(f for f in self.finished if f[0] < upto)
        self.finished = [f for f in self.finished if f[0] >= upto]
        for _, score in ready:
            if self._count(score):  # moved on: the rest began under the old stage
                self.finished = []
                return

    def record(self, opponent: str, score: float) -> None:
        """A finished game counted at once: a self-play game (into its snapshot's record) or a ladder game (tests)."""
        if opponent in self.ladder:
            if opponent == self.rung:
                self._count(score)
            return
        for s in self.snapshots:
            if s.name == opponent:
                s.games += 1
                s.learner_wins += score

    def _count(self, score: float) -> bool:
        """One ladder game into the window, in start order; True when that moved the learner on (a stage, a rung)."""
        self.results.append(score)
        self.games += 1
        rate = np.mean(self.results) if self.games % WINDOW == 0 else 0.0  # a block's end (the window holds the block)
        if self.games == FAST_GAMES and sum(self.results) >= FAST_WINS:  # far too easy: on at once
            rate = 1.0
        if self.hold:
            return False
        if rate >= PROMOTE and self.stage < len(self.handicaps) - 1:  # the next stage: a stronger opponent
            self.stage += 1
            self.results.clear()
            self.games = self.extended_at = 0
            return True
        if rate >= PROMOTE and self.level < len(self.ladder) - 1:
            self.level += 1
            self.stage = 0  # a new rung starts at its first stage (an AI handicapped)
            self.results.clear()
            self.games = self.extended_at = 0
            return True
        if rate >= EXTEND and self.minutes < MAX_MINUTES and self.games - self.extended_at >= EXTEND_GAMES:
            self.minutes += MINUTES_STEP
            self.extended_at = self.games
        return False

    def plan(self) -> dict:
        """What actors need: the ladder's rung, its stage and handicap, each snapshot's PFSP weight and the time limit."""
        weights = {s.name: (1 - s.win_rate()) ** 2 for s in self.snapshots}
        return {"rung": self.rung, "stage": self.stage, "handicap": self.handicap, "snapshots": weights,
                "minutes": self.minutes}


def choose(plan: dict, rng: np.random.Generator) -> str:
    """A self-play game's opponent from League.plan(): SELF, or a snapshot's name."""
    weights = plan["snapshots"]
    if not weights or rng.random() < LATEST_SHARE:
        return SELF
    names = list(weights)
    p = np.array([weights[n] for n in names]) + 1e-3
    return names[int(rng.choice(len(names), p=p / p.sum()))]
