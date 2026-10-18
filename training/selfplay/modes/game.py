"""Whole games (--mode game): Echo Isles (or --map), every matchup in turn unless --races; a pro's plan for most
learning players; the ladder is Warcraft's AI, handicapped by the league's stages; the shaping rewards
(reward.Shaping) fade as the learner wins; a game stopped at the league's time limit is a loss and is loaded again.
"""

from __future__ import annotations

from ...policy.features import PAIRS, STEP_MS
from ..league import HANDICAPS, LADDER
from ..reward import CHANNELS, TIME_LIMIT, Shaping, win
from . import Episode, Step

PLAN_SHARE = 0.9  # players given a pro plan (the rest play the empty plan, as 10% of cloning's games did)


class Games:
    map = "Maps/FrozenThrone/(2)EchoIsles.w3x"
    ladder, handicaps = LADDER, HANDICAPS  # Warcraft's AI, easy to insane, each rung handicapped first (league.py)

    def pairs(self, races) -> list[tuple[str, str]]:
        return [tuple(races)] if races else PAIRS

    def plan(self, actor, race: str, other: str) -> dict | None:
        if actor.bank is not None and actor.rng.random() < PLAN_SHARE:
            return actor.bank.sample(race, other, actor.rng)
        return None

    def config(self, game, rung: str) -> dict:
        ai = game.ladder and rung.startswith("ai")
        return {"ai_agents": (game.other,) if ai else (), "ai_difficulty": int(rung[2:]) if ai else None}

    def episode(self, actor, game) -> Episode:
        return GameEpisode(actor, game)

    def shaping(self, fade: float) -> float:
        return fade


class GameEpisode(Episode):
    def __init__(self, actor, game):
        self.actor, self.game = actor, game
        self.shaping = {slot: Shaping(game.seats[slot].plan) for slot, p in game.policy.items() if p == "latest"}

    def start(self, i: int, obs: dict, session) -> dict:
        """At the start of a ladder game against a handicapped AI rung (league.HANDICAPS): the AI's handicap."""
        handicap = self.actor.handicap(self.game)
        if self.game.ladder and self.game.seatless and handicap < 1:
            session.debug(i, "handicap", player=self.game.other, hp=float(handicap))
        return obs

    def step(self, final: dict, done: bool, steps: int) -> Step:
        too_long = not done and steps >= self.actor.league["minutes"] * 60 * 1000 / STEP_MS
        rewards = {}
        for slot, shaping in self.shaping.items():
            r = shaping.step(final[slot])
            r[CHANNELS.index("win")] = win(final[slot]) + (TIME_LIMIT if too_long else 0.0)
            rewards[slot] = r
        us = final[self.game.slot]
        outcome = "time limit" if too_long else us.result or "draw"
        return Step(rewards, done or too_long, outcome, us.game_time_seconds / 60, reload=too_long)
