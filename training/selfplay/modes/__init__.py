"""What self-play plays: a mode's map, matchups and episodes; the actor, the inference process and the learner are the
same for every mode (docs/training/7-selfplay.md, docs/training/combat.md).

    game    whole games on Echo Isles (modes/game.py): pro plans, Warcraft's AI on the ladder, shaping that fades
    fast    whole games as game's, on Echo Isles 3 times as fast with a bolder AI (modes/fast.py): half as long
    combat  fights on the flat arena (modes/combat.py): mirrored armies with heroes, Warcraft's attack-move on the
            ladder, the army value traded every step

A mode (one per actor and learner, chosen by --mode) has:
    map                         the games' map, under the game's folder (the actor's --map overrides it)
    ladder, handicaps           the league's ladder: its rungs, and each rung's stages (league.py)
    pairs(races)                the matchups: (own race, opponent's), the actor's games take them in turn
    plan(actor, race, other)    a learning player's plan for an episode (None: the empty plan)
    config(game, rung)          GameConfig's arguments that differ by mode (who the computer plays: the
                                learner plays game.slot, 0 or 1, the opponent game.other)
    episode(actor, game)        an Episode: one episode's rules, made at its start (its seats and policies chosen)
    shaping(fade)               the learner's weight for the shaping channels, given the league's fading one

An Episode:
    start(i, obs, session)      its first observations to decide on; or None, a staging step sent (VectorSession.stage)
                                whose observations come back through recv and go to staged()
    staged(obs)                 those first observations
    commands(obs)               the scripted side's commands this step ({slot: [command]})
    step(final, done, steps)    a Step: each learning player's reward, whether the episode is over, how it ended
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Step:
    rewards: dict  # slot -> its reward channels (reward.CHANNELS)
    over: bool  # the episode ended with this step
    outcome: str = ""  # then, for the learning player (Game.slot): "victory", "defeat", "draw" or "time limit"
    minutes: float = 0.0  # then, its length in game minutes
    reload: bool = False  # then, the map loaded again before the next episode (else it begins where this one ended)


class Episode:
    """The defaults: nothing staged, nothing scripted."""

    def start(self, i: int, obs: dict, session) -> dict | None:
        return obs

    def staged(self, obs: dict) -> None:
        pass

    def commands(self, obs: dict) -> dict:
        return {}

    def step(self, final: dict, done: bool, steps: int) -> Step:
        raise NotImplementedError


def get(name: str):
    """The mode called `name`."""
    from .combat import Combat
    from .fast import Fast
    from .game import Games

    return {"game": Games, "fast": Fast, "combat": Combat}[name]()


NAMES = ("game", "fast", "combat")
