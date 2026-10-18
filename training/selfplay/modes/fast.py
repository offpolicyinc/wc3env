"""Fast games (--mode fast): whole games as modes/game.py, on Echo Isles with everything that takes time 3 times
faster (movement 1.3 times, the engine's limit) and Warcraft's AI attacking main bases from force level 20, by night
too: AI against AI at normal (2026-10-12), a median game of ~16 game minutes in orc mirrors against ~31 (10 and 3
games), ~8 against ~21 over 6 mixed matchups. For trying the training recipe in short games before whole ones. Build
the map first: python -m tools.prepare.fastmap.
"""

from __future__ import annotations

from .game import Games


class Fast(Games):
    map = "Maps/FrozenThrone/(2)EchoIsles_x3_s20n.w3x"  # tools/prepare/fastmap.py's defaults
