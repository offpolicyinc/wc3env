"""Read classic Warcraft III replays (.w3g): who played, on which map, and every command in order.

    replay = w3g.read("game.w3g")
    replay.map, replay.players, replay.duration_ms
    for time_ms, player_id, actions in replay.commands(): ...

A replay holds only the players' inputs: its lobby (players, slots, map and map checksum) and each
network turn's command blocks. Unit state exists only by playing it back in the game build that recorded
it (`launch(map=replay)`); commands name units by the engine's object ids. Format: w3g_format.txt
(http://w3g.deepnode.de); reads the classic header (version 1: up to patch 1.31).
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"Warcraft III recorded game\x1a\x00"
RACES = {0x01: "human", 0x02: "orc", 0x04: "nightelf", 0x08: "undead", 0x20: "random"}
OBSERVER_TEAM = 24


@dataclass(frozen=True)
class Player:
    slot: int  # the game's player slot (the JASS player id)
    id: int  # the replay's player id, which command blocks carry
    name: str
    race: str
    team: int
    observer: bool
    computer: bool


@dataclass(frozen=True)
class Replay:
    version: int  # the patch's minor number: 29 for 1.29.x
    build: int
    duration_ms: int
    map: str  # as the lobby named it, relative to the game folder
    map_checksum: int
    players: tuple[Player, ...]  # occupied slots, observers included
    data: bytes = b""  # the decompressed body
    turns_at: int = 0  # where the turn records start in `data`

    @property
    def opponents(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if not p.observer)

    def commands(self) -> Iterator[tuple[int, int, bytes]]:
        """(game time in ms, player id, that player's action records for the turn), in order."""
        b, at, t = self.data, self.turns_at, 0
        while at < len(b):
            rid = b[at]
            if rid in (0x1E, 0x1F):  # a turn: length, time increment, then each player's command block
                n, dt = struct.unpack_from("<HH", b, at + 1)
                t += dt
                end, p = at + 3 + n, at + 5
                while p < end:
                    pid, length = struct.unpack_from("<BH", b, p)
                    yield t, pid, b[p + 3 : p + 3 + length]
                    p += 3 + length
                at = end
            elif rid in (0x1A, 0x1B, 0x1C):
                at += 5
            elif rid == 0x17:  # a player leaves
                at += 14
            elif rid == 0x20:  # chat
                at += 4 + struct.unpack_from("<H", b, at + 2)[0]
            elif rid == 0x22:
                at += 2 + b[at + 1]
            elif rid == 0x23:
                at += 11
            elif rid == 0x2F:  # forced game end countdown
                at += 9
            elif rid == 0:  # padding after the last record
                return
            else:
                raise ValueError(f"unknown replay record {rid:#x} at {at}")

    def selections(self) -> Iterator[tuple[int, int, list[int]]]:
        """(game time in ms, player id, object ids) for each selection change a command block starts with."""
        for t, pid, data in self.commands():
            p = 0
            while p + 4 <= len(data) and data[p] == 0x16:  # change selection: mode, count, (id, salt) pairs
                n = struct.unpack_from("<H", data, p + 2)[0]
                yield t, pid, [struct.unpack_from("<I", data, p + 4 + 8 * i)[0] for i in range(n)]
                p += 4 + 8 * n


def read(path: str | Path) -> Replay:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError("not a Warcraft III replay")
    header_size, _, header_version, _, blocks = struct.unpack_from("<5I", raw, 0x1C)
    if header_version != 1:
        raise ValueError("only classic replays (header version 1, up to 1.31) are supported")
    _, version, build, _, duration_ms = struct.unpack_from("<4sIHHI", raw, 0x30)
    data, at = bytearray(), header_size
    for _ in range(blocks):
        compressed, size, _ = struct.unpack_from("<HHI", raw, at)
        data += zlib.decompressobj().decompress(raw[at + 8 : at + 8 + compressed])[:size]
        at += 8 + compressed
    data = bytes(data)

    names = {}
    at = 5  # 4 unknown bytes, then the host's player record
    at = _player_record(data, at, names)
    _, at = _string(data, at)  # game name
    settings, at = _string(data, at + 1)
    settings = _decode(settings)
    map_checksum = struct.unpack_from("<I", settings, 9)[0]
    map_path = _string(settings, 13)[0].decode("latin-1")
    at += 12  # player count, game type, language
    while data[at] == 0x16:
        at = _player_record(data, at + 1, names) + 4
    if data[at] != 0x19:
        raise ValueError("replay has no game start record")
    length, count = struct.unpack_from("<HB", data, at + 1)
    players = []
    for slot in range(count):
        pid, _, status, computer, team, _, race, _, _ = struct.unpack_from("<9B", data, at + 4 + 9 * slot)
        if status == 2:  # occupied
            players.append(
                Player(
                    slot,
                    pid,
                    names.get(pid, ""),
                    RACES.get(race & 0x2F, "unknown"),
                    team,
                    team == OBSERVER_TEAM,
                    bool(computer),
                )
            )
    return Replay(version, build, duration_ms, map_path, map_checksum, tuple(players), data, at + 3 + length)


def _string(b: bytes, at: int) -> tuple[bytes, int]:
    end = b.index(b"\0", at)
    return b[at:end], end + 1


def _player_record(b: bytes, at: int, names: dict[int, str]) -> int:
    """A lobby player record at `at` (after its record id): player id, name, extra data. Returns its end."""
    pid = b[at]
    name, at = _string(b, at + 1)
    names[pid] = name.decode("utf-8", "replace")
    return at + 1 + b[at]


def _decode(encoded: bytes) -> bytes:
    """The lobby's encoded settings: each group of 7 bytes follows a mask byte whose bit k says byte k is odd."""
    out = bytearray()
    for i in range(0, len(encoded), 8):
        mask = encoded[i]
        out += bytes(c if mask & (1 << k) else c - 1 for k, c in enumerate(encoded[i + 1 : i + 8], 1))
    return bytes(out)
