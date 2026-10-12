"""Build a flat, open arena from a 2-player melee map, for fights where terrain must not matter.

Every terrain point gets the same height and cliff level, with no water, ramps or blight; all trees and
doodads go; the pathing grid is open inside the playable area; and the script no longer creates the
map's preplaced units (creeps, gold mines, shops). Start locations and melee setup stay, so each
player still starts with a town hall and workers at its start location.

    python -m tools.prepare.arena            # writes build/maps/(2)FlatArena.w3x and its map data

The map data (src/wc3env/data/maps/(2)FlatArena.json, as `python -m tools.prepare map` writes for stock maps) is
Echo Isles' with the arena's open pathing and nothing placed: no trees, mines, camps or shops; the same bounds and
start locations. The self-play policy's features read it (training/policy/features.py Static, combat fights).
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

from .artifacts import digest, write_json
from .mapinfo import grid_origin, pathing
from .mpq import Archive

SOURCE = "(2)EchoIsles.w3x"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build" / "maps" / "(2)FlatArena.w3x"
GROUND = 0x2000  # height of the zero level
LAYER = 2  # cliff level for every point
BOUNDARY = 0x4000  # in the water word: the point lies outside the playable map
NO_WALK, NO_FLY, NO_BUILD = 0x02, 0x04, 0x08


def flat_terrain(data: bytes):
    """war3map.w3e (version 11) with every tile point level, dry and on texture 0; and its centre offset."""
    if data[:4] != b"W3E!" or struct.unpack_from("<i", data, 4)[0] != 11:
        raise ValueError("expected a version 11 war3map.w3e")
    offset = 4 + 4 + 1 + 4
    (grounds,) = struct.unpack_from("<i", data, offset)
    offset += 4 + 4 * grounds
    (cliffs,) = struct.unpack_from("<i", data, offset)
    offset += 4 + 4 * cliffs
    width, height = struct.unpack_from("<ii", data, offset)
    offset += 8 + 8  # size, then the centre offset
    out = bytearray(data)
    for i in range(width * height):
        at = offset + 7 * i
        _, water, texture, variation, cliff = struct.unpack_from("<hHBBB", data, at)
        struct.pack_into(
            "<hHBBB",
            out,
            at,
            GROUND,
            (water & BOUNDARY) | 0x1000,  # water level below the ground, boundary flag kept
            texture & 0x80,  # texture 0; of the flags only "boundary" survives (no ramp, blight or water)
            variation,
            (cliff & 0xF0) | LAYER,
        )
    return bytes(out), struct.unpack_from("<ff", data, offset - 8)


def open_pathing(data: bytes, centre, bounds) -> bytes:
    """war3map.wpm: walkable, flyable and buildable inside `bounds`, closed outside."""
    if data[:4] != b"MP3W":
        raise ValueError("expected war3map.wpm")
    width, height = struct.unpack_from("<ii", data, 8)
    left, bottom, right, top = bounds
    cells = bytearray(width * height)
    for j in range(height):
        y = centre[1] + 32 * j
        for i in range(width):
            x = centre[0] + 32 * i
            if not (left <= x <= right and bottom <= y <= top):
                cells[j * width + i] = NO_WALK | NO_FLY | NO_BUILD
    return data[:16] + bytes(cells)


def build(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with tempfile.TemporaryDirectory() as temp, Archive(output, readonly=False) as archive:
        terrain, centre = flat_terrain(archive.read("war3map.w3e"))
        script = archive.read("war3map.j").decode("latin-1")
        # Playable area from the camera bounds the script sets (left, bottom, right, top).
        numbers = re.search(
            r"SetCameraBounds\(\s*(-?[\d.]+)[^,]*,\s*(-?[\d.]+)[^,]*,\s*(-?[\d.]+)[^,]*,\s*(-?[\d.]+)", script
        )
        bounds = tuple(float(v) for v in numbers.groups())
        if "call CreateAllUnits(  )" not in script:
            raise ValueError("the map script does not create its units with CreateAllUnits")
        files = {
            "war3map.w3e": terrain,
            "war3map.wpm": open_pathing(archive.read("war3map.wpm"), centre, bounds),
            "war3map.shd": bytes(len(archive.read("war3map.shd"))),  # no baked tree or cliff shadows
            "war3map.doo": b"W3do" + struct.pack("<iiiii", 8, 11, 0, 0, 0),  # no doodads, no special doodads
            "war3map.j": script.replace("call CreateAllUnits(  )", "").encode("latin-1"),
        }
        for name, data in files.items():
            path = Path(temp) / name
            path.write_bytes(data)
            archive.add(path, name)
    return output


def map_data(output: Path) -> Path:
    """The arena's map data, beside Echo Isles' (see the module's docstring)."""
    from wc3env.data import MAPS

    data = json.loads((MAPS / "(2)EchoIsles.json").read_text(encoding="utf-8"))
    with Archive(output) as archive:
        grid = pathing(archive.read("war3map.wpm"), grid_origin(archive.read("war3map.w3e")))
    data.update(map=output.name, pathing=grid, trees={"type_id": [], "x": [], "y": []}, gold_mines=[], creep_camps=[],
                neutral_buildings=[])  # fmt: skip
    data["source"] = {**data["source"], "map_sha256": digest(output)}
    target = MAPS / f"{output.stem}.json"
    write_json(target, data)
    return target


def main():
    from wc3env.settings import settings

    source = settings().game_dir / "Maps" / "FrozenThrone" / SOURCE
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    print(build(source, output))
    print(map_data(output))


if __name__ == "__main__":
    main()
