"""Echo Isles with faster game rules and a bolder built-in AI, for short games while the training recipe is tested.

Everything is map data, the files the game reads from a map before its own (as warcraftsim's duel maps,
RobinKa/warcraftsim data/duelmap.py): nothing in the hook or the engine changes, and replays of these games
play back as any other map's.

* `--speed K`: everything that takes time runs K times faster, as if the game ran at K times its speed: build,
  train and research times, attack and cast timings, turning, ability cooldowns and durations, mining,
  regeneration, day and night, a few gameplay constants that are times, and the built-in AI's own waits.
  Movement can't follow: the engine stops units at 522, so every unit moves min(K, 522 / 400) times faster
  (the fastest, 400, at the limit). Walking takes a larger share of a game than at K = 1, most for human and
  orc workers, who walk every load of gold home (warcraftsim at K = 7: night elf then beat the others 96%).
  Costs, hit points, damage and income per trip are the game's.
* `--siege-level N`: the built-in AI attacks the enemy's main base once its force level reaches N (the game's:
  40; a hero is 4, a footman 2, a knight 5, human.ai force_level) or it has siege units.
* `--any-time`: it does so by night too (the game's rule: by day only, game hours 4-12, unless it has air units).

    python -m tools.prepare.fastmap        # self-play's fast mode (training/selfplay/modes/fast.py): 3x, level 20, by night
        # writes build/maps/(2)EchoIsles_x3_s20n.w3x and src/wc3env/data/maps/(2)EchoIsles_x3_s20n.json
    python -m tools.prepare.fastmap --speed 2 --siege-level 40 --no-any-time   # another: (2)EchoIsles_x2.w3x

AI against AI at normal, 6 matchups (2026-10-12), median game minutes: the game's rules 21, 2x 13, 3x with the
bolder AI 8. The bolder AI alone made them longer (31, two of six undecided at 40): it attacks with half an army.

The map data is Echo Isles' (the same terrain, trees, mines, camps and shops) under the new map's name. The unit
and upgrade tables the policy's catalog reads (reference.json) stay the game's: they hold costs and hit points,
which these rules keep; cooldowns come from the engine at run time.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import digest, write_json
from .gamedata import read_all, slk
from .mpq import Archive

SOURCE = "(2)EchoIsles.w3x"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "build" / "maps"
MOVE_LIMIT = 522  # the engine's movement limit (MaxUnitSpeed above it changes nothing)
FASTEST_UNIT = 400  # the base movement speed of the fastest melee units (gyrocopter, hippogryph)
TABLES = ("UnitBalance", "UnitWeapons", "UnitData", "UpgradeData", "AbilityData")
RACE_SCRIPTS = ("human.ai", "orc.ai", "undead.ai", "elf.ai")


@dataclass(frozen=True)
class Rules:
    speed: float = 1.0
    siege_level: int = 40
    any_time: bool = False

    @property
    def move(self) -> float:
        return min(self.speed, MOVE_LIMIT / FASTEST_UNIT)

    @property
    def tag(self) -> str:
        tag = f"_x{self.speed:g}" if self.speed != 1.0 else ""
        if (self.siege_level, self.any_time) != (40, False):
            tag += f"_s{self.siege_level}{'n' if self.any_time else ''}"
        return tag


def patch_slk(text: str, changes: dict[str, dict[str, str]]) -> str:
    """A SYLK table's text with cells replaced: {the row's id (its first column): {column name: value}}. Only cells
    the table has (the rules only scale values above 0, which are always written)."""
    lines = text.split("\n")
    where: list[tuple[int, int] | None] = []
    cells: dict[tuple[int, int], str] = {}
    x = y = 0
    for line in lines:
        at = None
        if line.startswith("C;"):
            value = None
            for field in line.rstrip("\r")[2:].split(";"):
                if field[:1] == "X":
                    x = int(field[1:])
                elif field[:1] == "Y":
                    y = int(field[1:])
                elif field[:1] == "K":
                    value = field[1:]
            if value is not None:
                at = (y, x)
                cells[at] = value.strip('"')
        where.append(at)
    column = {v: c for (r, c), v in cells.items() if r == 1}
    row = {v: r for (r, c), v in cells.items() if c == 1}
    new = {}
    for rid, values in changes.items():
        for name, value in values.items():
            if rid not in row or name not in column or (row[rid], column[name]) not in cells:
                raise KeyError(f"no cell {rid}.{name} in the table")
            new[(row[rid], column[name])] = value
    for i, at in enumerate(where):
        if at in new:
            end = "\r" if lines[i].endswith("\r") else ""
            fields = [f for f in lines[i].rstrip("\r").split(";") if f[:1] != "K"]
            lines[i] = ";".join(fields + ["K" + new[at]]) + end
    return "\n".join(lines)


def _number(row: dict, key: str) -> float | None:
    try:
        v = float(row[key])
    except (KeyError, ValueError, TypeError):
        return None
    return v if v > 0 else None


def table_changes(tables: dict[str, dict[str, dict[str, str]]], k: float, move: float) -> dict[str, dict]:
    """{table: {row id: {column: new value}}}: every time in the game's tables divided by k, rates times k."""
    changes: dict[str, dict] = {name: {} for name in TABLES}

    def put(table, rid, col, value):
        changes[table].setdefault(rid, {})[col] = value

    def times(table, rid, row, cols, whole=False):
        for col in cols:
            if (v := _number(row, col)) is not None:
                put(table, rid, col, str(max(1, round(v / k))) if whole else f"{v / k:.4g}")

    for rid, row in tables["UnitBalance"].items():
        times("UnitBalance", rid, row, ("bldtm", "reptm"), whole=True)
        for col in ("spd", "maxSpd"):
            if (v := _number(row, col)) is not None:
                put("UnitBalance", rid, col, str(min(MOVE_LIMIT, round(v * move))))
        for col in ("regenHP", "regenMana"):
            if (v := _number(row, col)) is not None:
                put("UnitBalance", rid, col, f"{v * k:.4g}")
    for rid, row in tables["UnitWeapons"].items():
        times(
            "UnitWeapons", rid, row, ("cool1", "cool2", "dmgpt1", "dmgpt2", "backSw1", "backSw2", "castpt", "castbsw")
        )
    for rid, row in tables["UnitData"].items():
        if (v := _number(row, "turnRate")) is not None:
            put("UnitData", rid, "turnRate", f"{min(v * k, 3.0):.4g}")
    for rid, row in tables["UpgradeData"].items():
        times("UpgradeData", rid, row, ("timebase", "timemod"), whole=True)
    for rid, row in tables["AbilityData"].items():
        levels = int(_number(row, "levels") or 1)
        times(
            "AbilityData",
            rid,
            row,
            [f"{c}{n}" for n in range(1, levels + 1) for c in ("Cool", "Dur", "HeroDur", "Cast")],
        )
    for rid in ("Agld", "Abgm", "Aegm"):  # a worker's time inside a gold mine or haunted mine
        times("AbilityData", rid, tables["AbilityData"].get(rid, {}), ("DataB1",))
    return {name: rows for name, rows in changes.items() if rows}


def gameplay_constants(misc: str, rules: Rules) -> str:
    """MiscGame.txt (the map's war3mapMisc.txt) with the constants that are times or rates scaled, and the
    movement limits raised to the engine's own."""
    k, move = rules.speed, rules.move
    constants = dict(MaxUnitSpeed=MOVE_LIMIT, MaxBldgSpeed=MOVE_LIMIT, MinUnitSpeed=150 * move, MinBldgSpeed=25 * move,
                     StrRegenBonus=0.05 * k, IntRegenBonus=0.05 * k, GuardReturnTime=5.0 / k,
                     HeroMaxReviveTime=150 / k, ConstructionLifeDrainRate=10.0 * k)  # fmt: skip
    for key, value in constants.items():
        misc, n = re.subn(rf"(?m)^{key}=[^\t\r\n/]*", f"{key}={value:g}", misc)
        if n != 1:
            raise ValueError(f"gameplay constant {key} not found")
    return misc


def ai_scripts(scripts: dict[str, str], rules: Rules) -> dict[str, str]:
    """The built-in AI's scripts (the map's copies override the game's): every Sleep through W3S_Sleep, which
    waits 1 / speed as long; the siege level and the daytime rule as `rules` say."""
    out = {}
    for name, text in scripts.items():
        text = re.sub(r"\bcall Sleep\(", "call W3S_Sleep(", text)
        if name in RACE_SCRIPTS:
            text, n = re.subn(r"(set has_siege\s*=\s*level >= )40\b", rf"\g<1>{rules.siege_level}", text)
            if n != 1:
                raise ValueError(f"{name}: the siege level rule was not found")
        else:
            if rules.any_time:
                text, n = re.subn(r"set can_siege = has_siege and \(air_units or \(daytime>=4 and daytime<=12\)\)",
                                  "set can_siege = has_siege", text)  # fmt: skip
                if n != 1:
                    raise ValueError("common.ai: the daytime rule was not found")
            first = re.search(r"(?m)^function ", text)
            text = (text[: first.start()] + "function W3S_Sleep takes real seconds returns nothing\r\n"
                    f"    call Sleep(seconds * {1 / rules.speed:.4f})\r\nendfunction\r\n\r\n" + text[first.start():])  # fmt: skip
        out[name] = text
    return out


def map_script(script: str, rules: Rules) -> str:
    """war3map.j with day and night `speed` times as fast (after melee initialization)."""
    if rules.speed == 1.0:
        return script
    script, n = re.subn(r"(\n\s*call RunInitializationTriggers\(\s*\))",
                        rf"\1\r\n    call SetTimeOfDayScale({rules.speed:.3f})", script, count=1)  # fmt: skip
    if n != 1:
        raise ValueError("the map script has no RunInitializationTriggers call")
    return script


def files_for(rules: Rules, script: str, game_dir: Path | None = None) -> dict[str, bytes]:
    """The files to put in the map (archive name -> bytes)."""
    names = [f"Units\\{t}.slk" for t in TABLES] + ["Units\\MiscGame.txt"]
    names += [f"Scripts\\{s}" for s in ("common.ai",) + RACE_SCRIPTS]
    raw = read_all(names, game_dir)
    missing = [n for n in names if n not in raw]
    if missing:
        raise FileNotFoundError(f"not in the game's archives: {missing}")
    files: dict[str, str] = {"war3map.j": map_script(script, rules)}
    if rules.speed != 1.0:
        tables = {t: slk(raw[f"Units\\{t}.slk"]) for t in TABLES}
        for table, rows in table_changes(tables, rules.speed, rules.move).items():
            files[f"Units\\{table}.slk"] = patch_slk(raw[f"Units\\{table}.slk"], rows)
        files["war3mapMisc.txt"] = gameplay_constants(raw["Units\\MiscGame.txt"], rules)
    scripts = ai_scripts({s: raw[f"Scripts\\{s}"] for s in ("common.ai",) + RACE_SCRIPTS}, rules)
    files.update({f"Scripts\\{s}": text for s, text in scripts.items()})
    return {name: text.encode("latin-1") for name, text in files.items()}


def build(source: Path, output: Path, rules: Rules) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with tempfile.TemporaryDirectory() as temp, Archive(output, readonly=False) as archive:
        script = archive.read("war3map.j").decode("latin-1")
        for i, (name, data) in enumerate(files_for(rules, script).items()):
            path = Path(temp) / str(i)
            path.write_bytes(data)
            archive.add(path, name)
    return output


def map_data(output: Path) -> Path:
    """Echo Isles' map data under the new map's name (the terrain and everything placed are the same)."""
    from wc3env.data import MAPS

    data = json.loads((MAPS / "(2)EchoIsles.json").read_text(encoding="utf-8"))
    data["map"] = output.name
    data["source"] = {**data["source"], "map_sha256": digest(output)}
    target = MAPS / f"{output.stem}.json"
    write_json(target, data)
    return target


def main():
    from wc3env.settings import settings

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed", type=float, default=3.0, help="everything that takes time this many times faster")
    ap.add_argument(
        "--siege-level", type=int, default=20, help="the AI's force level for attacking a main base (the game's: 40)"
    )
    ap.add_argument(
        "--any-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the AI attacks main bases by night too (the game's: by day only)",
    )
    ap.add_argument("--out", type=Path, help="the map file (default build/maps/(2)EchoIsles<tag>.w3x)")
    args = ap.parse_args()
    rules = Rules(args.speed, args.siege_level, args.any_time)
    if rules == Rules():
        ap.error("these are the game's own rules")
    output = args.out or OUTPUT_DIR / f"(2)EchoIsles{rules.tag}.w3x"
    print(build(settings().game_dir / "Maps" / "FrozenThrone" / SOURCE, output, rules))
    print(map_data(output))


if __name__ == "__main__":
    main()
