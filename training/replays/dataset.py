"""Replays to training data: per player and 250 ms step, the features, the label and its masks.

    python -m training.replays.dataset runs/replays/1.29 --out runs/dataset --workers 6
    python -m training.replays.dataset --bots 10 --out runs/bots --workers 6     # games of Warcraft's AI against itself

Each replay is played with both players observed (training/replays/check.py `play`). For each player and step,
the features of the observation the player acted on, the step's label (at most one action: a pro's burst of two or
three within 250 ms carries the rest over to the next steps; a step without one is a no-op) and the masks for that
label's heads, each given the heads before it. Features and masks come from their one implementation, the C the
game hook runs (policy/native.py), fed each observation and the label before it, as the hook is in self-play.
One file per replay and player, `<replay>_<slot>.npz`; a game of Warcraft's AI against itself (--bots: the insane AI
on both sides, its orders converted as a player's, the game's result kept) is `bots-<seed>_<slot>.npz`:

- unit rows, only the filled ones: `units` [rows, UNIT_FLOATS] float16 and the id arrays, with `unit_count`
  [T] giving each step's rows;
- `trees` [T, MAP, MAP] uint8 (the remembered trees); the map's other channels are rebuilt from the rows,
  the labels and the map data (`static_*`, once per file), so they are not stored;
- `scalars` [T, S] float16, `previous_type` [T, HISTORY];
- the label: `type`, `argument` (argument_ids() index), `actor` and `target` (row indices, -1 for none),
  `same_type`, `size` (SIZES index), `queued` (Shift), `point` (a MAP x MAP cell, row-major, -1 for none);
- masks: `type_mask` [T, len(TYPES)], `argument_mask` [T, len(argument_ids()) + 1] (bit-packed),
  `actor_mask` and `target_mask` [T, MAX_UNITS] (bit-packed);
- `game_time` [T] and `result` (1 win, 0 loss, -1 unknown).
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..policy.actions import SIZES, TYPES
from ..policy.features import MAP, PATCH, _cell

ID_FIELDS = (
    "unit_type",
    "unit_owner",
    "unit_order",
    "unit_queue",
    "unit_inventory",
    "unit_buffs",
    "unit_abilities",
    "unit_id",
    "unit_cell",
)


def write(path: str, out: Path) -> dict:
    """Write one replay's data; returns counts. A replay of Warcraft's AI (saved by write_bots: its startup
    options beside it name AI slots) gives `script` orders, a player's replay `player` orders."""
    from wc3env.game import replay_startup_path

    from .check import play

    startup = replay_startup_path(Path(path))
    origin = "script" if startup.is_file() and json.loads(startup.read_text()).get("ai_agents") else "player"
    return _write(play(Path(path)), origin, Path(path).stem, out)


def write_bots(seed: int, out: Path, minutes: float, maps: tuple[str, ...] = ()) -> dict:
    """Write one game of Warcraft's AI against itself: both slots played by the insane AI, its orders converted as
    `script` orders, up to `minutes` of game time; returns counts. The seed picks the map (`maps` in turn, Echo Isles
    when none) and then the matchup (every race pair in turn on each map). The game is kept as a replay too,
    <out>/../replays/bots-<seed>.w3g: the data can be written again from it (`write`) when the features change."""
    from .check import BOT_MAP, RACE_PAIRS, play_bots

    maps = maps or (BOT_MAP,)
    races = RACE_PAIRS[(seed // len(maps)) % len(RACE_PAIRS)] if len(maps) > 1 else RACE_PAIRS[seed % len(RACE_PAIRS)]
    replay = out.parent / "replays" / f"bots-{seed}.w3g"
    return _write(play_bots(seed, races, maps[seed % len(maps)], minutes, replay=replay), "script", f"bots-{seed}", out)


def _write(stream, origin: str, name: str, out: Path) -> dict:
    """Each player's steps of one game (`stream`: (observations, statics) per step) as a data set file."""
    from ..policy.masks import argument_ids
    from ..policy.native import Features
    from .convert import convert

    arg_ids = argument_ids()
    per = {}  # player -> lists
    waited, dropped = Counter(), Counter()  # labels taken a step or more late; labels never possible
    previous, states, pending, results, rallies, static_of = {}, {}, {}, {}, {}, {}
    try:
        for observations, statics in stream:
            for player, obs in observations.items():
                if player not in states:
                    states[player], static_of[player] = Features(statics[player]), statics[player]
                before, f_before = previous.get(player, (None, None))
                label = None
                if before is not None:
                    queue = pending.setdefault(player, [])
                    converted = convert(before, obs, origin, rallies=rallies.setdefault(player, {}))
                    queue += [[c.action, 0] for c in converted if c.action is not None and not c.reason]
                    rows = [int(u) for u in f_before["unit_id"]]
                    # the oldest action the player could take on what this step shows; one that is not possible yet
                    # (a skill point, gold or a cooldown that came during the step) waits for the next step, up to
                    # MAX_WAIT steps
                    for entry in list(queue):
                        a, age = entry
                        if age > MAX_WAIT:
                            queue.remove(entry)
                            dropped[player] += 1
                        elif label is None and _possible(a, rows, f_before, arg_ids):
                            queue.remove(entry)
                            label = a
                            waited[player] += age > 0
                        else:
                            entry[1] += 1
                    append_label(
                        per.setdefault(player, _lists()), before, f_before, label, rows, statics[player], arg_ids
                    )
                previous[player] = obs, states[player].step(obs, label)
                if obs.result:
                    results[player] = obs.result
    except Exception:
        return {"id": name, "error": traceback.format_exc()[-1200:]}
    counts = {}
    for player, lists in per.items():
        static = static_of[player]
        result = {"victory": 1, "defeat": 0}.get(results.get(player, ""), -1)
        target = out / f"{name}_{player}.npz"
        _save(target, lists, static, result)
        counts[player] = {
            "steps": len(lists["type"]),
            "actions": int((np.array(lists["type"]) != 0).sum()),
            "waited": waited[player],  # taken a step or more after the player gave it
            "never possible": dropped[player],  # never allowed by the masks within MAX_WAIT steps
            "bytes": target.stat().st_size,
        }
    return {"id": name, "players": counts}


MAX_WAIT = 4  # steps (one second) an action may wait to become possible


def _possible(a, rows, f, arg_ids) -> bool:
    """Whether the step's masks (f: its features block) allow the action (and its actor and target are rows)."""
    from ..policy.native import chosen_masks

    if (a.actor and a.actor not in rows) or (a.target and a.target not in rows):
        return False
    kind = TYPES.index(a.type)
    if not f["type_mask"][kind]:
        return False
    argument = arg_ids.get(a.argument, 0) if a.argument else 0
    masks = chosen_masks(f, kind, argument)
    if a.argument and not masks["argument"][argument]:
        return False
    return not a.actor or bool(masks["actor"][rows.index(a.actor)])


def _lists():
    keys = ("unit_count", "units", *ID_FIELDS, "trees", "scalars", "previous_type", "game_time")
    keys += ("type", "argument", "actor", "target", "same_type", "size", "queued", "point")
    keys += ("type_mask", "argument_mask", "actor_mask", "target_mask")
    return {k: [] for k in keys}


ID_TYPES = {"unit_id": np.int64, "unit_cell": np.int16}  # as the files have always stored them (the rest int32)


def _append_features(d, obs, f) -> None:
    """One step's features (a features block, policy/native.py) into the lists: its unit rows, its trees."""
    n = int(f["unit_count"])
    d["unit_count"].append(n)
    d["units"].append(f["units"][:n].astype(np.float16))
    for k in ID_FIELDS:
        d[k].append(f[k][:n].astype(ID_TYPES.get(k, np.int32)))
    d["trees"].append(f["trees"])
    d["scalars"].append(f["scalars"].astype(np.float16))
    d["previous_type"].append(f["previous_type"])
    d["game_time"].append(np.float32(obs.game_time_seconds))


def _append_masks(d, masks: dict) -> None:
    d["type_mask"].append(masks["type"])
    d["argument_mask"].append(np.packbits(masks["argument"]))
    d["actor_mask"].append(np.packbits(masks["actor"]))
    d["target_mask"].append(np.packbits(masks["target"]))


def append_label(d, obs, f, label, rows, static, arg_ids) -> None:
    """A replay's step: its features block, its label (an Action, or None for a noop) as the heads' indices, and the
    masks of the label's type and argument."""
    from ..policy.native import chosen_masks

    _append_features(d, obs, f)
    kind = label.type if label is not None else "noop"
    d["type"].append(TYPES.index(kind))
    d["argument"].append(arg_ids.get(label.argument, 0) if label is not None and label.argument else 0)
    d["actor"].append(rows.index(label.actor) if label is not None and label.actor in rows else -1)
    d["target"].append(rows.index(label.target) if label is not None and label.target in rows else -1)
    d["same_type"].append(bool(label.same_type) if label is not None else False)
    d["size"].append(SIZES.index(label.size) if label is not None and label.size in SIZES else 0)
    d["queued"].append(bool(label.queued) if label is not None else False)
    if label is not None and label.x is not None:
        cy, cx = _cell(label.x, label.y, static.bounds)
        d["point"].append(int(cy) * MAP + int(cx))
    else:
        d["point"].append(-1)
    _append_masks(d, chosen_masks(f, TYPES.index(kind), d["argument"][-1]))


def append_decision(d, obs, f, choices: dict, masks: dict) -> None:
    """A self-play step (training/selfplay/actor.py): its features, the choices as the heads' indices
    (policy/agent.py's Decision) and the masks they were sampled under."""
    _append_features(d, obs, f)
    for k in ("type", "argument", "actor", "target", "same_type", "size", "queued"):
        d[k].append(choices[k])
    region, cell = choices["region"], choices["cell"]
    if region >= 0:
        py = (region // (MAP // PATCH)) * PATCH + cell // PATCH
        px = (region % (MAP // PATCH)) * PATCH + cell % PATCH
        d["point"].append(int(py) * MAP + int(px))
    else:
        d["point"].append(-1)
    _append_masks(d, masks)


def _save(target: Path, d: dict, static, result: int) -> None:
    np.savez_compressed(target, **arrays(d, static, result))


def arrays(d: dict, static, result: int) -> dict:
    """The lists of _append as a file's arrays (what ReplayData reads)."""
    out = {
        "unit_count": np.array(d["unit_count"], np.int16),
        "units": np.concatenate(d["units"]) if d["units"] else np.zeros((0, 1), np.float16),
        **{k: np.concatenate(d[k]) for k in ID_FIELDS},
        "trees": np.stack(d["trees"]),
        "scalars": np.stack(d["scalars"]),
        "previous_type": np.stack(d["previous_type"]).astype(np.int8),
        "game_time": np.array(d["game_time"], np.float32),
        "type": np.array(d["type"], np.int8),
        "argument": np.array(d["argument"], np.int32),
        "actor": np.array(d["actor"], np.int16),
        "target": np.array(d["target"], np.int16),
        "same_type": np.array(d["same_type"], bool),
        "size": np.array(d["size"], np.int8),
        "queued": np.array(d["queued"], bool),
        "point": np.array(d["point"], np.int32),
        "type_mask": np.stack(d["type_mask"]),
        "argument_mask": np.stack(d["argument_mask"]),
        "actor_mask": np.stack(d["actor_mask"]),
        "target_mask": np.stack(d["target_mask"]),
        "result": np.array(result, np.int8),
        "bounds": np.array(static.bounds, np.float32),
        "races": np.array(static.races),
    }
    for name in ("walkable", "buildable", "mines", "starts"):
        grid = getattr(static, name)
        out[f"static_{name}"] = (
            np.zeros((MAP, MAP), np.uint8)
            if grid is None
            else np.minimum(grid * 255 if name in ("walkable", "buildable") else grid, 255).astype(np.uint8)
        )
    return out


def _twice(write_one, *args) -> dict:
    """write_one(*args), once more if it failed: a game that stops answering (an RPC timeout, 9 of 675 pro replays
    on busy 8-core VMs) writes fine on a second go."""
    row = write_one(*args)
    return write_one(*args) if "error" in row else row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "replays",
        type=Path,
        nargs="?",
        help="a folder of .w3g replays: those listed in its good.txt "
        "(tools/check_replays.py), or all of them when it has none (bot games' replays)",
    )
    parser.add_argument("--bots", type=int, default=0, help="instead: this many games of Warcraft's AI against itself")
    parser.add_argument("--first-seed", type=int, default=0, help="the bot games' seeds start here")
    parser.add_argument("--minutes", type=float, default=30, help="a bot game's time limit, in game minutes")
    parser.add_argument(
        "--maps",
        default="",
        help="bot games' maps, comma-separated names under Maps/FrozenThrone "
        "(default: Echo Isles), taken in turn by seed",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/dataset"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ids", type=Path, help="a file of replay ids to write instead of the folder's good.txt")
    parser.add_argument("--index", default="index.jsonl", help="the file under --out that gets one line per replay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.bots:
        seeds = [
            s for s in range(args.first_seed, args.first_seed + args.bots) if not any(args.out.glob(f"bots-{s}_*.npz"))
        ]
        maps = tuple(f"Maps/FrozenThrone/{m}.w3x" for m in args.maps.split(",") if m)
        jobs = [(write_bots, s, args.out, args.minutes, maps) for s in seeds]
    else:
        listing = args.ids or args.replays / "good.txt"
        if listing.is_file():
            ids = [line.split()[0] for line in listing.read_text().splitlines() if line.strip()]
        else:
            ids = sorted(p.stem for p in args.replays.glob("*.w3g"))
        ids = [i for i in ids if not any(args.out.glob(f"{i}_*.npz"))][: args.limit or None]
        jobs = [(write, str((args.replays / f"{i}.w3g").absolute()), args.out) for i in ids]
    print(f"{len(jobs)} games to write", flush=True)
    total_steps = total_bytes = errors = 0
    with ProcessPoolExecutor(args.workers) as pool, (args.out / args.index).open("a") as index:
        futures = [pool.submit(_twice, *job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            index.write(json.dumps(row) + "\n")
            index.flush()
            errors += "error" in row
            for c in row.get("players", {}).values():
                total_steps += c["steps"]
                total_bytes += c["bytes"]
            if "error" in row or n % 25 == 0 or n == len(jobs):
                print(
                    f"{n}/{len(jobs)} steps {total_steps} {total_bytes / 1e9:.2f} GB errors {errors} {row.get('error', '')[-300:]}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
