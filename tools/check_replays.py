"""Play replays to their end and check each stays in sync with the recorded game.

    python tools/check_replays.py runs/replays/1.29 --out runs/replays/1.29/check.jsonl --workers 6

A replay stores only inputs, and its commands name units by the engine's object ids, so a playback that
drifts from the recorded game shows up as players selecting units that do not exist. Each game second the
replay's selections are checked against the units both players can see (a unit that died within that
second counts as present). Per replay, one JSON line: status `ok` (with the miss rate, the worst minute's
rate and the final results), `map_missing`, `no_load` (never leaves the start hold: recorded on another
patch or map version) or `error`. In sync, misses stay around 1%: units dying between checks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from wc3env import w3g
from wc3env.settings import settings


def check(path: str) -> dict:
    from wc3env.binary import SharedObservations
    from wc3env.game import launch

    replay = w3g.read(path)
    players = replay.opponents
    slots = tuple(p.slot for p in players)
    row = {"id": Path(path).stem, "map": replay.map, "seconds": replay.duration_ms // 1000, "slots": slots}
    if not (settings().game_dir / replay.map.replace("\\", "/")).is_file():
        return {**row, "status": "map_missing"}
    slot_of = {p.id: p.slot for p in players}  # observers may select anything they watch
    selected = defaultdict(set)  # second -> object ids
    for t, pid, ids in replay.selections():
        if pid in slot_of:
            selected[t // 1000].update(ids)
    started, game = time.monotonic(), None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            game = launch(map=Path(path), agents=slots, render=False, sound=False)
        rpc = game.rpc
        rpc.timeout = 20  # a replay from another patch or map version never leaves the start hold
        try:
            rpc.create_game(str(Path(path).absolute()), [{"slot": s, "control": "agent"} for s in slots])
        except TimeoutError:
            return {**row, "status": "no_load", "map_checksum": replay.map_checksum}
        rpc.timeout = 120
        rpc.debug("speed", factor=1024)
        rpc.debug("waitfloor", ms=1)
        shared, before, second, results = None, set(), 0, {}
        minutes = defaultdict(lambda: [0, 0])  # minute -> [missing, selected]
        while True:
            try:
                step = rpc.step(1000, observe=list(slots))
            except Exception:
                break  # the recording ended with the simulation
            shared = shared or SharedObservations(game.pid)
            units = set()
            for o in step["observations"]:
                observation = shared.read(o["offset"], o["size"])
                units.update(observation.units["unit_id"].tolist())
                if observation.result:
                    results[observation.player] = observation.result
            ids = selected.get(second, set())
            minutes[second // 60][0] += sum(1 for i in ids if i not in units and i not in before)
            minutes[second // 60][1] += len(ids)
            before, second = units, second + 1
            if step["reason"] != "target":
                break
        missing, total = (sum(m[i] for m in minutes.values()) for i in (0, 1))
        worst = max((m / n for m, n in minutes.values() if n >= 20), default=0.0)
        return {
            **row,
            "status": "ok",
            "played_seconds": second,
            "selected": total,
            "miss_rate": round(missing / max(total, 1), 4),
            "worst_minute_rate": round(worst, 3),
            "result": results,
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    except Exception as exc:
        return {
            **row,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
        }
    finally:
        if game is not None:
            game.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", type=Path, help="a folder of .w3g files")
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON lines, appended; replays already in it are skipped"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [str(p.absolute()) for p in sorted(args.replays.glob("*.w3g")) if p.stem not in done]
    print(f"{len(todo)} to check, {len(done)} already checked", flush=True)
    counts = defaultdict(int)
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("a") as out:
        for n, future in enumerate(as_completed([pool.submit(check, p) for p in todo]), 1):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            counts[row["status"]] += 1
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
