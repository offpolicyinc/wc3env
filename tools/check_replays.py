"""Play replays to their end and check each stays in sync with the recorded game.

    python tools/check_replays.py runs/replays/1.29 --out runs/replays/1.29/check.jsonl --workers 6

A replay stores only inputs, and its commands name units by the engine's object ids. In sync, every order a
player gave reaches their units: the observation's orders show each player command (origin `player`) at the
game time the replay gives it (its clock less any paused time). A playback that drifts from the recorded game
shows up as order actions that reach no unit. Each minute's order actions are matched to the commands the
playback gave (same player and order, within a turn). The playback ends when the first player leaves; later
commands are not checked. Per replay, one JSON line: status `ok` with the miss rate, the worst minute's rate,
`synced_seconds` (the start of the first minute with 20 or more order actions of which more than a quarter
reached nothing; the whole game when there is none) and the final results; `map_missing`; `no_load` (never
leaves the start hold: recorded on another patch or map version); or `error`. In sync, 1-2% miss: commands
to units the player does not own, units that just died, or orders the game refused.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import io
import json
import struct
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from wc3env import w3g
from wc3env.settings import settings

MATCH_MS = (-500, 1000)  # a turn's actions carry its end time and run during it; the capture lags a little
DESYNC = 0.25


def check(path: str) -> dict:
    from wc3env.binary import ORIGINS, SharedObservations
    from wc3env.game import launch

    replay = w3g.read(path)
    players = replay.opponents
    slots = tuple(p.slot for p in players)
    row = {"id": Path(path).stem, "map": replay.map, "seconds": replay.duration_ms // 1000, "slots": slots}
    if not (settings().game_dir / replay.map.replace("\\", "/")).is_file():
        return {**row, "status": "map_missing"}
    slot_of = {p.id: p.slot for p in players}  # observers' actions reach no unit
    wanted = [  # at game time: the replay's clock also counts paused time
        (slot_of[pid], replay.game_ms(t), struct.unpack_from("<I", body, 2)[0])
        for t, pid, a, body in replay.actions()
        if pid in slot_of and 0x10 <= a <= 0x14
    ]
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
        shared, seconds, results = None, 0, {}
        given = defaultdict(set)  # (slot, order id) -> game times of the player's commands
        while True:
            try:
                step = rpc.step(1000, observe=list(slots))
            except Exception:
                break  # the recording ended with the simulation
            shared = shared or SharedObservations(game.pid)
            for o in step["observations"]:
                observation = shared.read(o["offset"], o["size"])
                for r in observation.orders[observation.orders["origin"] == ORIGINS.index("player")]:
                    given[(observation.player, int(r["order_id"]))].add(int(r["time_ms"]))
                if observation.result:
                    results[observation.player] = observation.result
            seconds += 1
            if step["reason"] != "target":
                break
        times = {key: sorted(v) for key, v in given.items()}
        minutes = defaultdict(lambda: [0, 0])  # minute -> [missing, order actions]
        for slot, t, order in (w for w in wanted if w[1] < 1000 * seconds):  # the rest were never played
            ts = times.get((slot, order), [])
            i = bisect.bisect_left(ts, t + MATCH_MS[0])
            minutes[t // 60000][0] += not (i < len(ts) and ts[i] <= t + MATCH_MS[1])
            minutes[t // 60000][1] += 1
        missing, total = (sum(m[i] for m in minutes.values()) for i in (0, 1))
        lost = [m for m, (miss, n) in sorted(minutes.items()) if n >= 20 and miss / n > DESYNC]
        return {
            **row,
            "status": "ok",
            "played_seconds": seconds,
            "order_actions": total,
            "miss_rate": round(missing / max(total, 1), 4),
            "worst_minute_rate": round(max((m / n for m, n in minutes.values() if n >= 20), default=0.0), 3),
            "synced_seconds": min(60 * lost[0], seconds) if lost else seconds,
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
