"""What a self-play run's folder holds besides log.txt and games.jsonl, written here and read by the run board
(training/board): run.json (how the run was launched), replays/ (a few fights a run, unit by unit) and, for the
whole runs/ folder, evals.jsonl (each fixed evaluation's results, training/bench/scenarios.py --record).

    run.json       {"learner": its arguments, "launch": run.py's, "commit": the code's, "started": unix seconds}
    replays/*.json {"update", "actor", "slot" (the learning player's), "opponent", "stage", "outcome",
                    "units": {id: [type, player, max_hp]},
                    "frames": [[game seconds, [[id, x, y, hp, order id, order x, order y], ...]], ...]}
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from wc3env.binary import DEAD, OWN, STRUCTURE, fourcc

ROOT = Path(__file__).resolve().parents[2]


def update_run(out: str | Path, **sections) -> None:
    """run.json with these sections added (a section's values JSON-safe: anything else as its string)."""
    path = Path(out) / "run.json"
    run = json.loads(path.read_text()) if path.exists() else {"started": round(time.time(), 1), "commit": _commit()}
    for name, values in sections.items():
        run[name] = {k: v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)
                     for k, v in values.items()}  # fmt: skip
    path.write_text(json.dumps(run, indent=1))


def _commit() -> str | None:
    """The code's git commit (with a "+" when it has uncommitted changes), or None outside a checkout."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()  # fmt: skip
        return head + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def memory_gb() -> float | None:
    """The memory of this process's cgroup that cannot be reclaimed, in GB: under a systemd scope (home runs) the
    whole run's, games included. Its anonymous memory, shared memory (the batches' files in /dev/shm) and kernel
    memory: what a scope's MemoryMax kills for. memory.current also counts the page cache (checkpoints just written),
    which the kernel drops first. None where it cannot be read."""
    try:
        group = Path("/proc/self/cgroup").read_text().split("::", 1)[1].strip()
        stat = dict(line.split() for line in Path(f"/sys/fs/cgroup{group}/memory.stat").read_text().splitlines())
        return sum(int(stat[k]) for k in ("anon", "shmem", "kernel")) / 1e9
    except (OSError, IndexError, ValueError, KeyError):
        return None


class Recording:
    """One fight, step by step, for replays/: each player's living units but buildings, as each sees its own."""

    def __init__(self, actor: int, slot: int, opponent: str, stage):
        self.replay = {"actor": actor, "slot": slot, "opponent": opponent, "stage": stage, "units": {}, "frames": []}

    def add(self, observations: dict) -> None:
        rows, units = [], self.replay["units"]
        for player, obs in observations.items():
            u = obs.units
            for r in u[((u["flags"] & OWN) != 0) & ((u["flags"] & (DEAD | STRUCTURE)) == 0)]:
                key = int(r["unit_id"])
                if str(key) not in units:
                    units[str(key)] = [fourcc(int(r["type_id"])), player, round(float(r["max_hp"]))]
                rows.append([key, *(round(float(r[k])) for k in ("x", "y", "hp")), int(r["order_id"]),
                             round(float(r["order_x"])), round(float(r["order_y"]))])  # fmt: skip
        self.replay["frames"].append([round(observations[0].game_time_seconds, 2), rows])

    def finish(self, outcome: str) -> dict:
        return {**self.replay, "outcome": outcome}


def write_replay(out: str | Path, replay: dict) -> None:
    """A recorded fight into the run's replays/ folder, named by the update it ended in and its actor."""
    folder = Path(out) / "replays"
    folder.mkdir(exist_ok=True)
    name = f"u{replay['update']:06d}-a{replay['actor']}-{int(time.time())}.json"
    tmp = folder / (name + ".tmp")
    tmp.write_text(json.dumps(replay, separators=(",", ":")))
    os.replace(tmp, folder / name)  # whole or not at all: the board may read the folder meanwhile
