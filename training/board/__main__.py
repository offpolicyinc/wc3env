"""The run board: every self-play run under runs/selfplay on one local page, live while a run trains.

    python -m training.board                 # http://localhost:8770 (--port, --runs, --host 0.0.0.0 for the LAN)

A small server over the run folders (training/selfplay/runfiles.py says what they hold), nothing else to install:
it reads each run's log.txt (the learner's update lines, parsed as training/train/track.py does for W&B, and the
actors' and inference processes' reports), games.jsonl (wins binned by update, with 95% intervals), run.json, its
checkpoints, replays/ and runs/evals.jsonl, and docs/training/rsi/log.md for the experiments that name the run.
A file is parsed again only when it changed; the page asks again every few seconds while a run is live.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..train.track import parse

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("index.html")
LIVE_SECONDS = 90  # a run whose log changed this recently is live
BIN = 10  # updates per bin of the win rate
REPORT = re.compile(r"^(\d\d:\d\d:\d\d) (actor (\d+)|inference): (.*)$")
PAIR = re.compile(r"([a-z_]+) (-?[0-9.]+)\b")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of k successes in n."""
    if n == 0:
        return 0.0, 1.0
    centre, half = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


class Cache:
    """A file's parse, kept until the file changes (its size and modification time)."""

    def __init__(self):
        self.entries: dict = {}

    def get(self, path: Path, read):
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        hit = self.entries.get(path)
        if hit is None or hit[0] != key:
            hit = self.entries[path] = (key, read(path))
        return hit[1]


def read_log(path: Path) -> dict:
    """log.txt: the update lines' values by update, the decided steps/s from the actors' step counters (as
    ~/wc3/tools/rate.py), the actors' and inference processes' latest timings, and the learner's notes."""
    updates: dict[int, dict] = {}
    day, last, first = 0.0, None, None
    actors: dict[str, int] = {}
    rate: list[list[float]] = []  # [minutes since the first report, decided steps/s over the last minute]
    totals: list[tuple[float, int]] = []
    reports: dict[str, dict] = {}
    notes: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        stamp = line[:8]
        try:
            t = datetime.strptime(stamp, "%H:%M:%S").timestamp() + day
        except ValueError:
            continue
        if last is not None and t < last - 3600:  # past midnight
            day += 86400
            t += 86400
        last = t
        first = t if first is None else first
        if parsed := parse(line):
            update, values = parsed
            updates[update] = {"minutes": round((t - first) / 60, 2), **values}
            continue
        if m := REPORT.match(line):
            who, body = ("inference" if m[3] is None else f"actor {m[3]}"), m[4]
            timings = body.split("; ms: ", 1)
            reports[who] = {
                "at": round((t - first) / 60, 2),
                "summary": timings[0],
                **{k: float(v) for k, v in PAIR.findall(timings[1] if len(timings) > 1 else "")},
            }
            if m[3] is not None and (steps := re.match(r"(\d+) game steps", body)):
                actors[m[3]] = int(steps[1])
                totals.append((t, sum(actors.values())))
                back = next((x for x in reversed(totals) if t - x[0] >= 60), None)
                if back is not None and t > back[0]:
                    rate.append([round((t - first) / 60, 2), round((totals[-1][1] - back[1]) / (t - back[0]))])
            continue
        text = line[9:]
        if not text.startswith(("update ", "actor ", "inference")):
            notes.append(line)
    keys = sorted({k for v in updates.values() for k in v})
    order = sorted(updates)
    return {
        "updates": order,
        "series": {k: [updates[u].get(k) for u in order] for k in keys},
        "rate": rate,
        "reports": reports,
        "notes": notes[-200:],
    }


def read_games(path: Path) -> dict:
    """games.jsonl: the ladder games' outcomes binned by the update they ended in, each bin's win rate with its
    interval and its stage; the scenarios' fights (actor --scenarios: "focus_fire/1") binned the same, each apart;
    the evaluation games' by policy."""
    bins: dict[int, list] = defaultdict(lambda: [0, 0, None])  # bin -> [won, games, last stage]
    scenarios: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
    evals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = won = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            g = json.loads(line)
        except ValueError:
            continue
        if "outcome" not in g:
            continue
        victory = g["outcome"] == "victory"
        if "eval" in g:
            evals[g["eval"]][0] += victory
            evals[g["eval"]][1] += 1
            continue
        scenario = "/" in g.get("opponent", "")
        b = (scenarios[g["opponent"]] if scenario else bins)[g.get("update", 0) // BIN]
        b[0] += victory
        b[1] += 1
        if g.get("stage"):
            b[2] = g["stage"][1]
        if not scenario:
            total, won = total + 1, won + victory
    return {
        "bins": binned(bins),
        "games": total,
        "won": won,
        "scenarios": {name: binned(b) for name, b in sorted(scenarios.items())},
        "evals": {p: {"won": v[0], "games": v[1]} for p, v in evals.items()},
    }


def binned(bins: dict) -> list:
    """Bins as rows [update, win rate, interval low, high, games, stage], the bin still filling left out."""
    order = sorted(bins)
    if len(order) > 2 and bins[order[-1]][1] < 0.5 * sorted(b[1] for b in bins.values())[len(bins) // 2]:
        order = order[:-1]  # the bin still filling: too few fights to read yet
    rows = []
    for k in order:
        w, n, stage = bins[k]
        lo, hi = wilson(w, n)
        rows.append([k * BIN + BIN / 2, round(w / n, 4), round(lo, 4), round(hi, 4), n, stage])
    return rows


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_experiments(path: Path) -> dict[str, list[dict]]:
    """docs/training/rsi/log.md: run name -> the experiments (E<n>, their hypothesis and verdict) that name it in
    their last column (names, and ranges as "s0-s6")."""
    out: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not re.match(r"\| E\d+ \|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 8:
            continue
        row = {"id": cells[0], "date": cells[1], "ideas": cells[2], "hypothesis": cells[3], "verdict": cells[-2]}
        names = set(re.findall(r"[A-Za-z][\w.]*", cells[-1]))
        for a, b in re.findall(r"([A-Za-z]+\d+)-([A-Za-z]*\d+)", cells[-1]):  # s0-s6
            prefix, lo = re.match(r"([A-Za-z]+)(\d+)", a).groups()
            hi = int(re.search(r"\d+", b)[0])
            names.update(f"{prefix}{i}" for i in range(int(lo), hi + 1))
        for name in names:
            out[name].append(row)
    return out


def owner(checkpoint: str) -> str | None:
    """The run a checkpoint belongs to: runs/selfplay/<run>/..., or runs/keep/<run>-<update>.pt (one kept aside)."""
    if m := re.search(r"runs/selfplay/([^/]+)/", checkpoint):
        return m[1]
    if m := re.search(r"runs/keep/(.+)-\d+\.pt$", checkpoint):
        return m[1]
    return None


class Board:
    def __init__(self, runs: Path):
        self.runs, self.cache = runs, Cache()

    def folders(self) -> list[Path]:
        root = self.runs / "selfplay"
        return [p for p in root.iterdir() if (p / "log.txt").exists()] if root.exists() else []

    def experiments(self) -> dict:
        return self.cache.get(ROOT / "docs/training/rsi/log.md", read_experiments) or {}

    def evals(self) -> list[dict]:
        def read(path: Path) -> list[dict]:
            out = []
            for line in path.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
            return out

        return self.cache.get(self.runs / "evals.jsonl", read) or []

    def parent(self, run: dict | None, notes: list[str]) -> str | None:
        """The run (or checkpoint) this one started from: its --start, else --init; a run from before run.json, the
        checkpoint its learner said it carried Adam's state from."""
        learner = (run or {}).get("learner", {})
        start = learner.get("start") or learner.get("init") or ""
        if not start:
            start = next((m[1] for line in notes if (m := re.search(r"carried from (\S+)", line))), "")
        m = re.search(r"runs/selfplay/([^/]+)/", start)
        return m[1] if m else start or None

    def summary(self, folder: Path) -> dict:
        log = self.cache.get(folder / "log.txt", read_log) or {"updates": [], "series": {}, "rate": [], "notes": []}
        games = self.cache.get(folder / "games.jsonl", read_games) or {"games": 0, "won": 0, "bins": []}
        run = self.cache.get(folder / "run.json", read_json)
        changed = (folder / "log.txt").stat().st_mtime
        series, last = log["series"], len(log["updates"]) - 1
        recent = games["bins"][-3:]
        won = sum(b[1] * b[4] for b in recent) / max(sum(b[4] for b in recent), 1) if recent else None
        return {
            "name": folder.name,
            "changed": changed,
            "live": time.time() - changed < LIVE_SECONDS,
            "started": (run or {}).get("started") or (folder / "log.txt").stat().st_ctime,
            "update": log["updates"][-1] if log["updates"] else 0,
            "games": games["games"],
            "won_recent": won,
            "handicap": series["play/handicap"][last] if last >= 0 and "play/handicap" in series else None,
            "steps_per_s": log["rate"][-1][1] if log["rate"] else None,
            "parent": self.parent(run, log["notes"]),
            "commit": (run or {}).get("commit"),
            "experiments": [e["id"] for e in self.experiments().get(folder.name, [])],
        }

    def runs_list(self) -> list[dict]:
        return sorted((self.summary(f) for f in self.folders()), key=lambda r: -r["changed"])

    def run(self, name: str) -> dict | None:
        folder = self.runs / "selfplay" / name
        if not (folder / "log.txt").exists():
            return None
        checkpoints = []
        for path in [folder / "last.pt", *sorted((folder / "pool").glob("*.pt"))]:
            if path.exists():
                stat = path.stat()
                checkpoints.append({"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                                    "mb": round(stat.st_size / 1e6), "saved": stat.st_mtime})  # fmt: skip
        mine = {c["path"] for c in checkpoints}
        evals = [e for e in self.evals() if any(owner(p) == name or p in mine for p in e.get("results", {}))]
        replays = (
            sorted((p.name for p in (folder / "replays").glob("*.json")), reverse=True)
            if (folder / "replays").exists()
            else []
        )
        return {
            **self.summary(folder),
            "run": self.cache.get(folder / "run.json", read_json),
            "log": self.cache.get(folder / "log.txt", read_log),
            "outcomes": self.cache.get(folder / "games.jsonl", read_games),
            "checkpoints": checkpoints,
            "evals": evals,
            "replays": replays,
            "experiment_rows": self.experiments().get(name, []),
        }

    def replay(self, name: str, file: str):
        if "/" in file or not file.endswith(".json"):
            return None
        return read_json(self.runs / "selfplay" / name / "replays" / file)


def handler(board: Board):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the page polls
            pass

        def send(self, status: int, body: bytes, kind: str) -> None:
            if len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, 5)
                encoded = True
            else:
                encoded = False
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            if encoded:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            try:
                if not parts:
                    return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "runs"]:
                    data = board.runs_list()
                elif parts == ["api", "evals"]:
                    data = [{**e, "owners": {p: owner(p) for p in e.get("results", {})}} for e in board.evals()]
                elif len(parts) == 3 and parts[:2] == ["api", "run"]:
                    data = board.run(parts[2])
                elif len(parts) == 4 and parts[:2] == ["api", "replay"]:
                    data = board.replay(parts[2], parts[3])
                else:
                    data = None
            except Exception as e:  # a half-written file, a run removed meanwhile: say so, keep serving
                return self.send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
            if data is None:
                return self.send(404, b'{"error": "not found"}', "application/json")
            self.send(200, json.dumps(data, separators=(",", ":"), allow_nan=False, default=str).encode(),
                      "application/json")  # fmt: skip

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(ROOT / "runs"), help="the runs folder (its selfplay/ and evals.jsonl)")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0: reachable from other machines on the network")
    a = p.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), handler(Board(Path(a.runs))))
    print(f"run board on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
