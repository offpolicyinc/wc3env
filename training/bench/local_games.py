"""Variants of the games' setup on this Linux machine, A/B: the same seeded games (Warcraft's AI against itself on Echo
Isles, render off) played under each variant in turn, rounds alternating so every variant sees the same load. Per
variant: CPU per game-step of its own processes (the games, Windows Python, wineserver: every process whose
WINEPREFIX is one of its prefixes), early and late in the games (a cost that grows over a game shows there), wall time
per step, and whether every step's state (time, economy, score, every unit's row) hashes the same as the first
variant's.

    python -m training.bench.local_games --games 16 --steps 2400 --rounds 2 \
        --variant base --variant cuts:cuts=1225 \
        --variant ge:prefix=~/.wine-wc3-ge,wine=~/wc3/ge/GE-Proton10-34/files/bin,fsync=1

A variant is name[:key=value,...]: prefix (a Wine prefix: default $WINEPREFIX), wine (a Wine build's bin folder, as
$WC3_WINE; "": the one on PATH), fsync (0/1, as $WINEFSYNC), noaudio (1: Wine's audio drivers off), hook (wc3hook.dll for the games, as WC3_HOOK_DLL), cuts
(GameConfig.game_cuts), env.NAME (any variable for the games).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.cloud.scripts import LOW_VIDEO  # noqa: E402
from wc3env.session import GameConfig, MatchSetup, PlayerConfig  # noqa: E402
from wc3env.vector import VectorSession, wine_env, wine_workers  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")
# Xvfb's GLX through Mesa's software rasterizer, as the worker image (docker/Dockerfile); with NVIDIA's driver
# installed, libglvnd would otherwise pick NVIDIA's GLX, which Xvfb lacks ("unable to initialize DirectX")
GL = {
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "GALLIUM_DRIVER": "llvmpipe",
    "LP_NUM_THREADS": "2",
    "__GLX_VENDOR_LIBRARY_NAME": "mesa",
}
_low = set()


def low_video(base: Path, env: dict) -> None:
    """The lowest video options in `base` (as self-play's sandboxes have them), before it is copied."""
    if str(base) in _low:
        return
    key = r"HKCU\Software\Blizzard Entertainment\Warcraft III\Video"
    env = {**env, "WINEPREFIX": str(base)}
    for name, value in LOW_VIDEO.items():
        subprocess.run(["wine", "reg", "add", key, "/v", name, "/t", "REG_DWORD", "/d", str(value), "/f"], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # fmt: skip
    # ends the prefix's processes, which writes its registry to disk before the copies are made (wineserver -w would
    # wait for services left running by earlier games, for ever)
    subprocess.run(["wineserver", "-k"], env=env, check=False)
    _low.add(str(base))


def parse(spec: str) -> dict:
    name, _, rest = spec.partition(":")
    v = {"name": name, "env": {}}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        if key.startswith("env."):
            v["env"][key[4:]] = value
        else:
            v[key] = value
    return v


def cpu_of(prefixes: set[str]) -> float:
    """CPU seconds so far of every live process whose WINEPREFIX is one of `prefixes`."""
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            prefix = next((e[11:].decode() for e in env if e.startswith(b"WINEPREFIX=")), None)
            if prefix not in prefixes:
                continue
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            total += int(fields[11]) + int(fields[12])  # utime, stime
        except (OSError, IndexError, ValueError):
            continue
    return total / TICK


def state(o) -> str:
    return hashlib.sha1(json.dumps([o.game_time_seconds, o.gold, o.lumber, o.food_used, o.food_cap, o.score], sort_keys=True).encode()
                        + o.units.tobytes()).hexdigest()[:12]  # fmt: skip


def available_gib() -> float:
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    return int(line.split()[1]) / 2**20


def play(v: dict, games: int, steps: int, group: int, scratch: Path, resets: int = 0) -> dict:
    need = 0.5 * games + 4  # ~0.3 GiB a game on Echo Isles (9-benchmarks.md), with room for armies and the rest
    if available_gib() < need:  # other work on the machine: an OOM kill takes whatever it finds
        raise SystemExit(f"{available_gib():.1f} GiB available, {need:.0f} needed for {games} games; not starting")
    base = Path(os.path.expanduser(v.get("prefix") or os.environ["WINEPREFIX"]))
    os.environ["TMPDIR"] = str(scratch / v["name"])  # this variant's copies of the prefix
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = None
    for k, val in v["env"].items():
        os.environ[k] = val
    if v.get("noaudio") == "1":  # Wine's audio drivers off (the games are silent; Proton's audio codec setup hangs)
        os.environ["WINEDLLOVERRIDES"] = "mscoree,mshtml=;winealsa.drv,winepulse.drv,wineoss.drv=d"
    if v.get("hook"):
        os.environ["WC3_HOOK_DLL"] = "Z:" + str(Path(v["hook"]).expanduser().resolve()).replace("/", "\\")
    if "wine" in v:
        os.environ["WC3_WINE"] = os.path.expanduser(v["wine"])
    if "fsync" in v:
        os.environ["WINEFSYNC"] = v["fsync"]
    low_video(base, wine_env(os.environ.get("WC3_WINE")))
    launch = wine_workers(str(base))
    workers = -(-games // group)
    prefixes = {str(base)} | {
        str(Path(tempfile.gettempdir()) / f"wine-prefix-{base.name}-{w}") for w in range(1, workers)
    }
    configs = [GameConfig(map="Maps/FrozenThrone/(2)EchoIsles.w3x", players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human")),
                          ai_agents=(0, 1), setup=MatchSetup(seed=k), render=False, sound=False, step_ms=250, observation="binary",
                          game_cuts=int(v.get("cuts", 0)))
               for k in range(games)]  # fmt: skip
    hashes: dict[int, list] = {i: [] for i in range(games)}
    marks = []  # (step, cpu, wall) at the start, the middle and the end
    try:
        with VectorSession(configs, group_size=group, launch=launch) as s:
            s.reset()
            for step in range(steps):
                if step in (20, steps // 2, steps - 1):
                    marks.append((step, cpu_of(prefixes), time.time()))
                for i in range(games):
                    s.send(i, {0: [], 1: []})
                for _ in range(games):
                    i, obs, done, info = s.recv()
                    hashes[i].append(state(obs[0]) + state(obs[1]))
            failed = 0
            for _ in range(resets):  # map reloads in the running games (Proton 11's games died at them on Modal)
                for i in range(games):
                    try:
                        s.reset_one(i)
                    except RuntimeError:
                        failed += 1
                        s.reset_one(i)  # as the actor does: once more, in a new process
                for _ in range(40):
                    for i in range(games):
                        s.send(i, {0: [], 1: []})
                    for _ in range(games):
                        s.recv()
    finally:
        for k in v["env"]:
            os.environ.pop(k, None)
        os.environ["WINEDLLOVERRIDES"] = "mscoree,mshtml="
        os.environ.pop("WC3_HOOK_DLL", None)
    (s0, c0, w0), (s1, c1, w1), (s2, c2, w2) = marks
    return {"early_cpu_ms": 1000 * (c1 - c0) / ((s1 - s0) * games), "late_cpu_ms": 1000 * (c2 - c1) / ((s2 - s1) * games),
            "wall_ms": 1000 * (w2 - w0) / (s2 - s0), "failed_resets": failed, "hashes": hashes}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--steps", type=int, default=2400, help="per game (2400: 10 game minutes)")
    p.add_argument("--group", type=int, default=2, help="games per worker")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument(
        "--resets", type=int, default=0, help="then this many map reloads of every game, 40 steps after each"
    )
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--out", default="", help="a JSON file for the results")
    a = p.parse_args()
    variants = [parse(x) for x in a.variant]
    scratch = Path(tempfile.mkdtemp(prefix="local-games-"))
    display = None
    if not os.environ.get("DISPLAY"):  # a virtual display of its own, as docker/with-display.sh
        display = subprocess.Popen(["Xvfb", ":93", "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-maxclients", "2048"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        os.environ["DISPLAY"] = ":93"
        time.sleep(1)
    os.environ.update(
        {
            "WINEDEBUG": "-all",
            "WINEDLLOVERRIDES": "mscoree,mshtml=",
            "WC3_GAME_DIR": "C:\\wc3",
            "WC3_USER_DIR": "C:\\scratch\\Warcraft III",
            **GL,
        }
    )
    results: dict[str, list] = {v["name"]: [] for v in variants}
    common = dict(os.environ)  # each variant sets its own variables on top of these
    try:
        for r in range(a.rounds):
            for v in variants:
                os.environ.clear()
                os.environ.update(common)
                out = play(v, a.games, a.steps, a.group, scratch, a.resets)
                results[v["name"]].append(out)
                print(f"round {r} {v['name']}: cpu/game-step {out['early_cpu_ms']:.2f} ms early, {out['late_cpu_ms']:.2f} late; "
                      f"wall/step {out['wall_ms']:.1f} ms; {out['failed_resets']} failed resets", flush=True)  # fmt: skip
    finally:
        if display:
            display.terminate()
    reference = results[variants[0]["name"]][0]["hashes"]
    summary = {}
    for name, runs in results.items():
        same = [sum(x == y for x, y in zip(run["hashes"][i], reference[i])) for run in runs for i in reference]
        first_diff = min((next((k for k, (x, y) in enumerate(zip(run["hashes"][i], reference[i])) if x != y), a.steps)
                          for run in runs for i in reference), default=a.steps)  # fmt: skip
        summary[name] = {k: sum(r[k] for r in runs) / len(runs) for k in ("early_cpu_ms", "late_cpu_ms", "wall_ms")}
        summary[name]["steps_identical"] = f"{sum(same)} of {len(same) * a.steps}"
        summary[name]["first_difference"] = first_diff
        summary[name]["failed_resets"] = f"{sum(r['failed_resets'] for r in runs)} of {a.resets * a.games * len(runs)}"
        print(f"{name}: {json.dumps(summary[name])}")
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
