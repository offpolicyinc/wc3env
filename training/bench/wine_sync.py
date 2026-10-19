"""The games' CPU under other Wines, in one games sandbox: N no-op games stepped (250 ms steps) with a wait after each
round like self-play's loop, under each variant in turn, and the sandbox's CPU time per game-step (from /proc/stat).

Variants: WineHQ stable 11.0 (the image's, today's), the same without Wine's HID/USB/Bluetooth bus drivers (their
event loops wake every few ms in every game), and other Wine 11 builds with fsync: Wine's events, mutexes and waits on
futexes (futex_waitv) instead of a wineserver round trip each. Every Wine here is a wow64 build (32-bit Windows code in
64-bit Linux processes): Modal's kernel runs no 32-bit processes (a 32-bit futex call fails with ENOSYS, glibc aborts).

Tried and dropped: GE-Proton 10-34's bin-wow64 (Wine 10; Warcraft III page-faults in Wine's 32-bit code, fsync on or
off; its ucrtbase also lacks crealf, which numpy 2 calls) and GE-Proton 11 (no wow64 build).

    python -m training.bench.wine_sync --games 64 --steps 300 --wait-ms 50 --cpu 16 --only stable,proton-fsync,stable,proton-fsync
"""

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud.scripts import _preamble, low_video  # noqa: E402
from training.cloud.selfplay import _local, games_image, games_sandbox  # noqa: E402

# other Wines (Kron4ek's builds of Valve's Proton Wine and of Wine staging with TkG's patches): name -> its folder
KRON4EK = "https://github.com/Kron4ek/Wine-Builds/releases/download"
WINES = {
    "proton": ("proton-11.0-2", "wine-proton-11.0-2-amd64-wow64"),
    "tkg": ("11.18", "wine-11.18-staging-tkg-amd64-wow64"),
}
SITE = r"drive_c/Python311/Lib/site-packages/wc3env"
WINEDEBUG = ""
MAXCLIENTS = 0
NO_DRIVERS = ";winebus.sys,wineusb.sys,winebth.sys=d"

BENCH = r"""
import os, sys, time
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, wine_workers
games, steps, wait = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 1000
def cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]  # all, idle + iowait (jiffies)
configs = [GameConfig(map="Maps/FrozenThrone/(2)EchoIsles.w3x", players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human")),
                      setup=MatchSetup(seed=k), render=False, sound=False, step_ms=250, observation="binary") for k in range(games)]
with VectorSession(configs, group_size=2, launch=wine_workers()) as s:
    s.reset()
    def rounds(n):
        for _ in range(n):
            for i in range(games): s.send(i, {0: [], 1: []})
            for _ in range(games): s.recv()
            time.sleep(wait)
    rounds(40)  # warm up
    a, t = cpu(), time.time()
    rounds(steps)
    b, dt = cpu(), time.time() - t
busy = (b[0] - a[0]) - (b[1] - a[1])
hz, ncpu = os.sysconf("SC_CLK_TCK"), os.cpu_count()
work = dt / steps * 1000 - wait * 1000
print(f"{os.environ['VARIANT']:<22} {games} games: {work:.1f} ms a round besides the wait, "
      f"CPU {busy / hz / (games * steps) * 1000:.2f} ms a game-step, {busy / hz / dt:.1f} of {ncpu} threads busy", flush=True)
"""


def wine_env(wine: str) -> str:
    """Shell lines using another Wine and its own prefix (a prefix belongs to the Wine that made it)."""
    return f"export PATH=/opt/wines/{WINES[wine][1]}/bin:$PATH WINEPREFIX=/home/worker/prefix-{wine}\n"


def make_prefix(wine: str) -> str:
    """Shell lines making `wine`'s prefix like the image's: Python, the image's wheels, this checkout's wc3env."""
    return (
        wine_env(wine)
        + rf"""wine wineboot -u 2>&1 | tail -3; wineserver -w
wine 'Z:\opt\python-installer.exe' /quiet InstallAllUsers=0 'TargetDir=C:\Python311' Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0 Include_pip=1 > /dev/null 2>&1
for wheel in /opt/wheels/environment/*.whl; do wine 'C:\Python311\python.exe' -m pip install --no-index --no-deps "$(wine winepath -w "$wheel")" > /dev/null 2>&1; done
cp -r /opt/env/wc3env/. $WINEPREFIX/{SITE}/
ln -sfn /opt/game $WINEPREFIX/drive_c/wc3
{low_video()}
wine 'C:\Python311\python.exe' -c 'import numpy, wc3env; print("{wine} prefix: ok")'
wineserver -w
"""
    )


def variant(name: str, wine: str, fsync: bool, drivers: bool, games: int, steps: int, wait: int) -> str:
    """Shell lines running the bench once under one Wine, its worker prefixes made from that Wine's own."""
    lines = [f"# {name}", "(", f"export VARIANT={name} TMPDIR=/home/worker/tmp-{name}", "mkdir -p $TMPDIR"]
    if wine:
        lines += [wine_env(wine).strip(), f"export WINEFSYNC={int(fsync)}"]
    if WINEDEBUG:
        lines.append(f"export WINEDEBUG={WINEDEBUG}")
    if not drivers:
        lines.append(f'export WINEDLLOVERRIDES="$WINEDLLOVERRIDES{NO_DRIVERS}"')
    if MAXCLIENTS:  # a display of its own that takes more X clients than Xvfb's default 256
        lines += [
            f"Xvfb :98 -screen 0 1024x768x24 -nolisten tcp -maxclients {MAXCLIENTS} > /tmp/xvfb98.log 2>&1 &",
            "XPID=$!; sleep 2",
            "export DISPLAY=:98",
        ]
    else:
        lines.append(
            "XPID=$(pgrep -x Xvfb 2>/dev/null || ls -l /proc/*/exe 2>/dev/null | grep Xvfb | sed 's#.*/proc/\\([0-9]*\\)/exe.*#\\1#' | head -1)"
        )
    # the display's connections (its sockets) 90 s in: the games are up by then
    lines += [
        f'(sleep 90; echo "{name}: $(ls -l /proc/$XPID/fd 2>/dev/null | grep -c socket) X connections" >> /sessions/out/bench.txt) &',
        f"python3 /work/bench.py {games} {steps} {wait} >> /sessions/out/bench.txt 2>> /sessions/out/{name}.err || echo {name} failed",
        # every worker prefix has a wineserver of its own: a failed run leaves games in all of them
        "for p in $WINEPREFIX $TMPDIR/wine-prefix-*; do WINEPREFIX=$p wineserver -k 2>/dev/null || true; done; sleep 5",
    ]
    if MAXCLIENTS:
        lines.append("kill $XPID 2>/dev/null || true; sleep 1")
    lines.append(")")
    return "\n".join(lines) + "\n"


VARIANTS = [  # name, Wine ("": the image's), fsync, the bus drivers loaded
    ("stable", "", False, True),
    ("stable-nodrivers", "", False, False),
    ("proton", "proton", False, False),
    ("proton-fsync", "proton", True, False),
    ("tkg-fsync", "tkg", True, False),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=64, help="self-play's sandboxes: 2 actors of 32")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--wait-ms", type=int, default=50, help="after each round, like the loop's tunnel and inference")
    p.add_argument("--cpu", type=float, default=16)
    p.add_argument("--only", default="", help="comma-separated variant names, run in this order (repeats allowed)")
    p.add_argument("--winedebug", default="", help="WINEDEBUG for the games, to debug a variant (e.g. err+all)")
    p.add_argument(
        "--maxclients", type=int, default=0, help="the games on an Xvfb of their own taking this many X clients"
    )
    a = p.parse_args()
    global WINEDEBUG, MAXCLIENTS
    WINEDEBUG, MAXCLIENTS = a.winedebug, a.maxclients
    # --only in its order, repeats allowed (stable,proton-fsync,stable,proton-fsync: each twice, alternating)
    by_name = {v[0]: v for v in VARIANTS}
    variants = [(f"{n}-{k}", *by_name[n][1:]) for k, n in enumerate(a.only.split(","))] if a.only else VARIANTS
    wines = sorted({v[1] for v in variants if v[1]})
    _, stream, _code, _send = _local()
    args = types.SimpleNamespace(cpu=a.cpu, memory=0, games=a.games, per_box=1, rest=["--hours", "1"])
    made: list = []  # stopped however this ends
    try:
        box, at = games_sandbox(args, games_image(), _code(), made)
        print("sandbox in", at, flush=True)
        fetch = " && ".join(f"curl -sSL {KRON4EK}/{WINES[w][0]}/{WINES[w][1]}.tar.xz | tar xJ" for w in wines)
        if fetch:
            stream(box, "(command -v xz > /dev/null || (apt-get update -qq && apt-get install -y -qq xz-utils > /dev/null)) && "
                        f"mkdir -p /opt/wines && cd /opt/wines && {fetch} && chown -R 10001:10001 /opt/wines && ls /opt/wines", root=True)  # fmt: skip
        _send(box, "/work/bench.py", BENCH.encode())
        script = _preamble("/work:/opt/host", "/sessions/out") + low_video()
        script += "".join(f"(\n{make_prefix(w)}) || echo {w} prefix failed\n" for w in wines)
        script += "".join(variant(n, w, fs, dr, a.games, a.steps, a.wait_ms) for n, w, fs, dr in variants)
        _send(box, "/work/bench.sh", script.encode())
        stream(box, "chown 10001:10001 /work/bench.sh /work/bench.py", root=True)
        p = box.exec("bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/bench.sh' 2>&1 | tail -20; "
                     "cat /sessions/out/bench.txt; for f in /sessions/out/*.err; do echo == $f; grep -v -e '^$' -e VkInstance $f | "
                     "grep -E -i 'err:|exception|fault|^wine:|sync|Error' | sort | uniq -c | sort -rn | head -20; done")  # fmt: skip
        p.wait()
        print(p.stdout.read(), p.stderr.read()[-2000:])
    finally:
        for box in made:
            box.terminate()


if __name__ == "__main__":
    main()
