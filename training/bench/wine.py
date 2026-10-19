"""The games' own cost under Wine, in a Modal sandbox like self-play's (4 cores, 32 GiB): N no-op games stepped
(250 ms steps, speed 2048) through wine_workers with no model and no network; per run the wall time per step,
the sandbox's CPU time per game-step (from /proc/stat) and how busy its cores were."""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud.scripts import _preamble  # noqa: E402
from training.cloud.selfplay import _local, games_image, games_sandbox  # noqa: E402

BENCH = r"""
import os, sys, time
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, wine_workers, link_prefix
from pathlib import Path
games, steps = int(sys.argv[1]), int(sys.argv[2])
def cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]  # all, idle + iowait (jiffies)
configs = [GameConfig(map="Maps/FrozenThrone/(2)EchoIsles.w3x", players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human")),
                      setup=MatchSetup(seed=k), render=False, sound=False, step_ms=250, observation="binary") for k in range(games)]
with VectorSession(configs, group_size=2, launch=wine_workers()) as s:
    s.reset()
    for _ in range(20):  # warm up
        for i in range(games): s.send(i, {0: [], 1: []})
        for _ in range(games): s.recv()
    a, t = cpu(), time.time()
    for _ in range(steps):
        for i in range(games): s.send(i, {0: [], 1: []})
        for _ in range(games): s.recv()
    b, dt = cpu(), time.time() - t
busy = (b[0] - a[0]) - (b[1] - a[1])
hz, ncpu = os.sysconf("SC_CLK_TCK"), os.cpu_count()
print(f"{games} games: {dt / steps * 1000:.1f} ms wall per step, {games * steps / dt:.0f} game-steps/s; "
      f"CPU {busy / hz / (games * steps) * 1000:.1f} ms per game-step; cores busy {busy / (b[0] - a[0]) * 100:.0f}% of {ncpu} threads")
"""


def main() -> None:
    _, stream, _code, _send = _local()
    # self-play's games sandbox (the licence and this checkout's code), 4 cores and 32 GiB
    args = types.SimpleNamespace(cpu=4, memory=32768, games=0, per_box=1, rest=["--hours", "0.5"])
    made: list = []  # stopped however this ends
    try:
        box, _ = games_sandbox(args, games_image(), _code(), made)
        _send(box, "/work/bench.py", BENCH.encode())
        script = (
            _preamble("/work:/opt/host", "/sessions/out")
            + "\n".join(
                f"python3 /work/bench.py {n} 300 >> /sessions/out/bench.txt 2>> /sessions/out/bench.err"
                for n in (1, 2, 8, 16, 32)
            )
            + "\n"
        )
        _send(box, "/work/bench.sh", script.encode())
        stream(box, "chown 10001:10001 /work/bench.sh", root=True)
        p = box.exec("bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/bench.sh'; "
                     "cat /sessions/out/bench.txt; tail -5 /sessions/out/bench.err")  # fmt: skip
        p.wait()
        print(p.stdout.read(), p.stderr.read()[-500:])
    finally:
        for box in made:
            box.terminate()


if __name__ == "__main__":
    main()
