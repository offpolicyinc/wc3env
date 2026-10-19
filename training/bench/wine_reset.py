"""Do games survive their next game's start (reset) in one games sandbox? sp33's games (Proton 11 Wine, fsync) died
starting their next game after the 15-minute limit, 69 times in 36 minutes; sp31's (WineHQ stable) about once a run.

N games against the easy AI, each stepped STEPS steps (250 ms) and reset, ROUNDS times; reports the resets that
failed ("pipe closed": the game's process died), under the image's Wine as is (a new game in the same process) and
with a new process for every game (GameConfig.max_episodes_per_process=1).

    python -m training.bench.wine_reset --games 32 --steps 1200 --rounds 3
"""

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud.scripts import _preamble, low_video  # noqa: E402
from training.cloud.selfplay import _local, games_image, games_sandbox  # noqa: E402

BENCH = r"""
import sys, time
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, wine_workers
games, steps, rounds, per_process = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) or None
configs = [GameConfig(map="Maps/FrozenThrone/(2)EchoIsles.w3x", players=(PlayerConfig(0, "orc"), PlayerConfig(1, "orc")),
                      ai_agents=(1,), ai_difficulty=0, setup=MatchSetup(seed=k, randomize_starts=True), render=False,
                      sound=False, step_ms=250, observation="binary", max_episodes_per_process=per_process) for k in range(games)]
failed = 0
began = time.time()
with VectorSession(configs, group_size=2, launch=wine_workers()) as s:
    s.reset()
    for r in range(rounds):
        for _ in range(steps):
            for i in range(games): s.send(i, {0: [], 1: []})
            for _ in range(games): s.recv()
        for i in range(games):
            try:
                s.reset_one(i)
            except RuntimeError as e:
                failed += 1
                print(f"round {r} game {i}: {e}", flush=True)
                s.reset_one(i)
        print(f"round {r}: {failed} failed resets so far, {time.time() - began:.0f} s", flush=True)
print(f"RESULT per_process={per_process}: {failed} of {games * rounds} resets failed", flush=True)
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--steps", type=int, default=1200, help="per game between resets (1200: 5 game minutes)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--cpu", type=float, default=16)
    p.add_argument("--per-process", default="0,1", help="max_episodes_per_process values to try in turn (0: unlimited)")
    a = p.parse_args()
    _, stream, _code, _send = _local()
    args = types.SimpleNamespace(cpu=a.cpu, memory=0, games=a.games, per_box=1, rest=["--hours", "1"])
    made: list = []
    try:
        box, at = games_sandbox(args, games_image(), _code(), made)
        print("sandbox in", at, flush=True)
        _send(box, "/work/bench.py", BENCH.encode())
        script = _preamble("/work:/opt/host", "/sessions/out") + low_video()
        for n in a.per_process.split(","):
            script += (
                f"python3 /work/bench.py {a.games} {a.steps} {a.rounds} {n} >> /sessions/out/bench.txt "
                f"2>> /sessions/out/err-{n}.txt || echo per_process={n} crashed >> /sessions/out/bench.txt\n"
                "for p in $WINEPREFIX $TMPDIR/wine-prefix-*; do WINEPREFIX=$p wineserver -k 2>/dev/null || true; done; sleep 5\n"
            )
        _send(box, "/work/bench.sh", script.encode())
        stream(box, "chown 10001:10001 /work/bench.sh /work/bench.py", root=True)
        proc = box.exec("bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/bench.sh' 2>&1 | tail -5; "
                        "cat /sessions/out/bench.txt; for f in /sessions/out/err-*.txt; do echo == $f; grep -E -i 'err:|exception|fault|^wine:|Error' $f | "
                        "sort | uniq -c | sort -rn | head -15; done")  # fmt: skip
        proc.wait()
        print(proc.stdout.read(), proc.stderr.read()[-2000:])
    finally:
        for box in made:
            box.terminate()


if __name__ == "__main__":
    main()
