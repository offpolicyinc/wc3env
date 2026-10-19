"""Does another Wine play the same games? Warcraft's AI against itself on fixed seeds, played twice in a games sandbox of
each worker image, every step's state hashed: the two plays in one image must agree (else the games are not
reproducible and the check means nothing), then the images are compared step by step.

    python -m training.bench.wine_parity --images build/modal-image-stable.txt build/modal-image.txt
"""

import argparse
import json
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud import selfplay  # noqa: E402
from training.cloud.scripts import _preamble, low_video  # noqa: E402

PLAY = r"""
import hashlib, json, sys
from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, wine_workers
games, steps = int(sys.argv[1]), int(sys.argv[2])
configs = [GameConfig(map="Maps/FrozenThrone/(2)EchoIsles.w3x", players=(PlayerConfig(0, "orc"), PlayerConfig(1, "human")),
                      ai_agents=(0, 1), setup=MatchSetup(seed=k), render=False, sound=False, step_ms=250, observation="binary")
           for k in range(games)]
def state(o):  # what the game computed: time, economy, score, every unit's row
    return hashlib.sha1(json.dumps([o.game_time_seconds, o.gold, o.lumber, o.food_used, o.food_cap, o.score], sort_keys=True).encode()
                        + o.units.tobytes()).hexdigest()[:12]
with VectorSession(configs, group_size=2, launch=wine_workers()) as s:
    first = s.reset()
    log = {i: [] for i in range(games)}
    for _ in range(steps):
        for i in range(games): s.send(i, {0: [], 1: []})
        for _ in range(games):
            i, obs, done, info = s.recv()
            log[i].append([state(obs[0]), state(obs[1]), len(obs[0].units), len(obs[1].units)])
print(json.dumps(log))
"""


def play(image_file: Path, games: int, steps: int, out: dict) -> None:
    """Both plays in one sandbox of the image named in `image_file`."""
    _, stream, _code, _send = selfplay._local()
    made: list = []
    try:
        image = selfplay.games_image(image_file.read_text().strip())
        box, at = selfplay.games_sandbox(types.SimpleNamespace(cpu=4, memory=0, games=games, per_box=1, rest=["--hours", "1"]),
                                         image, _code(), made)  # fmt: skip
        _send(box, "/work/play.py", PLAY.encode())
        script = (
            _preamble("/work:/opt/host", "/sessions/out")
            + low_video()
            + "".join(
                f"python3 /work/play.py {games} {steps} > /sessions/out/play{k}.json 2>> /sessions/out/play.err\n"
                for k in (0, 1)
            )
        )
        _send(box, "/work/play.sh", script.encode())
        stream(box, "chown 10001:10001 /work/play.sh /work/play.py", root=True)
        p = box.exec("bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/play.sh' > /dev/null 2>&1; "
                     "wine --version; cat /sessions/out/play0.json; echo; cat /sessions/out/play1.json; echo; tail -5 /sessions/out/play.err")  # fmt: skip
        p.wait()
        lines = p.stdout.read().splitlines()
        plays = [json.loads(x) for x in lines[1:] if x.startswith("{")]
        out[image_file.name] = {
            "at": at,
            "wine": lines[0],
            "plays": plays,
            "err": [x for x in lines[1:] if x and not x.startswith("{")],
        }
    finally:
        for box in made:
            box.terminate()


def compare(a: dict, b: dict) -> str:
    """Steps that agree before the first difference, per game."""
    parts = []
    for g in a:
        n = next((k for k, (x, y) in enumerate(zip(a[g], b[g])) if x != y), None)
        parts.append(
            f"game {g}: "
            + ("all " + str(len(a[g])) + " agree" if n is None else f"differ from step {n} ({n / 4:.0f} s)")
        )
    return "; ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="+", type=Path, required=True, help="files holding worker image ids")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--steps", type=int, default=1200, help="250 ms each: 1200 is 5 minutes of game time")
    a = p.parse_args()
    out: dict = {}
    threads = [threading.Thread(target=play, args=(f, a.games, a.steps, out)) for f in a.images]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name, r in out.items():
        print(
            f"{name} ({r['wine']}, {r['at']}):",
            "two plays: " + compare(*r["plays"]) if len(r["plays"]) == 2 else f"failed: {r['err']}",
        )
    names = [n for n in out if out[n]["plays"]]
    for x, y in zip(names, names[1:]):
        print(f"{x} vs {y}:", compare(out[x]["plays"][0], out[y]["plays"][0]))


if __name__ == "__main__":
    main()
