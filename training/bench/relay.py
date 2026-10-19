"""The games' side of a self-play step, hop by hop: one games sandbox as self-play makes it (2 bridges x 32 games,
features from the hook) and a CPU container in place of the GPU container's 2 actors, stepping every game with no
moves and a pause per step like self-play's loop. Reports the median and p90 ms of what wc3env.vector's
info["hops"] has: the round trip from the host's socket (h_round), the reply's wait on the host (h_queue) and the
game's step in its worker (w_step); the rest of the round trip is the tunnel and the bridge (~1 ms).

    python -m training.bench.relay --seconds 300 --pause 70
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud import selfplay  # noqa: E402

app = modal.App("wc3-bench-relay", image=selfplay.image)
PORT = 7650
MAP = "Maps/FrozenThrone/(2)EchoIsles.w3x"


def _session(port: int, games: int, seconds: float, pause: float, seed: int) -> dict:
    """One actor's games through one bridge: {part: [ms, ...]} and the steps taken."""
    import heapq

    from training.policy.features import PAIRS, STEP_MS, Static, env_race
    from training.policy.native import hook_spec
    from wc3env.session import GameConfig, MatchSetup, PlayerConfig
    from wc3env.vector import VectorSession, remote_workers

    statics: dict = {}

    def static(races):
        if races not in statics:
            statics[races] = Static.load(MAP, races)
        return statics[races]

    configs = []
    for g in range(games):
        races = PAIRS[(seed * games + g) % len(PAIRS)]
        configs.append(GameConfig(
            map=MAP, players=(PlayerConfig(0, env_race(races[0])), PlayerConfig(1, env_race(races[1]))),
            setup=MatchSetup(seed=seed * 1000 + g, randomize_starts=True), render=False, sound=False, step_ms=STEP_MS,
            observation="binary", features={0: hook_spec(static(races)), 1: hook_spec(static(races[::-1]))},
        ))  # fmt: skip
    parts: dict[str, list] = {}
    steps = 0
    with VectorSession(configs, group_size=2, launch=remote_workers(port)) as session:
        session.reset()
        for i in range(games):
            session.send(i, {0: [], 1: []})
        due: list = []  # (when, game): replies held for the pause, as the actor and the inference process would
        start = time.time()
        warm = start + 60  # the first minute's steps are not counted (games loading, the bridge's first copies)
        while time.time() < start + seconds:
            while due and due[0][0] <= time.time():
                session.send(heapq.heappop(due)[1], {0: [], 1: []})
            if len(due) == games:  # every game paused: none is stepping
                time.sleep(max(0.0, due[0][0] - time.time()))
                continue
            got = session.recv(timeout=max(0.0, due[0][0] - time.time()) if due else None)
            if got is None:
                continue
            i, _, _, info = got
            heapq.heappush(due, (time.time() + pause, i))
            if time.time() < warm:
                continue
            steps += 1
            hops = info.get("hops", {})
            for part, ms in hops.items():
                if not part.startswith("_"):
                    parts.setdefault(part, []).append(ms)
        for _ in range(games - len(due)):  # the steps still in flight
            session.recv()
    return {"parts": parts, "steps": steps, "seconds": seconds - 60}


@app.function(cpu=4, memory=16384, timeout=3600)
def host(name: str, bridges: int, games: int, seconds: float, pause: float) -> dict:
    """The games' other end, in a container like the actors': one listener per bridge, through tunnels."""
    import contextlib
    import threading

    import numpy as np

    with contextlib.ExitStack() as stack:
        tunnels = [stack.enter_context(modal.forward(PORT + k, unencrypted=True)) for k in range(bridges)]
        registry = modal.Dict.from_name(selfplay.REGISTRY, create_if_missing=True)
        registry[name] = {
            "games": ["%s:%d" % t.tcp_socket for t in tunnels],
            "container_at": selfplay._placement(os.environ),
        }
        results: list = [None] * bridges

        def run(k):
            results[k] = _session(PORT + k, games, seconds, pause, k)

        threads = [threading.Thread(target=run, args=(k,)) for k in range(bridges)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    parts: dict[str, list] = {}
    for r in results:
        for part, values in r["parts"].items():
            parts.setdefault(part, []).extend(values)
    return {
        "at": selfplay._placement(os.environ),
        "steps_per_s": sum(r["steps"] for r in results) / results[0]["seconds"],
        "ms": {
            part: (float(np.median(v)), float(np.percentile(v, 90)), float(np.mean(v))) for part, v in parts.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=300, help="stepping, the first 60 not counted")
    p.add_argument(
        "--pause",
        type=float,
        default=70,
        help="ms each reply is held before the next step: self-play's actor and inference time (sp15: ~70)",
    )
    p.add_argument("--games", type=int, default=32, help="per bridge (self-play: one bridge per actor, 32 games)")
    p.add_argument("--bridges", type=int, default=2, help="per sandbox (self-play: --per-box 2)")
    p.add_argument("--region", default="us-east", help="the host container's (self-play pins its GPU container)")
    a = p.parse_args()
    name = f"relay-{int(time.time())}"
    args = types.SimpleNamespace(per_box=a.bridges, games=a.games, group=2, cpu=4, memory=0,
                                 rest=["--hours", str((a.seconds + 900) / 3600)])  # fmt: skip
    registry = modal.Dict.from_name(selfplay.REGISTRY, create_if_missing=True)
    box = None
    with modal.enable_output(), app.run():
        call = host.with_options(region=a.region).spawn(name, a.bridges, a.games, a.seconds, a.pause / 1000)
        try:
            while "games" not in registry.get(name, {}):
                time.sleep(3)
            image, code = selfplay.games_image(), selfplay._local()[2]()
            box, box_at = selfplay.games_sandbox(args, image, code)
            selfplay.start_bridges(box, registry[name]["games"], args)
            print(f"host in {registry[name]['container_at']}; sandbox {box.object_id} in {box_at}", flush=True)
            result = call.get()
        finally:
            if box:
                box.terminate()
            registry.pop(name, None)
    print(f"host {result['at']}, sandbox {box_at}: {result['steps_per_s']:.0f} steps/s")
    print(f"{'part':10s} {'median':>8s} {'p90':>8s} {'mean':>8s}   ms")
    for part, (med, p90, mean) in sorted(result["ms"].items(), key=lambda kv: -kv[1][2]):
        print(f"{part:10s} {med:8.2f} {p90:8.2f} {mean:8.2f}")


if __name__ == "__main__":
    main()
