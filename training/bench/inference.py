"""The decision path's speed on one H100 container, without games: a learner (serving its weights), the inference
process (or several, actors split among them), and K CPU-only actor processes, each deciding for 32 seats over the
recorded observations in GROUPS groups, each its own request (sp21's actors sent ~6-7 games a request, ~1.3 more in
flight). A group's next step waits GAME_MS after its decisions come back (the game's step and the tunnel, ~27-40 ms in
sp21), so the inference process sees a real run's flow of requests, not a flood. Decisions/s, each request's time
from sent to answered.

    python -m modal run training/bench/inference.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-inference", image=image())
SEATS, GROUPS, STEPS, WARM, GAME_MS = 32, 4, 160, 16, 35.0


def actor(seed: int, port: int, results) -> None:
    import collections
    import os
    import pickle
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from training.policy.agent import Agents, Seat
    from training.policy.features import Static
    from training.policy.native import Features
    from training.selfplay.transport import Client

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    static = Static.load(MAP, ("orc", "human"))
    client = Client(("127.0.0.1", port), b"k", wait=600, compress=False)

    class Remote:
        def send(self, inputs):
            client.send("act", ("latest", inputs))

        def receive(self):
            reply = client.receive()
            reply.pop("version")
            return reply

    agents = Agents(remote=Remote())
    seats = [Seat(static) for _ in range(SEATS)]
    states = [Features(static) for _ in range(SEATS)]
    games = [(seed * SEATS + j) % len(traces) for j in range(SEATS)]
    halves = [range(g * SEATS // GROUPS, (g + 1) * SEATS // GROUPS) for g in range(GROUPS)]
    step, ready, sent, begun = [0] * GROUPS, [0.0] * GROUPS, [0.0] * GROUPS, collections.deque()
    latencies, decided, started = [], 0, None
    while any(s < STEPS for s in step) or begun:
        now = time.perf_counter()
        for h, members in enumerate(halves):
            if step[h] < STEPS and h not in [b[0] for b in begun] and now >= ready[h]:
                obs = [traces[games[j]][step[h]] for j in members]
                blocks = [states[j].step(o, seats[j].last) for j, o in zip(members, obs)]
                sent[h] = time.perf_counter()
                begun.append((h, agents.begin([seats[j] for j in members], obs, blocks, [b["scalars"] for b in blocks])))
        if not begun:
            time.sleep(max(0.0, min(r for r, s in zip(ready, step) if s < STEPS) - time.perf_counter()))
            continue
        h, handle = begun.popleft()
        agents.end(handle)
        done = time.perf_counter()
        if step[h] >= WARM:
            started = started or done
            latencies.append(1000 * (done - sent[h]))
            decided += len(halves[h])
        step[h] += 1
        ready[h] = done + GAME_MS / 1000
    results.put((decided / (time.perf_counter() - started), latencies))


@app.function(gpu="H100", cpu=36, memory=131072, timeout=1200)
def bench(setups: list[tuple[int, int, float]]) -> list[dict]:
    """setups: (actors, inference processes, the inference process's --gather in seconds)."""
    import multiprocessing as mp
    import os
    import subprocess
    import tempfile

    from training.selfplay.learner import Learner, parser
    from training.selfplay.transport import Server

    learner = Learner(parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp()]))
    Server(("127.0.0.1", 7600), b"k", learner.handle)
    rows = []
    for n, (k, inference, gather) in enumerate(setups):
        ports = [7700 + 10 * n + i for i in range(inference)]
        servers = [subprocess.Popen(["python", "-m", "training.selfplay.inference", "--learner", "127.0.0.1:7600",
                                     "--listen", f"127.0.0.1:{port}", "--authkey", "k", "--gather", str(gather)],
                                    cwd="/root", env={**os.environ, "PYTHONPATH": "/root"}) for port in ports]  # fmt: skip
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        procs = [ctx.Process(target=actor, args=(s, ports[s % inference], results)) for s in range(k)]
        for p in procs:
            p.start()
        got = [results.get(timeout=900) for _ in procs]
        for p in procs:
            p.join()
        for server in servers:
            server.kill()
        latencies = sorted(x for g in got for x in g[1])
        rows.append({"actors": k, "inference": inference, "gather_ms": 1000 * gather, "seats": SEATS * k,
                     "decisions_per_s": round(sum(g[0] for g in got)),
                     "reply_ms_p50": round(latencies[len(latencies) // 2], 1),
                     "reply_ms_p90": round(latencies[int(0.9 * len(latencies))], 1)})  # fmt: skip
        print(rows[-1], flush=True)
    return rows


@app.local_entrypoint()
def main():
    setups = [(16, 2, 0.002), (16, 1, 0.002)]
    bench.remote(setups)
