"""Where one inference batch's time goes (policy/agent.py's Model.run) on an H100: wall time per batch, the GPU's
busy time and kernel count in it (torch.profiler), and the CPU's time per function (cProfile), at (seats, unit
rows) shapes self-play sees: a batch runs at its fullest seat's row size, and real games average ~55 units with
peaks past 128 (sp19), while the recorded observations have ~30-45 (their rows are padded out here).

    python -m modal run training/bench/profile_run.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-profile-run", image=image())
REPEATS = 30


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1200)
def bench(sizes: list[tuple[int, int]], compiled: bool = False) -> list[dict]:
    import cProfile
    import io
    import pickle
    import pstats
    import statistics
    import time

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile

    from training.policy.agent import UNIT_KEYS, Agents, Model, Seat
    from training.policy.features import Static
    from training.policy.model import load
    from training.policy.native import Features

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    static = Static.load(MAP, ("orc", "human"))
    rows = []
    for n, unit_rows in sizes:
        policy = load("/root/model.pt", "cuda")
        if compiled:  # the transformer blocks and SSM layers fused by torch.compile, inside the graph
            policy.compile_blocks()
        model = Model(policy, 1.0, 0)
        sent = []

        class Capture:  # runs the model and keeps the inputs of each step, as the inference process gets them
            def send(self, inputs):
                sent.append(inputs)
                self.out = model.run(inputs)

            def receive(self):
                return self.out

        agents = Agents(remote=Capture())
        seats = [Seat(static) for _ in range(n)]
        states = [Features(static) for _ in range(n)]
        for t in range(110):  # each seat from its game's start, so the last steps carry memory and a window
            obs = [traces[i % len(traces)][t] for i in range(n)]
            blocks = [state.step(o, seat.last) for state, o, seat in zip(states, obs, seats)]
            if t >= 100:
                agents.act(seats, obs, blocks, [b["scalars"] for b in blocks])
        inputs = {**sent[-1], "t": [10] * n, "trees": {}, "statics": {}}  # mid-chunk, nothing new to upload
        features = dict(inputs["features"])
        for k in UNIT_KEYS:  # out to `unit_rows` rows (padding rows: the cost is the row size's, not the count's)
            v = features[k]
            features[k] = np.concatenate([v, np.zeros((v.shape[0], unit_rows - v.shape[1], *v.shape[2:]), v.dtype)], 1)
        inputs["features"] = features
        for _ in range(5):
            model.run(inputs)
        torch.cuda.synchronize()

        walls = []
        for _ in range(REPEATS):
            began = time.perf_counter()
            model.run(inputs)  # ends with the choices copied to the CPU: the GPU's work is done when it returns
            walls.append(1000 * (time.perf_counter() - began))

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                model.run(inputs)
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernels = [e for e in events if "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]
        gpu_ms = sum(e.device_time for e in events) / 1000 / 5
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)

        stats_out = io.StringIO()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(10):
            model.run(inputs)
        profiler.disable()
        pstats.Stats(profiler, stream=stats_out).sort_stats("cumulative").print_stats(40)

        row = {
            "compiled": compiled,
            "seats": n,
            "unit_rows": int(inputs["features"]["units"].shape[1]),
            "wall_ms_median": round(statistics.median(walls), 1),
            "wall_ms_p90": round(sorted(walls)[int(0.9 * len(walls))], 1),
            "gpu_busy_ms": round(gpu_ms, 1),
            "kernels_per_batch": len(kernels) // 5,
            "copies_per_batch": (len(events) - len(kernels)) // 5,
        }
        row["model_stats"] = model.stats()
        rows.append(row)
        print(row, flush=True)
        if unit_rows == 96 and n == 58:
            print(table, flush=True)
            print(stats_out.getvalue(), flush=True)
    return rows


@app.local_entrypoint()
def main():
    sizes = [(58, 96), (300, 128)]
    bench.remote(sizes, False)
    bench.remote(sizes, True)
