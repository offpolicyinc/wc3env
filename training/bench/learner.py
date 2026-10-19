"""A learner update's time on an H100, alone on its GPU: chunks made from the recorded games (bc10 deciding),
reused to fill 128-chunk batches, then updates timed with the blocks compiled (the default) and eager; the GPU
memory each used.

    python -m modal run training/bench/learner.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-learner", image=image())
UPDATES = 3


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1200)
def bench() -> list[dict]:
    import pickle
    import tempfile
    import time

    import torch

    from training.bench.common import play
    from training.policy.agent import Agents
    from training.policy.features import Static
    from training.policy.model import load
    from training.selfplay.learner import Learner, parser

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    chunks = play(
        Agents(load("/root/model.pt", "cuda"), 1.0, 0), list(range(8)), 385, traces, Static.load(MAP, ("orc", "human"))
    )
    for k, c in enumerate(chunks):  # some rewards, so the advantages are not all zero
        c["reward"][-1] = float((-1, 0, 1)[k % 3])
    rows = []
    for compiled in (True, False):
        args = parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp(),
                                    "--compile" if compiled else "--no-compile"])  # fmt: skip
        learner = Learner(args)
        learner.update = args.warmup + 1  # past the warm-up: the policy loss runs too
        batch = [chunks[i % len(chunks)] for i in range(args.batch)]
        learner.step(batch)  # warm-up (compiles)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        for _ in range(UPDATES):
            learner.step(batch)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - began) / UPDATES
        rows.append({"compiled": compiled, "update_s": round(seconds, 2), "steps_per_s": round(args.batch * 64 / seconds),
                     "gpu_used_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                     "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 1)})  # fmt: skip
        print(rows[-1], flush=True)
        del learner
        torch.cuda.empty_cache()
    return rows


@app.local_entrypoint()
def main():
    bench.remote()
