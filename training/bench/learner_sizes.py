"""Does a learner update stay fast when a batch needs a unit row size it has not had yet? Updates on an H100 with
batches whose fullest step needs 64, 96, 128 and 256 rows in turn, after the first update (which makes every size's
compiled programs: Learner._warm); each update's time.

    python -m modal run training/bench/learner_sizes.py
"""

from __future__ import annotations

import modal

from training.bench.common import FLASH_ATTN_WHEEL, MAP, image

app = modal.App("wc3-bench-learner-sizes", image=image(f"pip install {FLASH_ATTN_WHEEL}"))


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1800)
def bench() -> str:
    import pickle
    import tempfile
    import time

    import numpy as np
    import torch

    from training.bench.common import play
    from training.bench.profile_learner import grow
    from training.policy.agent import Agents
    from training.policy.features import Static
    from training.policy.model import load
    from training.selfplay.learner import Learner, parser

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    base = play(
        Agents(load("/root/model.pt", "cuda"), 1.0, 0), list(range(8)), 385, traces, Static.load(MAP, ("orc", "human"))
    )
    args = parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp()])
    learner = Learner(args)
    learner.update = args.warmup + 1
    rng = np.random.default_rng(0)
    lines = []
    for most in (92, 60, 92, 120, 250, 60):  # the first one also makes every size (_warm)
        chunks = [grow(base[i % len(base)], rng, mean=min(53, most), most=most) for i in range(args.batch)]
        torch.cuda.synchronize()
        began = time.perf_counter()
        learner.step(chunks)
        torch.cuda.synchronize()
        lines.append(f"fullest step {most:3d} units: update {time.perf_counter() - began:6.2f} s")
        print(lines[-1], flush=True)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    from pathlib import Path

    Path("build/learner_sizes.txt").write_text(bench.remote(), encoding="utf-8")
