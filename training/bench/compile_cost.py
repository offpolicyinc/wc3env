"""What an inference process's start costs with the step compiled: Model(compile=True) making its 12 shapes, twice,
each in a fresh process, torch.compile's cache in a local folder: cold, then warm (what a run with the cache pays).

    python -m modal run training/bench/compile_cost.py
"""

from __future__ import annotations

import modal

from training.bench.common import image

app = modal.App("wc3-bench-compile-cost", image=image())


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3000)
def warm() -> list[float]:
    """The inference process's start (Model(compile=True) making its 12 shapes) twice, each a fresh process, with
    torch.compile's cache in a local folder: the second is what a run whose cache is in place pays."""
    import os
    import subprocess
    import time

    took = []
    for _ in range(2):
        began = time.perf_counter()
        subprocess.run(["python", "-c", "from training.bench.compile_start import start; start()"], check=True, cwd="/root",
                       env={**os.environ, "PYTHONPATH": "/root", "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor"})  # fmt: skip
        took.append(round(time.perf_counter() - began, 1))
        print("start took", took[-1], "s", flush=True)
    return took


@app.local_entrypoint()
def main():
    print(warm.remote())
