"""Does preparing batches alongside slow a learner update, and does a separate process avoid it? On an H100, the
same update (mid-game chunks, as training/bench/profile_learner.py grows them) timed alone, with Learner._host
running back to back on a thread of the learner's process (as Learner.run's preparing thread does when the
learner is never short of chunks), and with the same running back to back in another process.

    python -m modal run training/bench/learner_prep.py
"""

from __future__ import annotations

import modal

from training.bench.common import FLASH_ATTN_WHEEL, MAP, image

app = modal.App("wc3-bench-learner-prep", image=image(f"pip install {FLASH_ATTN_WHEEL}"))


def prepare_forever(path: str, stop) -> None:
    """In another process: Learner._host on the saved chunks, back to back (a CPU learner: nothing pinned)."""
    import pickle
    import tempfile

    from training.selfplay.learner import Learner, parser

    chunks = pickle.load(open(path, "rb"))
    learner = Learner(parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp(), "--no-compile"]))
    learner.device = __import__("torch").device("cpu")
    while not stop.is_set():
        learner._host(chunks)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1800)
def bench() -> str:
    import multiprocessing as mp
    import pickle
    import tempfile
    import threading
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
    rng = np.random.default_rng(0)
    learner = Learner(parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp()]))
    learner.update = learner.args.warmup + 1
    chunks = [grow(base[i % len(base)], rng) for i in range(learner.args.batch)]
    pickle.dump(chunks, open("/tmp/chunks.pkl", "wb"))
    host = learner._host(chunks)
    for _ in range(2):
        learner.step(chunks, host)

    def timed(label: str) -> str:
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(5):
            learner.step(chunks, host)
        torch.cuda.synchronize()
        line = f"{label:34s} {(time.perf_counter() - began) / 5:.3f} s an update"
        print(line, flush=True)
        return line

    lines = [timed("alone")]
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [learner._host(chunks) for _ in iter(lambda: stop.is_set(), True)], daemon=True
    )
    thread.start()
    time.sleep(1)
    lines.append(timed("with _host on a thread"))
    stop.set()
    thread.join()
    ctx = mp.get_context("spawn")
    stopping = ctx.Event()
    proc = ctx.Process(target=prepare_forever, args=("/tmp/chunks.pkl", stopping))
    proc.start()
    time.sleep(30)  # it loads its model
    lines.append(timed("with _host in another process"))
    stopping.set()
    proc.join(timeout=30)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(bench.remote())
