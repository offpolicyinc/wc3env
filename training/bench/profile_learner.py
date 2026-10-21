"""Where one learner update's time goes on an H100: wall time, the GPU's busy time and kernel count in it, the
waits for the GPU (syncs), the busiest kernels (torch.profiler), and the CPU's time per function (cProfile).
Chunks from the recorded games, as training/bench/learner.py makes them, with each step's unit rows repeated up to
mid-game counts (the recordings are the first 100 s: ~10-40 units; sp16-17's chunks ~50 a step, ~90 at most, so the
128-row size); the blocks compiled, as in a run.

    python -m modal run training/bench/profile_learner.py
"""

from __future__ import annotations

import modal

from training.bench.common import FLASH_ATTN_WHEEL, MAP, image

app = modal.App("wc3-bench-profile-learner", image=image(f"pip install {FLASH_ATTN_WHEEL}"))
UNITS_MEAN, UNITS_MAX = 52, 92  # a step's unit rows in sp16-17's chunks


def grow(chunk: dict, rng, mean: int = UNITS_MEAN, most: int = UNITS_MAX) -> dict:
    """The chunk with each step's unit rows repeated (cycled) up to a mid-game count."""
    import numpy as np

    from training.replays.dataset import ID_FIELDS

    count = chunk["unit_count"].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(count)])
    target = np.clip(rng.normal(mean, 15, len(count)), count, most).astype(np.int64)
    target[rng.integers(len(count))] = most  # the batch's fullest step decides its row size
    rows = np.concatenate([np.resize(np.arange(offsets[t], offsets[t + 1]), target[t]) for t in range(len(count))])
    out = dict(chunk)
    out["unit_count"] = target.astype(chunk["unit_count"].dtype)
    for k in ("units", *ID_FIELDS):
        out[k] = chunk[k][rows]
    return out


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1800)
def profile() -> str:
    import cProfile
    import io
    import pickle
    import pstats
    import tempfile
    import time

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    from training.bench.common import play
    from training.policy.agent import Agents
    from training.policy.features import Static
    from training.policy.model import load
    from training.selfplay.learner import Learner, parser

    out = io.StringIO()

    def say(line: str = "") -> None:
        print(line, flush=True)
        out.write(line + "\n")

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    chunks = play(
        Agents(load("/root/model.pt", "cuda"), 1.0, 0), list(range(8)), 385, traces, Static.load(MAP, ("orc", "human"))
    )
    for k, c in enumerate(chunks):  # some rewards, so the advantages are not all zero
        c["reward"][-1] = float((-1, 0, 1)[k % 3])
    rng = np.random.default_rng(0)
    chunks = [grow(c, rng) for c in chunks]
    args = parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp()])
    learner = Learner(args)
    learner.update = args.warmup + 1  # past the warm-up: the policy loss runs too
    batch = [chunks[i % len(chunks)] for i in range(args.batch)]
    host = learner._host(batch)  # in a run the next batch's CPU part is made on a thread meanwhile
    began = time.perf_counter()
    for _ in range(3):
        host = learner._host(batch)
    say(f"preparing a batch (_host, pinned): {(time.perf_counter() - began) / 3:.3f} s")
    for _ in range(2):  # compiles
        learner.step(batch, host)
    torch.cuda.synchronize()

    say(f"packed step transformer (a variable-length kernel found): {learner.varlen}")
    if learner.varlen:  # the packed step transformer against the padded one, on one piece, same weights
        from training.policy.model import float32
        from training.selfplay.learner import _row_size, _rows, pack

        b, st = _rows(*learner._to_device(host), torch.arange(32, device="cuda"))
        size = _row_size(int(b["unit_count"].max()))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            padded = float32(learner.model(b, st, rows=size))
            packed = float32(
                learner.model(
                    b, st, rows=size, pack=pack(b["unit_count"].reshape(-1).cpu().numpy(), size, learner.device)
                )
            )
            lp_padded = learner.model.log_probs(b, padded)[0]
            lp_packed = learner.model.log_probs(b, packed)[0]
        with torch.no_grad():  # float32 throughout (no autocast): the answer both bfloat16 ways round
            exact = learner.model(b, st, rows=size)
            lp_exact = learner.model.log_probs(b, exact)[0]
        live = b["step_valid"].bool()

        def gap(x, y):
            d = (x - y)[live].abs()
            return f"max {float(d.max()):.1e} p99 {float(d.quantile(0.99)):.1e}"

        say(
            f"core scale {float(exact['core'].abs().max()):.1f}; core diff from float32: padded {float((padded['core'] - exact['core']).abs().max()):.1e}, "
            f"packed {float((packed['core'] - exact['core']).abs().max()):.1e}; packed vs padded {float((padded['core'] - packed['core']).abs().max()):.1e}"
        )
        say(
            f"log-prob from float32: padded {gap(lp_padded, lp_exact)}; packed {gap(lp_packed, lp_exact)}; packed vs padded {gap(lp_packed, lp_padded)}"
        )

    began = time.perf_counter()
    for _ in range(3):
        stats = learner.step(batch, host)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - began) / 3
    say(
        f"update {wall:.3f} s ({args.batch * 64 / wall:.0f} steps/s); laps "
        f"{ {k: round(v, 3) for k, v in stats.items() if k.startswith('time_')} }; units per step "
        f"{stats['units_mean']:.0f} mean, {stats['units_max']:.0f} at most"
    )

    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        learner.step(batch, host)
        torch.cuda.synchronize()
    events = prof.key_averages()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_s = sum(e.self_device_time_total for e in kernels) / 1e6
    launches = sum(e.count for e in kernels)
    say(
        f"GPU busy {gpu_s:.3f} s of {wall:.3f} ({gpu_s / wall:.0%}); {launches} kernels, "
        f"{gpu_s / max(launches, 1) * 1e6:.1f} us each on average"
    )
    for name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "aten::item",
                 "aten::_local_scalar_dense", "cudaLaunchKernel"):  # fmt: skip
        e = next((e for e in events if e.key == name), None)
        if e is not None:
            say(f"  {name}: {e.count} calls, {e.cpu_time_total / 1e3:.0f} ms of CPU")
    say("\nbusiest kernels (GPU ms, calls):")
    for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:25]:
        say(f"  {e.self_device_time_total / 1e3:8.1f}  {e.count:6d}  {e.key[:110]}")
    say("\nCPU ops by self time (ms, calls):")
    cpu_ops = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU]
    for e in sorted(cpu_ops, key=lambda e: -e.self_cpu_time_total)[:25]:
        say(f"  {e.self_cpu_time_total / 1e3:8.1f}  {e.count:6d}  {e.key[:110]}")

    import collections
    import warnings

    torch.cuda.set_sync_debug_mode("warn")  # a warning, with the Python line, at every wait for the GPU
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.step(batch, host)
        torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(0)
    say(f"\nwhere the CPU waits for the GPU ({len(caught)} waits; count, line):")
    lines = collections.Counter(f"{w.filename.split('/root/')[-1]}:{w.lineno}" for w in caught)
    for where, count in lines.most_common(30):
        say(f"  {count:5d}  {where}")

    profiler = cProfile.Profile()
    profiler.enable()
    learner.step(batch, host)
    torch.cuda.synchronize()
    profiler.disable()
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(40)
    say("\ncProfile, cumulative:")
    say(text.getvalue())
    return out.getvalue()


@app.local_entrypoint()
def main():
    from pathlib import Path

    text = profile.remote()
    Path("build").mkdir(exist_ok=True)
    Path("build/profile_learner.txt").write_text(text, encoding="utf-8")
