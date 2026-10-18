"""Where a learner update's time goes outside its GPU work, under a run's traffic, on an H100: the learner's own loop
(Learner.run) fed by actor processes (~90 chunks/s, as sp17's 640 games made them, each step's unit rows grown to
mid-game counts) and two inference processes' weight pulls every 3 s. For every update: the time between updates,
and in it the wait for a batch, step() (its time_* parts), the weights' copy, the save and the log line; what is left
is time no part accounts for. A probe thread in the learner's process asks for the GIL every 1 ms: how late it
gets it says how long other threads hold it. Run with all the traffic, with parts of it, and with none.

    python -m modal run training/bench/learner_gap.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-learner-gap", image=image())
UPDATES, WARM = 14, 4
UNITS_MEAN, UNITS_MAX = 52, 92  # a step's unit rows in sp16-17's chunks


def grow(chunk: dict, rng) -> dict:
    """The chunk with each step's unit rows repeated (cycled) up to a mid-game count."""
    import numpy as np

    from training.replays.dataset import ID_FIELDS

    count = chunk["unit_count"].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(count)])
    target = np.clip(rng.normal(UNITS_MEAN, 15, len(count)), count, UNITS_MAX).astype(np.int64)
    target[rng.integers(len(count))] = UNITS_MAX
    rows = np.concatenate([np.resize(np.arange(offsets[t], offsets[t + 1]), target[t]) for t in range(len(count))])
    out = dict(chunk)
    out["unit_count"] = target.astype(chunk["unit_count"].dtype)
    for k in ("units", *ID_FIELDS):
        out[k] = chunk[k][rows]
    return out


def feeder(seed: int, port: int, per_second: float, seconds: float) -> None:
    """An actor's shipments of 4 chunks (compressed, as actors send them), `per_second` chunks a second, each made
    with the learner's current weights (so none is too old to train on)."""
    import pickle
    import time

    from training.selfplay.transport import Client

    chunks = pickle.load(open("/tmp/chunks.pkl", "rb"))
    client = Client(("127.0.0.1", port), b"k", wait=600)
    batches = Client(("127.0.0.1", port - 1), b"k", wait=600)  # the learner's batch process (batches.py)
    began, sent, k = time.time(), 0, seed
    while time.time() - began < seconds:
        version = client.call("league")[0]
        shipment = [{**chunks[(k + i) % len(chunks)], "version": version} for i in range(4)]
        batches.call("chunks", shipment)
        sent, k = sent + 4, k + 4
        ahead = began + sent / per_second - time.time()
        if ahead > 0:
            time.sleep(ahead)


def puller(port: int, every: float, seconds: float) -> None:
    """An inference process's weight pulls: every `every` s, asking with the version it has."""
    import time

    from training.selfplay.transport import Client

    client = Client(("127.0.0.1", port), b"k", wait=600)
    version, began = -1, time.time()
    while time.time() - began < seconds:
        new, state, _ = client.call("weights", version)
        if state is not None:
            version = new
        time.sleep(every)


@app.function(gpu="H100", cpu=48, memory=131072, timeout=3600)
def bench(setups: list[tuple[int, int, float]]) -> list[dict]:
    """setups: (feeding actors, weight pullers, chunks a second they send); with no feeders the buffer is filled
    ahead instead. Also the learner's own last log line's time_* means."""
    import gc
    import multiprocessing as mp
    import pickle
    import tempfile
    import threading
    import time

    import numpy as np

    from training.bench.common import play
    from training.policy.agent import Agents
    from training.policy.features import Static
    from training.policy.model import load
    from training.selfplay.learner import Learner, parser

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    rng = np.random.default_rng(0)
    chunks = play(Agents(load("/root/model.pt", "cuda"), 1.0, 0), list(range(8)), 385, traces, Static.load(MAP, ("orc", "human")))
    chunks = [grow(c, rng) for c in chunks]
    for k, c in enumerate(chunks):
        c["reward"][-1] = float((-1, 0, 1)[k % 3])
    pickle.dump(chunks, open("/tmp/chunks.pkl", "wb"))
    ctx = mp.get_context("spawn")
    rows = []
    collections_ = []  # (generation, seconds) of every garbage collection

    gc_began = [0.0]

    def gc_timer(phase, info):
        if phase == "start":
            gc_began[0] = time.perf_counter()
        else:
            collections_.append((info["generation"], time.perf_counter() - gc_began[0]))

    gc.callbacks.append(gc_timer)
    for n, (feeders, pullers, rate) in enumerate(setups):
        freeze = False  # the learner freezes the collector itself now
        port = 7600 + 2 * n  # its batch process listens on port - 1
        args = parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp(), "--listen", f"127.0.0.1:{port}",
                                    "--authkey", "k", "--stall-minutes", "0", "--log-every", "5", "--save-every", "10"])  # fmt: skip
        args.on_save = None
        learner = Learner(args)
        learner.update = args.warmup + 1  # past the warm-up: the policy loss runs too
        args.updates = learner.update + UPDATES
        assert feeders, "chunks come from feeders now (the batch process receives them)"

        # what each part of an update took, timed around the learner's own methods
        marks: list[tuple[str, float, float]] = []

        def timed(name, fn):
            def wrapper(*a, **k):
                began = time.perf_counter()
                out = fn(*a, **k)
                marks.append((name, began, time.perf_counter()))
                return out

            return wrapper

        parts: list[dict] = []
        step = learner.step

        def step_timed(chunks, host=None):
            began = time.perf_counter()
            stats = step(chunks, host)
            marks.append(("step", began, time.perf_counter()))
            parts.append(stats)
            return stats

        learner.step = step_timed
        learner._weights = timed("weights", learner._weights)
        learner._save = timed("save", learner._save)
        learner.say = timed("say", learner.say)

        # the probe: asks for the GIL every 1 ms; how late it runs is how long another thread kept it
        late, probing = [], [True]

        def probe():
            while probing[0]:
                began = time.perf_counter()
                time.sleep(0.001)
                late.append(time.perf_counter() - began - 0.001)

        threading.Thread(target=probe, daemon=True).start()
        procs = [ctx.Process(target=feeder, args=(f, port, rate / feeders, 600.0)) for f in range(feeders)]
        procs += [ctx.Process(target=puller, args=(port, 3.0, 600.0)) for _ in range(pullers)]
        for p in procs:
            p.start()
        late_from, gc_from = [0], [0]
        steps_at = []
        step_wrapped = learner.step

        def step_marked(chunks, host=None):
            if len(steps_at) == 1 and freeze:
                gc.freeze()  # everything alive after the first update (the compiled model's objects) out of the collector's sight
            if len(steps_at) == WARM:
                late_from[0] = len(late)
                gc_from[0] = len(collections_)
            steps_at.append(time.perf_counter())
            return step_wrapped(chunks, host)

        learner.step = step_marked
        learner.run()
        probing[0] = False
        for p in procs:
            p.kill()
        learner.server.close()
        # per update after the warm-up: from one step's start to the next
        ends = [m for m in marks if m[0] == "step"]
        spans = [(ends[i][1], ends[i + 1][1]) for i in range(WARM, len(ends) - 1)]
        out = {"interval": [], "step": [], "weights": [], "save": [], "say": []}
        for a, b in spans:
            out["interval"].append(b - a)
            for name in ("step", "weights", "save", "say"):
                out[name].append(sum(e - s for nm, s, e in marks if nm == name and a <= s < b))
        time_parts = {k: float(np.mean([p[k] for p in parts[WARM:-1] if k in p]))
                      for k in parts[-1] if k.startswith("time_") or k == "save_write"}  # fmt: skip
        mean = {k: float(np.mean(v)) for k, v in out.items()}
        measured = mean["step"] + mean["weights"] + mean["save"] + mean["say"] + time_parts.get("time_wait", 0.0)
        probe_late = np.array(late[late_from[0] :]) * 1000
        mine = collections_[gc_from[0] :]
        gc_seconds = {g: sum(t for gen, t in mine if gen == g) / max(len(spans), 1) for g in (0, 1, 2)}
        gc_count2 = sum(1 for gen, _ in mine if gen == 2)
        gc_max = max((t for _, t in mine), default=0.0)
        gc.unfreeze()
        logged = [line for line in open(f"{args.out}/log.txt") if " update " in line][-1]
        fields = logged.split(" | ")[-1].split()
        own = {k: float(v) for k, v in zip(fields[::2], fields[1::2]) if k.startswith("time_")}
        rows.append({"feeders": feeders, "pullers": pullers, "rate": rate, "learner_log": own,
                     "gc_s_per_update": {g: round(v, 3) for g, v in gc_seconds.items()}, "gc2_count": gc_count2,
                     "gc_max_s": round(gc_max, 3), "interval_s": round(mean["interval"], 3),
                     "step_s": round(mean["step"], 3), "weights_s": round(mean["weights"], 3), "save_s": round(mean["save"], 3),
                     "say_s": round(mean["say"], 3), **{k: round(v, 3) for k, v in time_parts.items()},
                     "unaccounted_s": round(mean["interval"] - measured, 3),
                     "step_unaccounted_s": round(mean["step"] - sum(v for k, v in time_parts.items() if k not in ("time_wait", "time_host", "save_write")), 3),
                     "gil_late_ms_p50": round(float(np.percentile(probe_late, 50)), 2),
                     "gil_late_ms_p99": round(float(np.percentile(probe_late, 99)), 2),
                     "gil_late_ms_max": round(float(probe_late.max()), 1)})  # fmt: skip
        print(rows[-1], flush=True)
        del learner
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    return rows


@app.local_entrypoint()
def main():
    bench.remote([(20, 2, 90.0), (20, 2, 200.0)])
