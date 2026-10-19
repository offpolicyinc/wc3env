"""The inference step's GPU time per part on an H100 (policy/agent.py's Model._step), each part captured as a CUDA
graph of its own and replayed with real inputs: the map, encoding a step (units, patches, the step transformer),
the memory (SSM layers, time attention), the conversion to float32, the heads (decide.sample) and the critic.

    python -m modal run training/bench/step_parts.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-step-parts", image=image())


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1200)
def bench(seats: int = 58, unit_rows: int = 96, compiled: str = "") -> dict:
    """compiled: "" none, "blocks" (Policy.compile_blocks), "more" (also encode_step and decide.sample whole)."""
    import pickle

    import numpy as np
    import torch

    import training.policy.decide as decide
    from training.policy.agent import TABLES, UNIT_KEYS, Agents, Model, Seat
    from training.policy.features import Static
    from training.policy.model import draw_map, float32, load
    from training.policy.native import Features

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    static = Static.load(MAP, ("orc", "human"))
    policy = load("/root/model.pt", "cuda")
    if compiled:
        policy.compile_blocks()
    if compiled == "more":
        import torch._dynamo

        import training.policy.agent as agent_module
        import training.policy.decide as decide_module

        torch._dynamo.config.cache_size_limit = 64
        policy.encode_step = torch.compile(policy.encode_step, dynamic=False)
        decide_module.sample = agent_module.sample = torch.compile(decide_module.sample, dynamic=False)
    model = Model(policy, 1.0, 0)
    sent = []

    class Capture:
        def send(self, x):
            sent.append(x)
            self.out = model.run(x)

        def receive(self):
            return self.out

    agents = Agents(remote=Capture())
    seat_list = [Seat(static) for _ in range(seats)]
    states = [Features(static) for _ in range(seats)]
    for t in range(102):
        obs = [traces[i % len(traces)][t] for i in range(seats)]
        blocks = [s.step(o, seat.last) for s, o, seat in zip(states, obs, seat_list)]
        if t >= 100:
            agents.act(seat_list, obs, blocks, [b["scalars"] for b in blocks])
    x = {**sent[-1], "t": [10] * seats, "trees": {}, "statics": {}}
    f = dict(x["features"])
    for k in UNIT_KEYS:
        v = f[k]
        f[k] = np.concatenate([v, np.zeros((v.shape[0], unit_rows - v.shape[1], *v.shape[2:]), v.dtype)], 1)
    x["features"] = f
    model.run(x)  # the _Step at this shape, its buffers filled
    step = next(iter(model.graphs.values()))
    b, dev = step.dev, model.device
    rows = b["rows"]
    batch = {k: b[k][:, None] for k in step.features}
    batch["static_map"] = model.static_table[b["maps"]]
    batch["trees"] = model.trees[rows][:, None]
    for k in step.plans:
        batch[f"plan_{k}"] = b[f"plan_{k}"]
    length = model.length[rows].clamp(max=62)
    window_mask = torch.arange(64, device=dev)[None] < length[:, None]
    state = [h[rows] for h in model.state]
    flat = {
        k: v[:, 0] for k, v in batch.items() if v.dim() >= 2 and k not in ("static_map",) and not k.startswith("plan_")
    }
    flat["static_map"] = batch["static_map"]
    auto = lambda: torch.autocast("cuda", torch.bfloat16)  # noqa: E731

    def full():
        with auto():
            return policy(batch, state, model.window[rows], window_mask, rows=unit_rows)

    with torch.no_grad():
        out = float32(full())
        tables = decide.unpack({k: b[f"table_{k}"] for k in TABLES})
        count = batch["unit_count"][:, 0]
        x_step = torch.zeros(rows.shape[0], 1, policy.d, device=dev)  # (the batch padded to its seat size)
        parts = {
            "map": lambda: draw_map(flat, flat["static_map"]),
            "encode_step": lambda: _encode(policy, flat, unit_rows, auto),
            "memory_and_time": lambda: _memory(policy, x_step, state, model.window[rows], window_mask, auto),
            "forward": full,
            "float32": lambda: float32(out),
            "heads": lambda: decide.sample(policy, out, tables, count, 1.0, None),
            "critic": lambda: policy.value(out, b["opponents"].float()[:, None]),
            "whole_step": lambda: model._step(b, unit_rows),
        }
        result = {"seats": seats, "unit_rows": unit_rows, "compiled": compiled}
        for name, fn in parts.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            result[name] = round(start.elapsed_time(end) / 50, 3)
    print(result, flush=True)
    return result


def _memory(policy, x, state, window, window_mask, auto):
    import torch

    with auto():
        for k, layer in enumerate(policy.memory):
            x, _ = layer(x, state[k])
        seq = torch.cat([window, x], 1)
        keys = torch.cat([window_mask, torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)], 1)
        return policy.time_attention(seq, key_mask=keys)


@app.local_entrypoint()
def main():
    bench.remote(compiled="more")


def _encode(policy, flat, unit_rows, auto):
    with auto():
        return policy.encode_step(flat, unit_rows)
