"""The embedding tables' backward on an H100 (26% of a learner update's GPU time with mid-game chunks: PyTorch's
sort-based backward sums every row of one index on one thread, and most rows are padding, index 0). Ways to get the
same gradient, timed forward + backward on one minibatch's ids (32 chunks, mid-game unit counts as
training/bench/profile_learner.py grows them), with each gradient's largest difference from PyTorch's:

    current     F.embedding / F.embedding_bag, as policy/model.py does
    onehot32    the same forward; backward as one-hot (counts for a bag) transposed times the output's gradient, float32
    onehot16    the same in bfloat16 (as autocast computes the Linear layers' weight gradients)
    index_add   the same forward; backward with index_add_ (atomic adds), index 0 of a bag skipped

    python -m modal run training/bench/embedding_grad.py
"""

from __future__ import annotations

import modal

from training.bench.common import MAP, image

app = modal.App("wc3-bench-embedding-grad", image=image())


@app.function(gpu="H100", cpu=8, memory=65536, timeout=1800)
def bench() -> str:
    import pickle
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from training.bench.common import play
    from training.bench.profile_learner import grow
    from training.policy.agent import Agents
    from training.policy.features import MAX_UNITS, Static
    from training.policy.model import load
    from training.train.loader import ReplayData

    lines = []

    def say(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    model = load("/root/model.pt", "cuda")
    chunks = play(Agents(model, 1.0, 0), list(range(8)), 385, traces, Static.load(MAP, ("orc", "human")))
    rng = np.random.default_rng(0)
    chunks = [grow(chunks[i % len(chunks)], rng) for i in range(32)]  # one minibatch
    rows = 128  # the row size of a batch whose fullest step has 65-128 units
    fields = {}
    for name in (
        "unit_type",
        "unit_queue",
        "unit_owner",
        "unit_order",
        "unit_inventory",
        "unit_buffs",
        "unit_abilities",
    ):
        steps = []
        for c in chunks:
            data = ReplayData(c)
            for t in range(64):
                v = data.step(t)[name][:rows]
                steps.append(v)
        fields[name] = torch.from_numpy(np.stack(steps)).cuda().long().reshape(len(steps) * rows, -1).squeeze(-1)
    p = model
    tables = [("type", p.type_emb.weight, fields["unit_type"], False), ("queue", p.type_emb.weight, fields["unit_queue"], False),
              ("owner", p.owner_emb.weight, fields["unit_owner"], False),
              ("order", p.order_emb.weight, fields["unit_order"].clamp(max=15), False),
              ("items", p.item_emb.weight, fields["unit_inventory"], True), ("buffs", p.buff_emb.weight, fields["unit_buffs"], True),
              ("abilities", p.ability_emb.weight, fields["unit_abilities"], True)]  # fmt: skip
    say(
        f"{len(fields['unit_type'])} unit rows ({MAX_UNITS} max, {rows} kept); share of index 0: "
        + ", ".join(f"{n} {float((ids == 0).float().mean()):.2f}" for n, _, ids, _ in tables)
    )

    class OneHot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, bag, dtype):
            ctx.save_for_backward(ids)
            ctx.shape, ctx.bag, ctx.dtype = weight.shape, bag, dtype
            return F.embedding_bag(ids, weight, mode="sum", padding_idx=0) if bag else F.embedding(ids, weight)

        @staticmethod
        def backward(ctx, grad):
            (ids,) = ctx.saved_tensors
            v = ctx.shape[0]
            if ctx.bag:
                counts = torch.zeros(ids.shape[0], v, device=ids.device, dtype=ctx.dtype)
                counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=ctx.dtype))
                counts[:, 0] = 0  # padding_idx: no gradient
            else:
                counts = F.one_hot(ids, v).to(ctx.dtype)
            return (counts.T @ grad.to(ctx.dtype)).float(), None, None, None

    class IndexAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, bag):
            ctx.save_for_backward(ids)
            ctx.shape, ctx.bag = weight.shape, bag
            return F.embedding_bag(ids, weight, mode="sum", padding_idx=0) if bag else F.embedding(ids, weight)

        @staticmethod
        def backward(ctx, grad):
            (ids,) = ctx.saved_tensors
            out = torch.zeros(ctx.shape, device=grad.device, dtype=torch.float32)
            if ctx.bag:
                slots = ids.shape[1]
                flat, g = ids.reshape(-1), grad.float().repeat_interleave(slots, 0)
                keep = flat != 0
                out.index_add_(0, flat[keep], g[keep])
            else:
                out.index_add_(0, ids, grad.float())
            return out, None, None

    ways = {
        "current": lambda w, ids, bag: (
            F.embedding_bag(ids, w, mode="sum", padding_idx=0) if bag else F.embedding(ids, w)
        ),
        "onehot32": lambda w, ids, bag: OneHot.apply(w, ids, bag, torch.float32),
        "onehot16": lambda w, ids, bag: OneHot.apply(w, ids, bag, torch.bfloat16),
        "index_add": lambda w, ids, bag: IndexAdd.apply(w, ids, bag),
    }
    total = dict.fromkeys(ways, 0.0)
    for name, weight, ids, bag in tables:
        g_out = torch.randn(ids.shape[0], weight.shape[1], device="cuda")
        reference = None
        cells = []
        for way, fn in ways.items():
            w = weight.detach().clone().requires_grad_(True)
            for _ in range(3):  # warm-up
                w.grad = None
                fn(w, ids, bag).backward(g_out)
            torch.cuda.synchronize()
            began = time.perf_counter()
            for _ in range(10):
                w.grad = None
                out = fn(w, ids, bag)
                out.backward(g_out)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - began) / 10 * 1000
            total[way] += ms
            if reference is None:
                reference = w.grad.clone()
            diff = float((w.grad - reference).abs().max() / (reference.abs().max() + 1e-12))
            cells.append(f"{way} {ms:6.2f} ms (diff {diff:.1e})")
        say(f"{name:9s} " + " | ".join(cells))
    say("all tables, one minibatch's forward + backward: " + ", ".join(f"{k} {v:.1f} ms" for k, v in total.items()))
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    from pathlib import Path

    Path("build").mkdir(exist_ok=True)
    Path("build/embedding_grad.txt").write_text(bench.remote(), encoding="utf-8")
