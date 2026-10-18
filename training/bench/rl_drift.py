"""What self-play changed in the policy, on the same pro games (teacher forced, held-out pro3 games, as clone_check):

- how much the core still changes over time: the mean cosine between the core at a step and 4 / 16 steps later
  (the auxiliary future heads are trained to predict exactly that; near 1 means the core barely moves);
- how spread the cores are: the mean distance of a step's core from the game's mean core, over the cores' mean norm;
- at the players' point orders, the probability the region head puts on the enemy's half (attacking needs it);
- each head's KL from the clone (bc12) on these states.

    python -m modal run training/bench/rl_drift.py
"""

from __future__ import annotations

import modal

from training.bench.clone_check import region_halves
from training.bench.common import FLASH_ATTN_WHEEL, image

app = modal.App("wc3-rl-drift", image=image(f"pip install {FLASH_ATTN_WHEEL}"))
VOLUME = modal.Volume.from_name("wc3-dataset")
CHECKPOINTS = ("bc12/last.pt", "sp39/pool/u000400.pt", "sp39/pool/u000800.pt", "sp40/pool/u000200.pt",
               "sp40/pool/u000400.pt", "sp40/pool/u000600.pt")  # fmt: skip
GAMES = 32
LAGS = (4, 16)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3600, volumes={"/dataset": VOLUME})
def drift() -> dict:
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch

    from training.policy.model import load
    from training.train.cloning import split
    from training.train.loader import ReplayData

    device = torch.device("cuda")
    _, held = split(Path("/dataset/pro3"))
    games = [ReplayData(f) for f in held[:GAMES]]
    halves = [region_halves(g.d) for g in games]
    keep = [i for i, h in enumerate(halves) if h is not None]
    games, halves = [games[i] for i in keep], torch.tensor(np.stack([halves[i] for i in keep]), device=device)
    reference = load("/dataset/runs/" + CHECKPOINTS[0], device)
    out = {}
    for name in CHECKPOINTS:
        model = load("/dataset/runs/" + name, device)
        sums = defaultdict(float)
        cores = [[] for _ in games]
        state = ref_state = None
        for start in range(0, max(g.steps for g in games), 64):
            chunks = [g.chunk(start, 64) for g in games]
            batch = {k: torch.from_numpy(np.stack([c[k] for c in chunks])).to(device) for k in chunks[0]}
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                o, r = model(batch, state), reference(batch, ref_state)
                h, hr = model.heads(o, batch), reference.heads(r, batch)
            state, ref_state = o["state"], r["state"]
            uses = model.uses(batch)
            for b in range(len(games)):
                cores[b].append(o["core"][b][batch["step_valid"][b].bool()].float().cpu())
            region = uses["region"]
            mass = (h["region"].float().softmax(-1) * halves[:, None, :].float()).sum(-1)
            sums["enemy_mass"] += float(mass[region].sum())
            sums["point_orders"] += float(region.sum())
            for head, logits in h.items():
                if head not in uses or head not in hr:
                    continue
                use = uses[head]
                lp, lq = logits.float().log_softmax(-1), hr[head].float().log_softmax(-1)
                kl = (lq.exp() * (lq - lp)).sum(-1).nan_to_num()
                if kl.shape != use.shape:  # a head whose logits are not [B, T, options]
                    continue
                sums[f"kl_{head}"] += float(kl[use].sum())
                sums[f"n_{head}"] += float(use.sum())
        sims = {k: [] for k in LAGS}
        spread = []
        for c in cores:
            c = torch.cat(c)
            if len(c) < 2 * max(LAGS):
                continue
            n = torch.nn.functional.normalize(c, dim=-1)
            for k in LAGS:
                sims[k].append(float((n[:-k] * n[k:]).sum(-1).mean()))
            spread.append(float((c - c.mean(0)).norm(dim=-1).mean() / c.norm(dim=-1).mean()))
        out[name] = {
            **{f"cos_{k}": float(np.mean(v)) for k, v in sims.items()},
            "spread": float(np.mean(spread)),
            "enemy_mass": sums["enemy_mass"] / max(sums["point_orders"], 1),
            **{k[3:]: sums[k] / max(sums["n_" + k[3:]], 1) for k in sums if k.startswith("kl_")},
        }
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    for name, row in drift.remote().items():
        print(name, {k: round(v, 4) for k, v in row.items()})
