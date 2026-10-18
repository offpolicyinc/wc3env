"""How far the learner's log-probabilities are from the inference step's at the same weights (no update), on an
H100: chunks of the recorded games decided by policy/agent.py's Model (bfloat16 trunk, float32 heads), first
uncompiled, then compiled as self-play runs it; the learner's log_probs on them as its loss computes them.
Percentiles of |learner - inference| and of |ratio - 1| (PPO clips at 0.2).

    python -m modal run training/bench/drift.py
"""

from __future__ import annotations

import modal

from training.bench.common import FLASH_ATTN_WHEEL, MAP, image

app = modal.App("wc3-bench-drift", image=image(f"pip install {FLASH_ATTN_WHEEL}"))


@app.function(gpu="H100", cpu=8, memory=65536, timeout=2400)
def drift() -> list[dict]:
    import pickle
    import tempfile

    import numpy as np
    import torch

    from training.bench.common import play
    from training.policy.agent import Agents
    from training.policy.features import Static
    from training.policy.model import float32, load
    from training.selfplay.learner import Learner, parser

    traces = pickle.load(open("/root/obs.pkl", "rb"))
    static = Static.load(MAP, ("orc", "human"))
    learner = Learner(parser().parse_args(["--init", "/root/model.pt", "--out", tempfile.mkdtemp()]))
    rows = []
    for compiled in (False, True):
        agents = Agents(load("/root/model.pt", "cuda"), 1.0, 0, compile=compiled)
        chunks = play(agents, [g % len(traces) for g in range(64)], 385, traces, static)
        gaps = []
        for k in range(0, len(chunks), 32):
            batch, state = learner._batch(chunks[k : k + 32])
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = learner.model(batch, state)
                log_prob, _ = learner.model.log_probs(batch, float32(out))
            gaps.append((log_prob - batch["old_log_prob"])[batch["step_valid"].bool()].abs().cpu().numpy())
        gap = np.concatenate(gaps)
        ratio = np.abs(np.exp(gap) - 1)
        rows.append({"inference_compiled": compiled, "steps": len(gap),
                     **{f"gap_p{q}": round(float(np.percentile(gap, q)), 5) for q in (50, 90, 99, 99.9)},
                     "gap_max": round(float(gap.max()), 4), "ratio_max": round(float(ratio.max()), 4),
                     "share_past_clip": float((ratio > 0.2).mean())})  # fmt: skip
        print(rows[-1], flush=True)
    return rows


@app.local_entrypoint()
def main():
    drift.remote()
