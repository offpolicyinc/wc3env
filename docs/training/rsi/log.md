# Experiment log

One row per experiment, newest last, never edited after the verdict (a correction is a new row). Each row is short:
the full story belongs in the run's logs and W&B, linked by run name.

- **Idea**: its id in [ideas.md](ideas.md).
- **Hypothesis**: the change and the number it should move, written before the run.
- **Setup**: start checkpoint, mode and stage, games, what differs from the baseline.
- **Result**: the numbers, experiment vs baseline: steps/s, win rate with its interval and games, and the
  diagnostic that explains it (update KL, clip fraction, explained variance, CPU/step, ...).
- **Verdict**: `keep` (merged, [8-decisions.md](../8-decisions.md) updated), `drop` (code removed, "tried and
  dropped" line added), `level` (within noise), or `rerun` (inconclusive, say why).

## Reference points (from before the loop)

| What | Numbers | Source |
|---|---|---|
| Best combat policy | cb9: ~0.58-0.67 of fights won at stage 0.85, update ~4,100 (training games) | [checkpoints/README.md](../../../checkpoints/README.md) |
| Best self-play throughput (Modal, 2 H100s + 8 sandboxes) | ~7,000-7,300 decided steps/s | [9-benchmarks.md](../9-benchmarks.md) |
| Learner offline | ~13,200 steps/s (0.62 s an update of 8,192 steps) | [9-benchmarks.md](../9-benchmarks.md) |
| warcraftsim (one 3090) | ~1,600-2,000 agent steps/s; 85-89% vs the insane AI on Echo Isles | [warcraftsim.md](../warcraftsim.md) |

## Experiments

| E | Date | Idea | Hypothesis | Setup | Result | Verdict | Run |
|---|---|---|---|---|---|---|---|
