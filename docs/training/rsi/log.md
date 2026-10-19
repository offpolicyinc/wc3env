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
| E15 | 2026-10-19 | T11 | Something in the engine keeps each staged fight's effects and units (E2) | one combat game (leak probe, WineHQ stable); hardware watchpoints on reference counts, heap scans for holders; 300-fight A/B against the same hook without the fix, computers on both slots | **Two owners. (1) Local timers never fire in step mode:** the frame's local event (0x40090094) ticks "Timer Dispatch (l)" by the frame's real dt, which stepping makes 0, and every spell's eye candy (CEffectImage + sprite, model, emitters; ~50 a fight from Spirit Link, Disenchant, Bloodlust, Healing Wave, Chain Lightning, plus ~130 CEffectImagePos) waits on one to end: 0 of ~5,000 freed. Now ticked by the game ms each frame ran. **(2) CWorldFrameWar3's unit list** (+0x6a0) holds 2 refs on every unit made until the map ends: rebuilt (its own RVA 0x20dc30) after a step that removed units. **250 -> 830 MB (1.98 MB/fight) becomes 248 -> 353 MB (0.36)**, -82%; without computer slots 1.53 -> 0.32 MB/fight (ABAB 60 fights: the rebuild ~0.03 of it). Same speed; every unit row after each of 30 fights hashes the same. GE-Proton 10-34 the same (30 fights: 2.10 -> 0.54 MB/fight vs stable's 2.09 -> 0.52): the owner is the engine, not Wine. Left: particles, the computers' HeroEval (~0.2 MB/fight), items; map reload unchanged (~6.1 MB each) | keep (wc3hook/step.c, stage.c, wc3hook.h); T10's recycle still bounds what is left | leak probe, scratch |
