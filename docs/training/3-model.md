# 3. Model

The network: features ([2-features.md](2-features.md)) in, action head scores
([1-actions.md](1-actions.md)) and a value out.

## Two axes

The model mixes information in two directions:

- **across units, within one step**: which of my units are near which enemies, is my hero in danger.
  Everything is present at once (a set of ~300 tokens), so plain attention fits;
- **across time, step to step**: what the enemy built two minutes ago, where my scout went. A game is
  5,000-7,000 steps, so this is where memory cost matters.

LLM hybrids alternate SSM and attention layers over one axis (time). We have a per-step encoder
first, then that same kind of hybrid over time, where each "token" is one step. Vision-language
models (image encoder, then the language model), robot policies and Pluto (unit transformer, then
GRU plus attention over a slow memory) have the same layout.

## Layout

```
each step:
  unit rows (≤256) ──┐
  map patches (64) ──┼─► step transformer (4 layers, 256 wide) ─► per-unit vectors, per-patch vectors
  scalar token (1) ──┘                                         └─► step vector (the scalar token's output)

over time (one step vector per step):
  SSM ─► SSM ─► SSM ─► attention over the last 256 steps (~1 min) ─► core output

heads (sampled in order, each conditioned on the previous choices):
  type      MLP on core output
  argument  MLP on core output + type
  anchor    pointer: query from core output + choices so far, keys = per-unit vectors (own units)
  same_type, N   MLPs
  target unit    pointer over per-unit vectors
  region    pointer over the 64 per-patch vectors
  cell      256 scores from the chosen patch's vector + choices so far
value: MLP on the core output (training also sees the opponent's step vector, see below)
```

## The plan input

Like AlphaStar's z, the policy is given a plan for the whole game (training/policy/plan.py), fixed from the first
step to the last, read from a pro game: its first 20 builds, trainings and research in order; its heroes, first to
third; and everything it made over the game. It is encoded as AlphaStar's is (detailed-architecture.txt, the
scalar encoder), and enters only through the scalar token (no gating of the heads, unlike AlphaStar's):

```
plan.order  20 items ─► embedding + place ─► its own attention layer ─► order vector ──────┐
            (+ a start token, read out)                                                     │
plan.heroes  3 items ─► embedding + place, summed ─► heroes vector ─────────────────────────┤ summed ─► MLP ─► added to
plan.mix   every type ─┬─ unit/building/hero types: made or not (0/1) ─► Linear 32, ReLU ─┐ │           the scalar token
                       └─ research: levels / 3 ─────────────────────► Linear 32, ReLU ─┴─► Linear ─┘
```

- A checkpoint from before loads with its old summed plan input dropped: its plan starts as the empty plan.
- Cloning gives each game its own plan in two parts, each kept or left out on its own: the order in 80% of games,
  the rest (heroes, types, research) in 50% (AlphaStar's `BO_PROBABILITY`, `BU_PROBABILITY`). So the policy plays
  with either part, both, or none.
- When playing, the plan is copied from a won pro game of the same matchup (90% of self-play players), or left
  empty. The rewards for following it are build, units and upgrades (rewards.md).
- Pluto's 32 named openings are a coarser version of the same thing: grouping pro plans by first hero and early
  builds gives such openings later, for a bandit to choose between per opponent, without retraining.

## Decided

| Decision | AlphaStar | Pluto | Us |
|---|---|---|---|
| Unit encoder | transformer 3 × 256 | transformer 6 × 768, RMSNorm, QK-norm, SwiGLU | transformer 4 × 256, RMSNorm, QK-norm, SwiGLU |
| Map encoder | CNN + scatter | CNN + scatter | 16×16 patches as tokens in the same transformer |
| Memory | LSTM 3 × 384 | GRU 4096 + attention over a slow memory | 3 SSM layers + 1 attention layer over 256 steps |
| Region choice | 256×256 deconv | pointer over 64 region vectors | pointer over the 64 patch vectors |
| Value sees opponent | yes (training only) | unknown | yes (training only) |
| Size | 139M | 315M | ~10-20M to start |

Why each:

- **Map patches instead of a CNN.** The 64 patches are exactly the 64 regions of the point head, so
  "pick a region" is a pointer like "pick a unit", and units attend to terrain directly. One kind of
  module instead of transformer + CNN + scatter. A CNN learns maps from less data; we train on one
  fixed map first, so that matters little. If exact placement inside a patch turns out weak, the cell
  head gets a small conv over that patch's pixels.
- **SSM + one attention layer for memory.** SSM layers keep a small fixed-size state per game, cost
  the same per step however long the game, and train in parallel over a chunk. A transformer over
  time could do it alone, but its cache grows with history and each training chunk would need the
  preceding window recomputed. One windowed attention layer adds exact recall of the last minute at
  ~0.5 GB of cache for 1,000 games. Same trend as LLM hybrids and Pluto's GRU + slow memory.
- **Value sees the opponent.** The value head only helps training (it judges how good the position
  is); it's never used to choose actions, so giving it the opponent's view is legal and makes its
  estimates much better. In self-play both views exist anyway. Against the built-in AI we observe
  its slot too.
- **Small.** Most of what decides success in RL is the learning signal (reward, exploration,
  opponents, masks) and the amount of experience, not the architecture; AlphaStar's ablations and
  OpenAI Five's plain LSTM point the same way. A small model trains faster per step; grow it once
  learning works.

## Implemented

`training/policy/model.py` (`Policy`), `training/replays/dataset.py` (replays to files: rows, trees, scalars, labels,
masks; ~2.7 MB per player-game), `training/train/loader.py` (files to [B, T] chunks; the map is rebuilt from the
rows, with `features.build_map`, as the model draws it on the GPU). One head beyond the layout above: **aim** (none, point,
unit), chosen after the argument, since build, rally, use-item and drop can take either a point or a unit.
`training/tests/test_dataset.py` writes a replay's data, checks the rebuilt map and the labels' masks, and
trains a small model on it.

## Implementation notes

- **SSM kernel.** `mamba-ssm` needs CUDA on Linux. Write a pure-PyTorch diagonal/gated SSM (~50
  lines) for laptop development and switch to the fast kernel on the GPU box; same weights.
- **Recurrent state in PPO.** Each training chunk (64 steps) stores the SSM states and the attention
  cache window from when it was played, so the learner replays it with the same context.
- **Swappable parts.** Memory (GRU / SSM / attention) and map encoder (patches / CNN) behind one
  interface, so a known-good baseline (GRU + CNN) can be compared on the same budget.
