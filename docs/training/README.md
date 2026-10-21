# Training

Plan for training an AlphaStar-style policy (small transformer, no LLM) on wc3env with a limited budget.
These docs record the design; the code is in `training/` at the repo root, while `src/wc3env` stays the
environment.

| Doc | Question it answers |
|---|---|
| [1-actions.md](1-actions.md) | What does the network output each step, and how does it become game orders? |
| [2-features.md](2-features.md) | What does the network see? |
| [3-model.md](3-model.md) | What is the network? |
| [4-reward.md](4-reward.md) | What is it rewarded for, and how does it get off the ground without human data? |
| [5-trainer.md](5-trainer.md) | Which algorithm, which opponents, in which order? |
| [6-infrastructure.md](6-infrastructure.md) | Where do games, inference and gradient steps run, and what does it cost? |
| [7-selfplay.md](7-selfplay.md) | How does RL from the cloned policy run: actors, learner, league? |
| [8-decisions.md](8-decisions.md) | Every training choice, one per line: the value, why, and what it rests on |
| [9-benchmarks.md](9-benchmarks.md) | Each part of self-play on Modal measured: the game, the hook, the tunnel, the actor, inference, the learner, cost; the run-by-run history |
| [10-actor-design.md](10-actor-design.md) | The decision path: what runs where (game and hook, actors, two inference processes on a GPU of their own, the learner) and why |
| [todo/](todo/README.md) | What is next, by area; finished lists in todo/done/ |
| [rsi/](rsi/loop.md) | The self-improvement loop: its goal and rules, the ranked ideas, one line per experiment |
| [12-hook-features.md](12-hook-features.md) | Features and masks: one implementation in C, in the hook and as a library; the block, the tests, how to change them |
| [combat.md](combat.md) | Fights instead of whole games: mirrored Orc armies on a flat map, for debugging RL in minutes |
| [selfplay-anatomy.html](selfplay-anatomy.html) | The whole pipeline on one page: machines, one decision, one PPO update, speeds, GPU, cost (open in a browser) |
| [warcraftsim.md](warcraftsim.md) | What did the closest prior work (same game build) learn, and what applies to us? |

## The two references

**AlphaStar** (DeepMind, SC2, 2019). 139M parameters, supervised on 971k human replays, then RL with
V-trace + UPGO in a 12-agent league; 44 days on 3,072 TPU cores and 50k CPU cores. Paper:
<https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf>.
Open reimplementations with readable code: DI-star (<https://github.com/opendilab/DI-star>),
mini-AlphaStar (<https://github.com/liuruoze/mini-AlphaStar>); the most honest small-scale report is
TStarBot-X (<https://arxiv.org/abs/2011.13729>), which found a faithful AlphaStar copy fails at
"only" 144 GPUs.

**Pluto** (tscmoo, Brood War, 2026). Won CoG 2026. Binary-only release
(<https://github.com/tscmoo/pluto>); its design was reconstructed from the DLL, the inference exe and
the weight file in <https://github.com/hwkim3330/pluto-re> (claims marked verified/inferred there). No
training code or writeup exists; what we know is its interface and model. 315M parameters, but the
interesting part is how small its *action interface* is: the DLL does the fiddly work (which unit
trains, which worker builds, addon placement) so the network picks from 20 action types × 233
arguments plus a unit or a map cell, 4 times per second.

## What we take from each

| Topic | AlphaStar | Pluto | Us |
|---|---|---|---|
| Decisions/s | ~2.7 (learned delay) | 4 (fixed, 6 frames) | 4 (fixed `step_ms=250`) |
| Action | type → delay → queued → up to 64 units (pointer) → unit or 256² point | type → argument → unit or coarse 8×8 + fine 16×16 cell | Pluto's shape; units picked per action by anchor + N nearest |
| Legality | masks on each head | masks computed by the DLL, also fed as inputs | masks computed in Python from obs + tech tree |
| Units in | ≤512 entities, transformer | ≤768 rows, 6-layer transformer | ≤256 rows, 4-layer transformer |
| Map in | 128² minimap CNN + scatter | 128² tile CNN + scatter | 128² grid as 64 patch tokens in the same transformer |
| Memory | 3×384 LSTM | 4096-wide attention GRU + slow memory | 3 SSM layers + 1 attention layer over the last minute |
| Size | 139M | 315M | ~10-20M |
| Bootstrap | human replays | unknown (self-play) | built-in AI: curriculum, then cloning it |
| Reward | win/loss (+ replay pseudo-rewards) | win/loss (inferred) | win/loss + annealed shaping from score fields |
| RL | V-trace + UPGO, league | KL-regularised PG, snapshot pool (inferred) | PPO (+UPGO), snapshot pool with PFSP |

## Order of work

1. **Action space + masks** (`training/policy/actions.py`) with a scripted policy that plays through it; the
   first test is that a hand-written build order wins against the easy AI using only these actions.
2. **Featurizer + model**, checked by overfitting a handful of recorded steps.
3. **Trainer vs built-in AI** on the laptop (8 games, CPU torch), Echo Isles, one race mirror.
4. **Cloud**: one GCP spot box with an L4 running everything; then scale out actors.
5. **Self-play** with a snapshot pool once it beats the normal AI.

Scope for the first run: one map (Echo Isles), one matchup (e.g. Orc vs Orc), 20-minute cap. DI-star
and TStarBot-X both restricted to one race and one map; widening later is cheap, debugging five
things at once is not.

## Environment gaps found while planning

Things the env does not give us yet that these docs rely on:

- Own research/upgrade levels are not in binary observations (needed for masks and features).
- No pathing/buildable grid or per-player visibility grid (needed for the map input).
- Player races are only in JSON observations (read once at reset; fine).
- No legality API: accepted orders can still be silently refused (no gold, missing tech), so the
  masks must be computed on our side from the tech tree in `src/wc3env/data/reference.json`
  (`wc3env.data`).
