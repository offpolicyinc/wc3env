# Ideas to test

The ideas themselves: one line each, what to change and why, ranked within each section by expected gain over cost.
No results here: what a run measured goes in [log.md](log.md), and an idea's status only points at its row (`E<n>`).
Changes to the network itself (size, depth, inputs, heads) need a retrain and are in [next-model.md](next-model.md).
Seeded 2026-10-18 from [todo/](../todo/README.md) and [warcraftsim.md](../warcraftsim.md); new ideas go where their
rank puts them, tested ones drop to the bottom of their section.

Status: `next` (top of the queue), `open`, `running`, `E<n>` (tested, see the log), `blocked: <why>`.

## 0. Baselines (before anything else)

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| B1 | Home throughput baseline: 4 actors x 8 games, combat mode, `--wine`; steps/s, CPU busy, actors' `ms:` split | steps/s | [home-setup.md](../home-setup.md#sizing-the-run-the-first-job) | E1 |
| B2 | Size the home run: more games until the CPU is ~90% busy or steps/s stops rising, then more actors with fewer games each | steps/s | warcraftsim (+13% from more actors) | E1 |
| B3 | Fixed evaluation: cb9 at stage 0.85, N fights on held-out seeds; its interval width sets N for every later verdict | eval | [combat.md](../combat.md) | E18 |
| B4 | A local run board (warcraftsim's dashboard as the model: a small server over `runs/`, one page): every run listed and compared, its lineage, live charts of the diagnostics that decide verdicts (wins by update, update KL, clip, KL to the clone, explained variance, steps/s and the loop's split, memory), its checkpoints and their fixed evaluations, a few replays a run; what the scratch scripts do today, rigorous and in one place for the user and the agents | observability, verdict speed | warcraftsim `warcraftsim/dashboard/`; ours: `training/cloud/selfplay_dashboard.py` (W&B, cloud runs only) | built (`python -m training.board`), E41 its first live run |

## 1. Throughput

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| T12 | Less GPU work a step: lower-precision inference, a cheaper inference step, a lighter learner update | steps/s | E8 | E14, E16 |
| T13 | The KL's anchor pass on a sample of each batch's chunks (`--kl-sample`), the rest weighed 0: an unbiased KL for less GPU (the pass is ~0.6 s of a ~3.2 s update) | steps/s | the update's time split | E30 |
| T14 | The learner's update as CUDA graphs, one per piece shape (warcraftsim: 3.35 -> 2.3 s an update) | steps/s | warcraftsim's optimization log | open |
| T15 | The learner's MPS share re-tuned once it waits on batches: less for it, more for inference | steps/s | E30 | E31 |
| T16 | The learner's step transformer on real tokens only at home too: PyTorch's own varlen attention where the flash-attn package has no wheel | steps/s | the update's profile (attention ~21% of its GPU) | E32 |
| T18 | Inference's graphs at finer sizes (seats by 8-16, 48 unit rows): less padding a batch | steps/s | combat's batches ~25 seats, <= 47 units | E35 |
| T19 | Shorter batching waits (actor and inference gather) once the GPU has room | steps/s | the actors' queued + inference's gather ~5 ms of a ~22 ms step | E36 |
| T20 | Two inference processes, half the actors each: no request waits behind the other half's batch | steps/s | the actors' round trip ~4 ms longer than inference's own time | E37 |
| T17 | Inference without the critic: the learner's own values for the advantages, computed on its GPU over each segment | steps/s | the critic is 55% of the parameters | open |
| T10 | Recycle a game process by its memory, not by its episode count, so long runs stay bounded | long runs, games | E2 | E12 |
| T11 | Fix the staging leak in the engine itself, so neither memory nor reloads grow over a run | RAM, reload time | E2 | E15 |
| T1 | Port warcraftsim's game-code cuts into the hook (map-version memo, A* cell cache, lazy heuristic, soft-float, no JPEG decode), with a replay parity check | CPU/step | warcraftsim `shim/cuts.c` (MIT), same exe RVAs | open |
| T4 | Games niced +10, so learner and inference get the CPU first | steps/s | warcraftsim | open |
| T5 | Paint every 50 steps instead of never, if CPU/step grows over a long game (fog pass) | CPU/step late-game | warcraftsim | open |
| T6 | 4 games per Python worker instead of 2 | memory, steps/s | [todo/done](../todo/done/2026-10-selfplay-infra.md) | E13, E33 |
| T7 | Evaluation games on the old 1080 Ti box over the LAN | verdict latency | warcraftsim | open |
| T8 | Less RAM a game, so more games fit | games, steps/s | E1 | E13 |
| T3 | Learner and inference sharing one GPU without time-slicing (pauses between minibatches, MPS) | inference latency, steps/s | warcraftsim | E3 |
| T9 | Shorter decisions: less of the actor's loop spent waiting on inference | steps/s | E1 | E4-E7 |
| T2 | GE-Proton Wine with fsync | CPU/step | warcraftsim; [9-benchmarks.md](../9-benchmarks.md) | E10 |

## 2. Learner (more learning per game played)

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| L1 | More gradient steps per update: 2 epochs and/or minibatch 16 | wins per game played | warcraftsim's recipe; Andrychowicz 2021 | E21, E26 |
| L2 | `--lr` 4e-5 (baseline), 1e-4, 2e-4 from the same checkpoint | wins per game played | [learner-tuning.md](../todo/learner-tuning.md) | E22 |
| L3 | KL-adaptive learning rate (targets 0.005, 0.01, 0.02), and early stop at 1.5x target | stability, wins | rl_games; SB3 `target_kl`; Schulman 2017 | E28, E39 |
| L4 | Extra critic-only passes (Phasic Policy Gradient) | explained variance, wins | Cobbe 2021 | open |
| L5 | Chunks replayed for a few updates with importance weights, if L1 helps | wins per game played | IMPACT (Luo 2020); V-trace (Espeholt 2018) | open |
| L6 | Anchor moved to a recent checkpoint, with a weaker pull, when a run plateaus (a rule, not by hand) | win-rate plateau | warcraftsim (+11 points, twice) | E23-E26 |
| L8 | The policy's GAE lambda below 1 (0.95, 0.8): today its advantages are whole returns to the fight's end | wins per game played | Schulman 2016 (GAE); Andrychowicz 2021 (0.9-0.95) | E28 |
| L9 | Smaller batches (`--batch` 64): twice the updates from the same games, on fresher data | wins per game played | Andrychowicz 2021 | E28 |
| L10 | Gradient clipping that binds only on outliers: at norm 1 it clips every step (norms ~50-100) | wins per game played | Andrychowicz 2021 | E28, E39 |
| L11 | Shorter horizons for the dense shaping channels (`--channel-gamma` dealt/taken < 1): less variance without leaning on the critic | wins per game played | 8-decisions per-channel discounts; Romoff 2019 | E29 |
| L12 | The shaping's weight (`--shaping-scale` 0.5, 2) and UPGO's (`--upgo` 0): which signal the policy follows | wins per game played | AlphaStar (UPGO); Ng 1999 | E29 |
| L13 | Exploration: the entropy bonus 10x (`--entropy` 1e-3); advantages not normalised (AlphaStar, DI-star) | wins per game played | AlphaStar; Andrychowicz 2021 | E29 |

## 3. Curriculum (after throughput and the learner: the next lever)

Tiers, scenarios and their progressions: [curriculum.md](../todo/curriculum.md). Each scenario is checked against
its scripted solution before it trains anything.

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| C2 | Scenario spec and runner: an `Episode` per scenario on combat staging, levels, success test, scripted solution, held-out seeds | all of tier 1-5 | SMACv2; PySC2 minigames | E40 |
| C3 | Tier 1 micro: focus fire, pull back, kite, surround, body block, hero snipe, spells one by one, spread, items, engage or not | win rate in fights, cast share | SMAC; Shao 2018 | E40, E42 (focus fire), E43 (hero snipe) |
| C4 | Scheduler: levels per scenario, weights by learning progress plus a floor, real games at ~50% | wins per game played | Matiisen 2017; PLR (Jiang 2021); ACCEL (Parker-Holder 2022) | fixed shares built (actor --scenarios), E42 |
| C5 | Snapshot benchmark: every snapshot scored on held-out scenarios | eval | | open |
| C6 | Terrain arena (chokes, tree lines) for body blocks, surrounds, kiting near trees | C3's terrain levels | | open |
| C7 | Replay staging: a pro game's state at minute N in one message | tiers 2-5 | Salimans & Chen 2018 | open |
| C8 | Tier 2: fights cut from pro replays at their first damage | win rate in real fights | | blocked: C7 |
| C9 | Tier 3 economy: saturation, no supply block, no idle money, upkeep, creeping | income, block and float rates | | blocked: C7 |
| C10 | Tier 4 builds: opening, tech timing, unit mix, placement | build-order distance, timings | AlphaStar's z | blocked: C7 |
| C11 | Tier 5 game plans: defend harass and rushes, attack at a timing, expand | win rate of the piece | | blocked: C7 |
| C1 | Scripted check of each skill before it becomes a scenario, first pulling back hurt units | does the skill pay? | [curriculum.md](../todo/curriculum.md) | E9 |
