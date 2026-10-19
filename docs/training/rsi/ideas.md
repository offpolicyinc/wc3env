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
| B3 | Fixed evaluation: cb9 at stage 0.85, N fights on held-out seeds; its interval width sets N for every later verdict | eval | [combat.md](../combat.md) | open |

## 1. Throughput

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| T12 | Less GPU work a step: lower-precision inference, a cheaper inference step, a lighter learner update | steps/s | E8 | running |
| T10 | Recycle a game process by its memory, not by its episode count, so long runs stay bounded | long runs, games | E2 | running |
| T11 | Fix the staging leak in the engine itself, so neither memory nor reloads grow over a run | RAM, reload time | E2 | E15 |
| T1 | Port warcraftsim's game-code cuts into the hook (map-version memo, A* cell cache, lazy heuristic, soft-float, no JPEG decode), with a replay parity check | CPU/step | warcraftsim `shim/cuts.c` (MIT), same exe RVAs | open |
| T4 | Games niced +10, so learner and inference get the CPU first | steps/s | warcraftsim | open |
| T5 | Paint every 50 steps instead of never, if CPU/step grows over a long game (fog pass) | CPU/step late-game | warcraftsim | open |
| T6 | 4 games per Python worker instead of 2 | memory, steps/s | [todo/done](../todo/done/2026-10-selfplay-infra.md) | open |
| T7 | Evaluation games on the old 1080 Ti box over the LAN | verdict latency | warcraftsim | open |
| T8 | Less RAM a game, so more games fit | games, steps/s | E1 | E13 |
| T3 | Learner and inference sharing one GPU without time-slicing (pauses between minibatches, MPS) | inference latency, steps/s | warcraftsim | E3 |
| T9 | Shorter decisions: less of the actor's loop spent waiting on inference | steps/s | E1 | E4-E7 |
| T2 | GE-Proton Wine with fsync | CPU/step | warcraftsim; [9-benchmarks.md](../9-benchmarks.md) | E10 |

## 2. Learner (more learning per game played)

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| L1 | More gradient steps per update: 2 epochs and/or minibatch 16 | wins per game played | warcraftsim's recipe; Andrychowicz 2021 | open |
| L2 | `--lr` 4e-5 (baseline), 1e-4, 2e-4 from the same checkpoint | wins per game played | [learner-tuning.md](../todo/learner-tuning.md) | open |
| L3 | KL-adaptive learning rate (targets 0.005, 0.01, 0.02), and early stop at 1.5x target | stability, wins | rl_games; SB3 `target_kl`; Schulman 2017 | open |
| L4 | Extra critic-only passes (Phasic Policy Gradient) | explained variance, wins | Cobbe 2021 | open |
| L5 | Chunks replayed for a few updates with importance weights, if L1 helps | wins per game played | IMPACT (Luo 2020); V-trace (Espeholt 2018) | open |
| L6 | Anchor moved to a recent checkpoint, with a weaker pull, when a run plateaus (a rule, not by hand) | win-rate plateau | warcraftsim (+11 points, twice) | open |

## 3. Curriculum (after throughput and the learner: the next lever)

Tiers, scenarios and their progressions: [curriculum.md](../todo/curriculum.md). Each scenario is checked against
its scripted solution before it trains anything.

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| C2 | Scenario spec and runner: an `Episode` per scenario on combat staging, levels, success test, scripted solution, held-out seeds | all of tier 1-5 | SMACv2; PySC2 minigames | open |
| C3 | Tier 1 micro: focus fire, pull back, kite, surround, body block, hero snipe, spells one by one, spread, items, engage or not | win rate in fights, cast share | SMAC; Shao 2018 | open |
| C4 | Scheduler: levels per scenario, weights by learning progress plus a floor, real games at ~50% | wins per game played | Matiisen 2017; PLR (Jiang 2021); ACCEL (Parker-Holder 2022) | open |
| C5 | Snapshot benchmark: every snapshot scored on held-out scenarios | eval | | open |
| C6 | Terrain arena (chokes, tree lines) for body blocks, surrounds, kiting near trees | C3's terrain levels | | open |
| C7 | Replay staging: a pro game's state at minute N in one message | tiers 2-5 | Salimans & Chen 2018 | open |
| C8 | Tier 2: fights cut from pro replays at their first damage | win rate in real fights | | blocked: C7 |
| C9 | Tier 3 economy: saturation, no supply block, no idle money, upkeep, creeping | income, block and float rates | | blocked: C7 |
| C10 | Tier 4 builds: opening, tech timing, unit mix, placement | build-order distance, timings | AlphaStar's z | blocked: C7 |
| C11 | Tier 5 game plans: defend harass and rushes, attack at a timing, expand | win rate of the piece | | blocked: C7 |
| C1 | Scripted check of each skill before it becomes a scenario, first pulling back hurt units | does the skill pay? | [curriculum.md](../todo/curriculum.md) | E9 |
