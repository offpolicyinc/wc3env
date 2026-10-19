# Ideas to test

Ideas that improve training of the current model, ranked within each section by expected gain over cost. Changes
to the network itself (size, depth, inputs, heads) need a retrain and are in [next-model.md](next-model.md). Seeded
2026-10-18 from [todo/](../todo/README.md) and [warcraftsim.md](../warcraftsim.md). One line each; the detail lives
in the linked doc. When an idea is tested, its status points to its row in [log.md](log.md) (`E<n>`); new ideas go
where their rank puts them.

Status: `next` (top of the queue), `open`, `running`, `E<n>` (tested, see the log), `blocked: <why>`.

## 0. Baselines (before anything else)

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| B1 | Home throughput baseline: 4 actors x 8 games, combat mode, `--wine`; steps/s, CPU busy, actors' `ms:` split | steps/s | [home-setup.md](../home-setup.md#sizing-the-run-the-first-job) | next |
| B2 | Size the home run: more games until the CPU is ~90% busy or steps/s stops rising, then more actors with fewer games each | steps/s | warcraftsim (+13% from more actors) | open |
| B3 | Fixed evaluation: cb9 at stage 0.85, N fights on held-out seeds; its interval width sets N for every later verdict | eval | [combat.md](../combat.md) | open |

## 1. Throughput

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| T1 | Port warcraftsim's game-code cuts into the hook (map-version memo, A* cell cache, lazy heuristic, soft-float, no JPEG decode), with a replay parity check | CPU/step -16 to -28% | warcraftsim `shim/cuts.c` (MIT), same exe RVAs | open |
| T2 | GE-Proton Wine with fsync; measure CPU/step and game-restart rate (Proton 11 crashed on Modal) | CPU/step ~-20-28% | warcraftsim; [9-benchmarks.md](../9-benchmarks.md) | open |
| T3 | Learner and inference sharing one GPU: learner pauses ~8 ms after each minibatch (CUDA events) | inference latency, steps/s | warcraftsim (39 -> 6 ms, +22% steps/s) | open |
| T4 | Games niced +10, so learner and inference get the CPU first | steps/s | warcraftsim | open |
| T5 | Paint every 50 steps instead of never, if CPU/step grows over a long game (fog pass) | CPU/step late-game | warcraftsim (fog 17% of main thread by step 700) | open |
| T6 | 4 games per Python worker instead of 2 | memory, steps/s | [todo/done](../todo/done/2026-10-selfplay-infra.md) | open |
| T7 | Evaluation games on the old 1080 Ti box over the LAN | verdict latency | warcraftsim (~30 min instead of 1.5-2 h) | open |

## 2. Learner (more learning per game played)

| # | Idea | Moves | Rests on | Status |
|---|---|---|---|---|
| L1 | More gradient steps per update: 2 epochs and/or minibatch 16 (learner idle ~2/3 of each update) | wins per game played | warcraftsim's recipe; Andrychowicz 2021 | open |
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
| C1 | Scripted retreat check: attack-move vs attack-move plus pulling back units under 30% hp, stages 0.85 and 1.0 | does the skill pay? | [curriculum.md](../todo/curriculum.md) | open |
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
