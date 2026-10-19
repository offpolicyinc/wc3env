# Philosophy: how the loop works and what it optimizes

This folder runs a self-improvement loop: the work of an elite ML research team (read, hypothesize, build, measure,
decide, write down), done continuously by agents with the user steering. Four files:

| File | What |
|---|---|
| [philosophy.md](philosophy.md) | This: the goal, what to optimize, the rules every experiment follows |
| [ideas.md](ideas.md) | What to try next on the current model, ranked; each idea with its source and the number it should move |
| [next-model.md](next-model.md) | Changes to the network that need a retrain, batched into the next model |
| [log.md](log.md) | One row per experiment (`E<n>`): what, why, against what, the numbers, the verdict |

## The goal

**An AlphaStar-level Warcraft III agent, trained on one machine (one RTX 3090 Ti), in record time.** Concretely, in
order: beat every handicap stage in combat mode; beat Warcraft's insane AI in whole games more often than
[warcraftsim](../warcraftsim.md) (85-89% on one 3090); then beat strong human players. AlphaStar took 44 days on
3,072 TPU cores; the point here is to get there with orders of magnitude less, through engineering and better ideas,
not scale.

## What to optimize, in order

1. **Throughput: steps/s, decided and learned.** Every other result arrives faster when the loop is faster. Measure
   the whole loop (games, actor, inference, learner) and fix the part that limits it; a faster part that is not the
   limit changes nothing ([9-benchmarks.md](../9-benchmarks.md#lessons)).
2. **Clean, minimal, beautiful code.** The repo is public and meant to be read. Fewer lines beat more; one way to do
   a thing; names that say what something is; no dead flags, no special cases left behind by experiments. An
   experiment that wins is merged cleanly or not at all; one that loses is deleted, its lesson kept in the log.
3. **Rapid iteration and evaluation.** Many small, decisive experiments beat a few large ones. Prefer the cheapest
   test that can answer the question: combat mode before whole games, a short run from a fixed checkpoint before a
   long one, an offline benchmark before a live run.

## How to work

**Research the frontier first.** Before building, find who has solved this: papers (highly cited ones first, then
strong recent work), open implementations (DI-star, warcraftsim, rl_games, CleanRL, SB3), and our own
[decisions](../8-decisions.md) and [tried and dropped](../8-decisions.md#tried-and-dropped). Cite what an idea
rests on in [ideas.md](ideas.md). Don't reinvent what a paper has already measured, and don't retry what we have
already dropped without a new reason.

**One experiment, one question.**
1. Take the top idea in [ideas.md](ideas.md). State the hypothesis and the number it should move, before running.
2. Build the smallest change that tests it, behind a flag if it must coexist with the baseline.
3. Run it beside a baseline from the same checkpoint, stage and seeds; change one thing.
4. Compare by **games played** (sample efficiency) and by **wall clock** (what we actually pay), and read the
   diagnostics too (update KL, clip fraction, KL to the clone, the critic's explained variance, order-type shares).
5. Stop as soon as the answer is clear: read steps/s from the change between two reports; a win-rate gap from a
   binomial interval, not one lucky block.
6. Write one line in [log.md](log.md). Keep: merge cleanly and update [8-decisions.md](../8-decisions.md). Drop:
   delete the code and add a "tried and dropped" line. Then re-rank [ideas.md](ideas.md).

**Be honest with numbers.** Every number in these files comes from a log, W&B or code, with the run named. A result
that is within noise is "level", not a win. Negative results are results: log them as carefully as wins.

**Keep evaluation fixed.** The same evaluation (opponent, stage, seeds, number of games) across experiments, so log
lines compare. Training-game win rates are a rough guide; verdicts come from evaluation games.

**Automate the loop itself.** Anything done twice by hand (launching a baseline, reading metrics, writing a log
line, an evaluation) becomes a script. The team's speed is the loop's speed.

## Rules the user has set

- Concise docs, high level, with references; don't add work the user didn't ask for.
- Check configs and numbers before stating them.
- Commit only when asked; the repo is public. Checkpoints and `.env` stay out of git; never print `.env`.
