# Scenarios and curriculum

Break the game into parts (micro skills, fights, macro), build a scenario for each, and train on them together with
real games. Each scenario's share of training follows how much the policy is still learning from it. The scenarios
double as a benchmark for every snapshot.

## Tiers

| Tier | What | Examples | Scored by |
|---|---|---|---|
| 1. Skills | One behaviour in a small fight that a script wins easily | focus fire, pull back a hurt unit, kite (ranged vs melee), potion timing, spell placement (chain lightning on a clump, purge a summon) | success rate vs the scripted solution's |
| 2. Fights | Combat mode as it is (../combat.md) | random armies, heroes, items; the attack-move ladder | win rate per stage |
| 3. Macro | Economy and production, no enemy army | play out a pro's opening (z); from a replay's minute N, the next 3 min: no supply block, no idle gold, workers saturated | the build reward's distances; block and idle rates |
| 4. Games | Whole games: ladder and self-play | | win rate, ladder stage |

## How the curriculum works

- **Mix, not stages.** Skills trained alone don't combine by themselves (a drill teaches how to kite, only fights
  teach when), and a skill nothing trains any more is forgotten. Real games (combat mode until whole-game RL is
  affordable) keep a fixed share, ~50%; scenarios fill the rest. AlphaStar, OpenAI Five and Pluto all trained on the
  whole task throughout.
- **Weight by learning progress, not only accuracy.** A scenario stuck at 5% wastes samples as much as one at 99%.
  Weight each by how fast its score moves (Matiisen 2017; PLR, Jiang 2021), plus a floor (~2-5%) so mastered
  skills keep being rehearsed.
- **Passed** = at the scripted solution's rate. It drops to the floor and stays as an eval.
- **Earlier starts as it succeeds.** Scenarios cut from replays start closer to the moment that matters, then earlier
  as the policy succeeds (reverse curriculum, Salimans & Chen 2018).

## Rules for every scenario

- The same network, observation, actions and reward channels as real games. No drill-only rewards.
- The policy is not told which scenario it is in; the critic may be (values differ by scenario).
- A scripted solution first. If it does not beat a script without the skill, the skill does not pay against this
  opponent, and RL is right to skip it.
- Start from pro replay states where possible: realistic positions without hand-placing units.
- Held-out seeds for evaluation.

## To build

**First, on the current setup**
- [ ] Scripted retreat check: attack-move vs attack-move plus pulling back units under 30% hp, combat mode, stages
      0.85 and 1.0.
- [ ] Moves relative to the unit (direction x distance from the anchor). cb9 moves on ~0.2% of steps: a random
      absolute move is bad on average, so move dies before its targets are learned; skill scenarios can't fix that.

**Scenarios**
- [ ] A scenario spec: setup (units and positions, or a replay and a moment), scripted solution, success test, time
      limit, seeds. Built on combat mode's staging (VectorSession.stage).
- [ ] Tier 1: the five skills above, each with its scripted rate measured.
- [ ] Tier 2: fights cut from pro replays at their first damage, as a combat-mode stage.
- [ ] Tier 3: an opening drill and a mid-game economy drill from replay states.

**Curriculum**
- [ ] A scheduler in the league: a score window per scenario, weights by learning progress plus floor, real games
      at a fixed minimum share.
- [ ] The scenario id as a critic-only input.
- [ ] Dashboard: each scenario's score, weight and scripted rate.

**Benchmark**
- [ ] Every snapshot scored on the scenarios (held-out seeds): which skill it gained or lost.

## References

- PySC2 minigames (Vinyals et al. 2017); SMAC (Samvelyan et al. 2019); SMACv2 (Ellis et al. 2023, arXiv 2212.07489):
  skill scenarios as benchmarks, each trained alone.
- Shao et al. 2018, StarCraft micromanagement with RL and curriculum transfer learning: small scenarios to large.
- Matiisen et al. 2017, teacher-student curriculum learning; Jiang et al. 2021, Prioritized Level Replay.
- XLand (Open Ended Learning Team 2021); AdA (Adaptive Agent Team 2023): general skills from a wide task mix.
- Salimans & Chen 2018: learning from a demonstration by starting from its states.
