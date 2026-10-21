# Scenarios and curriculum

Break the game into parts (micro, fights, economy, builds, game plans), build a scenario for each with a clear
progression from easy to the real thing, and train on them together with real games. Each scenario's share of
training follows how much the policy is still learning from it. The scenarios double as a benchmark for every
snapshot. After throughput and the learner ([../rsi/ideas.md](../rsi/ideas.md)), this is the next lever.

## Tiers

| Tier | What | Scored by |
|---|---|---|
| 1. Micro | One combat behaviour in a small fight that a script wins easily | success rate vs the scripted solution's |
| 2. Fights | Combat mode as it is ([combat.md](../combat.md)), then fights cut from pro replays | win rate per stage |
| 3. Economy | Workers, supply, spending, creeping; no enemy army | the build reward's distances; idle, block and float rates |
| 4. Builds | Openings and tech from a pro plan (z), against light pressure | build-order and unit-mix distances; timings |
| 5. Game plans | Pieces of whole games: defend a rush or harass, attack at a timing, expand | win rate of the piece |
| 6. Games | Whole games: ladder and self-play | win rate, ladder stage |

## Progressions

Each scenario climbs levels; a level passes as combat stages do (80 wins in a block of 100, or 18 of the first 20).
Levels make it harder along the same axes everywhere: more units, more randomness (SMACv2: fixed setups get
memorized), a stronger opponent (script, then Warcraft's AI, then self-play snapshots), terrain, and last, starts
cut from pro replays. A passed scenario drops to the floor and stays as an eval.

**Tier 1, micro** (Orc first, as combat mode)

| Scenario | Levels (easy -> real) | Success | Script |
|---|---|---|---|
| Focus fire | 4v4 Grunts at even hp -> mixed hp -> mixed units -> enemy hero in reach | win with more value left than the script's | attack the lowest effective hp in range |
| Pull back hurt units | 3v3 melee -> ranged -> under enemy spells -> mid-size armies | units saved (alive under 30% hp at the end) | below 30% hp move behind the army, return when healed |
| Kite | 1 Headhunter vs 1 Grunt -> 3v3 -> vs faster units -> near trees | win without losing the ranged units | attack, step away during the cooldown |
| Surround | 4 Grunts vs a fleeing hero in the open -> hero with escapes (Wind Walk) -> inside a fight | hero trapped (no path out) and killed | move to the free cells around the target |
| Body block | 1 unit blocks a fleeing unit at a choke -> 2 units in the open -> while the armies fight | the fleeing unit caught | stand in its path's next cell |
| Hero snipe | enemy hero behind its army -> with its potions -> with its escapes | enemy hero killed first | every unit on the hero, the rest ignored |
| Spell use, one at a time | static target -> moving -> choosing when and on whom: Chain Lightning on a clump, War Stomp into a surround, Shockwave along a line, Hex the strongest, Purge a summon, Bloodlust before contact, Healing Wave on the hurt hero | value traded vs no casting | cast at the best target by a hand rule |
| Spread vs area damage | under Chain Lightning, Shockwave, catapults | value lost vs standing still | move apart when a caster is in range |
| Items | potion at low hp -> under burst -> which item when | hero survives | drink below 40% hp |
| Engage or not | the same fight at 0.6x to 1.5x of our size | value kept: fight when ahead, leave when behind | attack-move if ahead, retreat if not |

**Tier 3, economy** (start from a game's state at minute N; needs replay staging, below)

| Scenario | Levels | Success | Script |
|---|---|---|---|
| Saturation | from the start -> from a minute-5 state -> with an expansion | gold and lumber per minute vs the pro's | 5 on gold per mine, the rest on lumber |
| No supply block | alone -> while producing -> at upkeep's thresholds | seconds supply-blocked | a burrow when food is within 4 of the cap |
| No idle money | spend gold above a threshold -> while teching | gold floated per minute | queue units at every free building |
| Upkeep | choose army size around 50 and 80 food | income kept vs army made | the pro's food curve |
| Creeping | one camp without losses -> a route of camps -> a route by a timing | hero xp by minute, units lost | the pro's creep route |

**Tier 4, builds** (the plan input as given)

| Scenario | Levels | Success | Script |
|---|---|---|---|
| Opening | first 5 builds -> 10 -> 20 -> under a harass | build-order distance to the plan | the pro's order |
| Tech timing | tier 2 by the pro's time -> tier 3 | seconds late | |
| Unit mix | the plan's mix by minute 8 -> 12 | unit-mix distance | |
| Placement | buildings legal and near the hall -> not blocking the mine path | placements refused, path length to the mine | |

**Tier 5, game plans**

| Scenario | Levels | Success | Script |
|---|---|---|---|
| Defend harass | a Blademaster hits the peons -> with Battle Stations (burrows) -> while expanding | peons lost | burrow, then fight with the hero |
| Defend a rush | an early tower or Grunt rush -> mixed | survive with the base | |
| Attack at a timing | push at the plan's timing against an AI that just expanded | buildings razed | attack-move with the army |
| Expand | take a second base at the right moment and hold it | expansion alive at minute 15 | |

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

- The same network, observation, actions and reward channels as real games. No drill-only rewards: the success
  test schedules and scores, it never pays.
- The policy is not told which scenario it is in; the critic may be (values differ by scenario).
- A scripted solution first. If it does not beat a script without the skill, the skill does not pay against this
  opponent, and RL is right to skip it.
- Start from pro replay states where possible: realistic positions without hand-placing units.
- Held-out seeds for evaluation.

## Design: scenarios on combat mode (C2)

A scenario is combat mode's `Fight` with four parts swapped; everything else (the network, the actions, the reward:
the win and the army value traded) stays combat's, so a scenario trains the same policy on the same signal:

| Part | Combat mode today | A scenario |
|---|---|---|
| Setup | `army(rng)` mirrored about a random point | its own draw per level: unit types and counts, positions, hp, items, trees |
| Opponent | attack-move every 3 s (`attack_move`) | its own script per level (attack-move, a fleeing hero, a kiter, a caster) |
| Scripted solution | none | a hand rule for the learner's side: the bar a level is passed at |
| Success | the win | its own test on the fight's record (units saved, hero killed first, kept value): schedules and scores, never pays |

- **Code.** `training/selfplay/scenarios.py`: each scenario a class (name, `LEVELS`; `stage_ops`, `opponent`,
  `solution`, `baseline`) in a registry; combat's fight is the first ("fight", level = its stage). In training the
  actor's `--scenarios focus_fire:0,focus_fire:1 --scenario-share 0.25` turns that share of the ladder's fights into
  scenario fights, the opponent named "focus_fire/1", outside the ladder's count (the board bins them apart). A
  scheduler (C4) would set the levels and shares from learning progress; until then they are fixed flags.
- **Runner.** `training/bench/scenarios.py` (was combat_balance.py): a scenario, a level, held-out seeds,
  and players among checkpoints, `solution` and the baseline script, all on the same draws, with Wilson intervals
  and `--record`. A scenario is built only after its runner shows the solution beats the baseline (curriculum's own
  rule; E9's pull-back did not).
- **Hook.** Spawned units' hp and mana (`spawn`'s "hp": mixed-hp levels); trees with `destructable` for terrain
  (body block, surround, kite near trees) on the flat arena instead of a new map (C6).
- **First:** focus fire, level 0 (4 v 4 Grunts at even hp; solution: each unit attacks the enemy in range with the
  lowest hp; baseline: attack-move), then the same with mixed units. Its runner result decides whether it is built.

## To build

**First, on the current setup**
- [ ] Scripted retreat check: attack-move vs attack-move plus pulling back units under 30% hp, combat mode, stages
      0.85 and 1.0.

**Infrastructure**
- [ ] A scenario spec: setup (units and positions, or a replay and a moment), levels, scripted solution, success test,
      time limit, seeds. Each scenario an `Episode` in `training/selfplay/modes/` on combat mode's staging
      (`VectorSession.stage`).
- [ ] An arena with terrain (chokes, tree lines) for body blocks, surrounds and kiting near trees; FlatArena has none.
- [ ] Replay staging: a pro game's state at minute N (buildings, workers, resources, research, hero levels, items)
      staged in one message, for tiers 2-5.
- [ ] Scripted opponents per scenario: a fleeing hero, a harassing Blademaster, a rush.

**Scenarios**, in order
- [ ] Tier 1 micro, each with its scripted rate measured.
- [ ] Tier 2: fights cut from pro replays at their first damage, as a combat-mode stage.
- [ ] Tier 3 economy, tier 4 builds, tier 5 game plans.

**Curriculum**
- [ ] A scheduler in the league: levels per scenario, weights by learning progress plus floor, real games at a fixed
      minimum share.
- [ ] The scenario id as a critic-only input.
- [ ] Dashboard: each scenario's level, score, weight and scripted rate.

**Benchmark**
- [ ] Every snapshot scored on the scenarios (held-out seeds): which skill it gained or lost.

## References

- PySC2 minigames (Vinyals et al. 2017); SMAC (Samvelyan et al. 2019); SMACv2 (Ellis et al. 2023, arXiv 2212.07489):
  skill scenarios as benchmarks, each trained alone.
- Shao et al. 2018, StarCraft micromanagement with RL and curriculum transfer learning: small scenarios to large.
- Matiisen et al. 2017, teacher-student curriculum learning; Jiang et al. 2021, Prioritized Level Replay;
  Portelas et al. 2020, automatic curriculum learning for deep RL (survey); Parker-Holder et al. 2022, ACCEL.
- XLand (Open Ended Learning Team 2021); AdA (Adaptive Agent Team 2023): general skills from a wide task mix.
- Salimans & Chen 2018: learning from a demonstration by starting from its states.
