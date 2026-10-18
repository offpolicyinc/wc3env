# Rewards

Every reward self-play pays, in one place. Code: [training/selfplay/reward.py](../../training/selfplay/reward.py)
(the numbers), [league.py](../../training/selfplay/league.py) `League.shaping` (the fade),
[learner.py](../../training/selfplay/learner.py) (how the channels are added up).

A win is worth 1. Everything else is measured against that.

## The terms

While it never wins, the shaping outweighs win/loss many times over: that is the tutorial. sp33 had it 10× smaller
(~0.3 a game against the time limit's −0.7, under the noise of the win channel's critic) and learned nothing from it.

| Channel | Pays for | Amount | Cap per game | Earlier pays more | Weight |
|---|---|---|---|---|---|
| **win** | winning; losing; reaching the time limit | +1; −1; −1 | — | no (the discount does that) | **1, always** |
| **build** | how close the player's first 20 builds are to the pro's opening (the plan's first 20 makes; workers and food buildings left out, as AlphaStar's): each build pays the edit distance it closed (insertions, deletions, substitutions); an extra or misplaced item can cost | 0.5 per step of distance | ~13 items: ~6.5 | yes | shaping |
| **units** | what the player has made vs what the pro made over its game, as two sets of types (units, buildings, heroes, tier upgrades; workers left out): the first of a type the pro made pays, the first of a type it never made costs (AlphaStar's built-units reward, the Hamming distance closed) | ±0.75 per type | ~13-17 types: ~10-13 | ×0.5 after 8 min, ×0.25 after 16, 0 after 24 | shaping |
| **upgrades** | the same for research, each level its own item (melee weapons 2 is one): a level the pro researched pays, one it did not costs (AlphaStar's upgrades reward) | ±0.75 per level | 1-5 levels: ~0.75-3.75 | as units | shaping |
| **dealt** | damage its own units dealt to the enemy's units (the game's damage events name the source: attacks and spells; not creeps' or anyone else's): damage, at most the hp the unit had, × its gold+lumber price / its max hp; an enemy building's whole price only when it is destroyed, if its own units damaged it in the last 5 s (hits on a building its workers repair pay nothing) | 2 per 1,000 | the enemy's whole side | no | 0.1 + 0.9 × shaping |
| **taken** | the enemy player's damage to own units, the same way (not creeps'), and own buildings' price as they are destroyed after the enemy's units damaged them within 5 s (negative) | −2 per 1,000 | — | no | 0.1 + 0.9 × shaping |

Examples. A grunt (200 gold) killed: dealt +0.4 while it never wins, +0.04 once it wins half its games. A burrow
(160) destroyed: +0.32; hit down to 10 hp and repaired: 0. The enemy's hall first seen: +0.5. The enemy's
base and army destroyed: ~+10 to +20, then ~+1 to +2. A pro who made grunts, raiders and a Stronghold: the first grunt
+0.75, more grunts 0, the Stronghold +0.75, a headhunter −0.75; spiked barricades researched (and the pro did) +0.75. Opening barracks, altar, grunt
(the pro's) built as altar, barracks, grunt: altar +0.5, barracks 0 (swapped), grunt +0.5; a fourth unplanned item −0.5.

An example plan (Orc vs Orc, the bank):

```
units     (17 types): otrb ogru orai ospm Obla Oshd Otch oalt obar obea ofor okod ovln owtw ostr ofrt ...
upgrades  (levels):   Roen 1, Rorb 1, Rowt 1, Rwdm 1
```

## Fading

**shaping = max(0, 1 − win rate / 0.5)**, where the win rate is over the last 50 ladder games against the current
rung and its handicap stage (the easy AI, then normal, then insane, each starting at 0.2 of its hit points).
Self-play games don't count toward it but get the same weight.

| Win rate on the rung | shaping | build, units, upgrades | dealt | taken |
|---|---|---|---|---|
| 0% (never wins) | 1 | full | full (2 per 1,000) | full (−2 per 1,000) |
| 25% | 0.5 | half | 0.55 of it | 0.55 of it |
| 50% or more | 0 | **off** | 0.1 of it (0.2 per 1,000) | 0.1 of it |

- Moving to the next handicap stage or up a rung (80 wins in a block of 100 games) starts the record over, so shaping goes
  back to 1 against the harder opponent.
- Dealt minus taken is the trade, from the start (2026-10-11; before, taken cost nothing until it won, and it lost
  ~46 units a game: sp35). Only the enemy player's hits count in taken: creeps' made creeping cost, with nothing on
  the other side. bc12 at hp 0.05 (16 games): dealt 4.4, taken −0.9 a game (11 of 16 ahead); sp35 at update 1,000
  at hp 0.2 (32 games): dealt 1.1, taken −4.7 (none ahead). Once it wins, the trade stays, small next to a win.

**Earlier pays more** (OpenAI Five's): the term × 0.6^(game minutes / 10). Minute 0: ×1, minute 10: ×0.6, minute
20: ×0.36. units and upgrades use AlphaStar's steps instead: ×1 to minute 8, ×0.5 to 16, ×0.25 to 24, then 0.

## Where the targets come from

Each learning player gets a pro's plan at the start of each game (90% of games; the rest get none, and then units
and upgrades pay nothing): a won game of the same matchup from the plan bank (`runs/plans.npz`, read from the pro2
replays). Orc vs Orc: 70 pro player-games, 32 won.

- **build** uses the plan's first 20 makes (`plan.order`); **units** and **upgrades** use everything the pro made over
  the game (`plan.mix`: sqrt(count) / 4; research ids are upgrades, the rest units), as types present (count ignored,
  except research levels).

## How they're added up

Each channel has its own critic output (`model.CRITIC`) and its own return (GAE). The learner multiplies each
channel's advantage by its weight now and adds them, so a fade changes only the weights, not what the critic has
learned.

## Why these and not others

| Not paid | Why | Run |
|---|---|---|
| gathering gold or lumber | mined without spending | sp1 |
| workers, farms, food in use | 39 workers, 3,200 lumber banked, never attacked | sp4 |
| units by cost | 10 grunts pay as much as 10 grunts' worth of tauren: spam the cheapest | — |
| any build-order item matched, uncapped (0.05 each, every time) | farmed: gold mined up, wins vs the idle player 20% → 6%. units pays each type once | sp6 |
| plan: each item up to the pro's count, one channel for units and research together (10 in all) | AlphaStar splits them, each with its own critic, and makes no count target | sp33-sp34 |
| the opening only in order, none after the first item out of order | paid 0 in sp34's first 1,100 updates: the clone's order is never the pro's exactly | sp34 |
| hero levels | paid 0 all of sp34: a level is 200+ experience, too far apart to learn from | sp34 |
| hero experience (0.5 per 200), creep value destroyed (1 per 1,000) | bc11 already creeps and levels its heroes, and both paid without ever meeting the enemy (the sp6 kind of farming); whatever of them helps to win, the win pays for. Dropped 2026-10-11 with their critic channels | bc11 eval |
| damage to buildings | repairable: hit, repaired, hit again pays every time | — |
| army near the enemy's base, per second | pays for parking there without fighting | — |
| the opening once per item, any order | the same items as `plan`, which already pays early ones more: dropped before any run, for the edit distance (AlphaStar's) | — |
| damage to enemy units from anyone | sp34's dealt grew with ~1 kill a game: creeps hitting the easy AI's creeping units paid it | sp34 |
| material held at the time limit | 98% of games went to the clock | sp6 |

Everything that pays is capped per game (the plan's items, the enemy's buildings and side), except damage, which
needs fighting.

## Hall and tower upgrades

A building's upgrade_start event names it as it was (Great Hall), its upgrade_finish as it became (Stronghold).
`plan` counts the finish. Until
2026-10-10 the replay converter matched the start against the new type and dropped every hall and tower upgrade
(0 of 344 orc pro plans had a Stronghold); pro3 and bots3k2 are the data converted again with the fix.
