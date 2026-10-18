# Lessons from warcraftsim

[RobinKa/warcraftsim](https://github.com/RobinKa/warcraftsim) trains agents on the same game build as
us (Legacy TFT 1.29.2 under Wine). Started 2026-09-23; about 200 commits in its first 6 days. Its
[experiment log](https://github.com/RobinKa/warcraftsim/blob/master/docs/experiments.md) is the most
relevant prior work we have. As of 2026-09-29 (read at commit of that date):

- **Small fights** (4v4/5v5 mirrors on flat maps): working RL, including focus fire, pulling hurt units
  back and hero spells; league self-play.
- **Whole games**: a clone of the built-in normal AI on its own sped-up 40-tile map (`duelrush`, ~2
  minute games) wins 6 of 32 against the normal AI (19%; human and orc 0). PPO from that clone has
  so far made it worse (0 of 60). Nothing beats the built-in AI in a real full game yet.

## How it differs from us

| | warcraftsim | us |
|---|---|---|
| Observations and orders | JASS script injected into each map; text tokens (changed units only); orders via script natives | DLL reads engine memory into binary records; orders as the player's network commands |
| Replays | its agents' orders are not in the `.w3g` | ours are; replays play back with observations |
| Maps | custom flat and sped-up duel maps | stock maps |
| Action | every own unit picks an order (or none) and a target every step | one decision per step: type, argument, anchor + N, target ([1-actions.md](1-actions.md)) |
| Input | ≤160 unit rows × 29 numbers + type/order embeddings, one global token; no map, items, buffs or spell state | richer rows, map patches, items, buffs, hero state ([2-features.md](2-features.md)) |
| Model | 3-layer transformer, 192 wide; minGRU memory optional (off) | [3-model.md](3-model.md) |

## Lessons, and how they apply to us

| Lesson (his evidence) | Applies to us? |
|---|---|
| **RL from scratch doesn't find team tactics.** Focus fire and pulling back each lose alone, so PPO ends at "let units fight" (~50%); 7.5% random orders turned 54% into 18%. Starting from a cloned script fixed it every time. | Yes. Supports cloning first. |
| **Plain PPO from a clone collapses** (57% → 21%): early updates come before the value head knows anything. Value-only warm-up, low learning rate and a per-update KL stop took a clone from 66% to 91%, past the script (71%). A KL leash to the clone helps early but caps later gains (75% vs 91%). | Yes: warm up the value head, move gently; treat the KL to the clone as a stabilizer to relax. |
| **Horizon matters.** Pulling back pays several seconds later: horizon 16 (8 s) never learned it, 64 did (48% → 60%). | Yes: long chunks, γ ≈ 0.999, λ ≈ 0.95. |
| **Self-play finds exploits.** The first league learned to run away (draws), then lost 96% to the game's simple attack-move opponent; policies cycled rock-paper-scissors. Draws as losses, a fixed built-in anchor in the league and opponent choice by who still wins (PFSP) fixed it. His curricula were exploited too (rushing an AI given a late start). | Yes: keep built-in AI anchors in every league; draws are not free; watch curricula for exploits. |
| **Pointing at specific units matters.** The pointer policy learned to finish the most damaged enemy, often the hero; fixed targeting rules could not. | Yes: our target head is a pointer. |
| **Clean the built-in AI's labels.** Engine-internal orders (resume harvesting was 29% of events, autocasts) and repeated harvest orders are not decisions; real gold↔lumber switches are (dropping them left human and orc without lumber). The AI retries unaffordable train orders every step: 52% of train labels were fake and the clone learned to hoard. | Yes, as a checklist for our cloning data. |
| **The clone cancels its own builds** (47%) because every unit is re-decided every step. | Mostly not: we decide once per step and a build keeps its worker unless the policy picks that worker again. Still worth a guard. |
| **Timing can't be cloned.** The AI's script runs on hidden timers; each local decision was calibrated, yet small deficits compounded (−15 points of win rate in the first 20 s alone). DAgger (the AI takes over from the clone's states) helped but not enough. | Yes. The clone's ceiling is low; human replays don't have hidden timers. |
| **"Can't afford" must be masked.** The AI only orders what it can pay for, so the clone never learned it; a third of its orders were refused. | Yes: our masks. |
| **Pick a point as x, then y given x.** Independent x and y paired one place's x with another's y. | Already: region then cell. |
| **Most unit-steps have no order** (~90%). | Not for us: one decision per step, no per-unit "none". |

## What we take

Clone first, as a start rather than the goal: built-in AI games (cleaned as above), plus human
replays where the version allows. Then RL with a warmed-up value, gentle updates, long horizons and
built-in AI anchors in the league.

## Update, 2026-10-02 (commit d282d33)

- **Results:** RL from their clone of the built-in AI, with a cloning loss alongside PPO (x0.005, on takeover games
  where the AI finishes a game the policy started), now wins 75-80% against the easy and normal AI and 16 of its
  first 30 against insane, all on their sped-up `duelrush` map (1-4 minute games). On the longer `duelfast` map
  (about 6 minutes) the same recipe won 4 of 346: too many workers, almost no heroes.
- **How they label the AI:** the orders the AI issues, intercepted in the map script, one label per unit per step
  (0 when it got none); engine orders, repeated harvest orders and orders it could not pay for are dropped. Army
  movement is their weakest part: each unit attacks on its own where the AI attacks as a group.
- **For us:** our AI games now include the captains' group orders (training/replays/convert.py `commands`): moves
  and attacks the engine gives several units at once, or to a point. Their material-lead shaping was farmed with
  workers, as ours was in sp4; a fine-tune on takeover games overwrote what RL had learned, a loss alongside it did not.
