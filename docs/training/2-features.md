# 2. Features

What the network sees each step: observation → arrays. The network itself is
[3-model.md](3-model.md).

## What the env gives us

Binary observations (`src/wc3env/binary.py`), per player, fog-filtered:

- header: game time, gold, lumber, food used/cap, result, 25 score fields;
- units: id, type, owner, x, y, hp, max hp, mana, max mana, flags (own, structure, hero, inside),
  level, current order + target, construction/upgrade state and progress, queue time;
- abilities of own units (level, mana cost, cooldown, cooldown left), buffs, build queues,
  hero inventories, ground items, visible destructables (trees), events (attacked, died, trained...).

All numbers already, no strings; turning them into arrays is numpy work next to the game.

## How the references do it

Both split the input into units, a map grid and scalars:

- **Units**: one row per unit. AlphaStar ≤512 rows, numbers bucketed (√hp → one-hot of 39), x/y as
  8-bit binary. Pluto ≤768 rows, plain fractions clipped to ±5, position plus sin/cos at 8 scales.
- **Map**: 128×128 in both: static (pathable, buildable, height), dynamic (visible, explored), and
  each unit's vector scattered onto its cell.
- **Scalars**: resources, supply, race, upgrades, action masks, last actions. AlphaStar includes game
  time; Pluto doesn't.

## Decided

| Decision | AlphaStar | Pluto | Us |
|---|---|---|---|
| Number encoding | buckets / binary | fractions, sin/cos position | fractions, sin/cos position |
| Map resolution | 128×128 | 128×128 | 128×128 (matches the 8×16 point grid) |
| Game time | yes | no | yes (day/night, timings) |
| Masks as inputs | yes | yes | yes |
| Recent actions | last action | last 4 + per-unit actor/target flags | last 4 + per-unit actor/target flags |
| Fogged enemy buildings | dropped | slow memory | kept as rows with "last seen N s ago" (the WC3 UI shows them) |

## Unit rows (≤256)

Own units, visible enemy units, remembered enemy structures, creeps, gold mines, ground items. Trees
go on the map grid (thousands of them), not in rows.

| Field | Encoding |
|---|---|
| type | embedding id (all unit/building/item types, ~900) |
| owner | embedding id: me, enemy, creep, neutral passive, item |
| x, y | position / map size, plus sin/cos at 4 scales |
| hp, mana | fraction of max, and max / 1000 |
| flags | structure, hero, inside, remembered (not currently visible) |
| last seen | seconds since visible (0 when visible) |
| level | / 10 |
| current order | embedding id of an order class (idle, move, attack, harvest, build, cast...) |
| order target | dx, dy to the target point or unit |
| progress | construction / upgrade / training fraction |
| queue | embedding id of the first queued type (a structure's unit or research; an own worker's building while it walks to place it), queue length |
| hero | XP progress, skill points, per ability: level, cooldown left fraction, castable |
| inventory | 6 item-type embedding ids (summed) |
| buffs | buff embedding ids (summed) |
| recent | was actor / was target in each of the last 4 decisions |
| hit | was attacked / lost hp this step, and by whom (attacked events carry the attacker) |

### WC3 vs SC2/BW: what a unit row must carry

AlphaStar's rows carry SC2 things: shields, energy, cloak/burrow, cargo, weapon cooldown, build
progress, orders, buffs, upgrade levels, harvester counts, power. WC3 has less of some (no shields,
no power) and much more around heroes, items, creeps and damage types.

The list below comes from auditing every game function the engine registers for scripts
(`python tools/natives.py`: 1,398 natives, including the `Blz*` ones of this game version) against
the binary observation. Status: **have** (in binary observations), **static** (per type, from
`reference.json`), **derive** (featurizer), **add** (a native the DLL can call), **hard** (no
native; needs reverse engineering the unit's memory).

| What | Why it matters in WC3 | Status / source |
|---|---|---|
| hp, mana, level (all visible units) | mana is everywhere; hero level | have |
| type stats: armor class, attack type, damage, range, attack period, move speed, air/ground, cost, food | WC3's damage table (pierce vs heavy, magic vs fortified...) decides fights; explicit stats help rare types | static |
| hero XP, skill points | when a level comes, when to learn | have (`heroes`: own heroes) |
| hero str/agi/int with item bonuses | hp, mana, damage, armor, attack speed follow them | have (`heroes`: any visible hero) |
| current armor, damage range, attack period, move speed, max hp/mana | the real fight math after upgrades, items, auras, attributes | have (the info panel's numbers: a Blademaster reads 26-48 damage and 5.2 armor with its agility) |
| facing | turn rate before attacking/casting | have |
| status: summoned, stunned, snared, polymorphed, ethereal, magic immune, invisible, illusion | status effects everywhere | have: each is a buff (timed life, stun, ensnare, polymorph, banish, wind walk...) or the type; own illusions are flagged `ILLUSION` (enemy ones look real to a player) |
| buffs (Slow, Bloodlust, Devotion Aura, Wind Walk, ...) | which effect exactly | have (ids) |
| buff time left | when Sleep/Entangle ends | hard (no native) |
| own hero abilities: level, cooldown, mana cost | spells decide fights | have |
| own inventory + charges | items | have |
| enemy hero inventory | a player can click the enemy hero and see its items | have |
| item cooldowns | potions and scrolls share cooldowns | hard (no native) |
| attack cooldown left | hit-and-run, orb walking | hard (no native; the attack period above is not the time left) |
| summon time to live | temporary units | hard (no native); the summoned flag is add |
| gold mine gold left | expansions, mines running dry (the UI shows it) | have (`resource`) |
| creeps: level, camp, asleep | creeping is half the early game | level have; camp static (map data); asleep not yet (`UnitIsSleeping` exists but could not be verified: no creep was seen asleep in tests) |
| own dead heroes | revive mask and choice | have (flag `DEAD`, with level, stats and items) |
| rally point | where new units go | add: `GetUnitRallyPoint/Unit` (lower priority) |
| construction / upgrade / training progress | timings | have |
| inside a transport / burrow / building | Orc burrows, zeppelins | have (flag) |
| worker carrying gold/lumber | income timing | derive from orders (approximate) |
| neutral buildings: shops, taverns, labs, mercenary camps, fountains | buying, hiring, healing | have (rows) + static (what they sell) |
| remembered enemy structures | the UI keeps them under fog | derive (featurizer memory) |
| who attacks whom | focus fire, peeling | derive from attacked events and own orders |

Scalars: own research levels (have: `research`), time of day (have:
`GetFloatGameState(GAME_STATE_TIME_OF_DAY)`), upkeep tier (derive from food), enemy upgrade levels
(add, lower priority: same native, for upgrades a player could see by clicking enemy units), blight
(Undead; map channel, later).

Done: the combat numbers, `heroes`, `research`, `time_of_day`, gold left in mines, own dead heroes, own
illusions and other players' inventories are in binary observations (tests/e2e/test_binary.py pins each
to a known value; no measurable cost in CPU or memory). Still open: the "hard" rows, rally points,
enemy upgrade levels, sleeping creeps and blight.

## Map (128×128 over the playable bounds)

11 channels: trees (remembered), gold mines, own units, own structures, enemy units, enemy structures
(including remembered), creeps, the last 4 decisions' target points, walkable and buildable ground, start
locations. Static parts come from each map's data (`python -m tools.prepare map --map <path>`: playable
bounds from the script's camera bounds, the 32-unit pathing grid, the placed trees, mines, camps, shops and
start locations), prepared for all 16 maps of the ladder replays. Trees change: the grid starts from the
map's trees and cells near our units take what the observation shows (a cut or eaten tree is gone; trees in
the fog stay as last seen). Checked on each map a few seconds into a replay: every tree the game shows is a
map tree at the same position, every ground unit stands on walkable ground inside the bounds.

The model cuts it into 64 patches of 16×16 (see [3-model.md](3-model.md)).

## Scalars

gold, lumber (log), food used/cap, upkeep tier, game minutes, time of day, own race and enemy race,
own research levels, own unit counts per type (√), the masks from [1-actions.md](1-actions.md), the
previous 4 decisions' type and argument.

## Environment gaps

A visibility/explored grid (the fog itself) is not observed: "near our units" stands in for it. The DLL's
fog query over a grid would give the real one.

## Checked on replays

`python -m training.replays.check` plays each replay with both players observed and checks one player's
features against the other's truth. On all 675 in-sync replays: 26M rows of visible enemies match their
owner's own rows in type, position and hit points; 99.3% of remembered enemy structures still stand (the
rest were destroyed out of sight); production progress stays in [0, 1] and is 0.9 or more when a unit
finishes in 99.6% (construction progress comes from hit points, which rise from 10% to full however many
workers speed it up; time since start misread Human power-building). It found one environment bug: a
player's own units were left out of its observation for a moment while teleporting or blinking.

## Code shape

A pure function `features(obs, static_map, history) -> dict of arrays`, running next to the game,
tested on recorded observations, reused unchanged by the trainer.
