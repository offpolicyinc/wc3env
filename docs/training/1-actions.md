# 1. Actions

What the network outputs every step and how it turns into wc3env orders.

## What the env accepts today

`VectorSession.send(i, {slot: [actions]})`, each action `{unit_id, command, arguments}`, one unit per
action. Commands: move, stop, attack, smart (right-click), harvest, build (optionally `auto_place`,
the engine's own site search), train, research, learn, cast (by order name), use_item, drop_item,
revive, select, buy. `queued` shift-queues. See `docs/specs/actions.md`.

The env has no notion of *selection* and no action mask. Orders the engine refuses (not enough gold,
missing tech) are dropped silently. So the action layer below is ours: it lives in `training/`, turns
one network decision into a list of per-unit orders, and computes the masks.

## How the references do it

**AlphaStar**: one step = type (~300) → delay (when to look next, 0-127 game steps) → queued → select
up to 64 units one at a time with a pointer network → target unit (pointer) or target point (256×256).
Every head is masked by what's legal. Rich, but the 64-step unit pointer and the 65k-way point head are
the expensive, hard-to-learn parts; TStarBot-X and DI-star shrank or down-weighted both.

**Pluto**: one step = type (20) → argument (233: which unit/building/tech/ability/group/size) → either
a unit pointer or a map cell picked coarse (8×8) then fine (16×16 inside it). Selection is "the N
nearest own units around an anchor unit", N from {1,2,4,8,16,all}. The DLL does the no-brainers:
chooses the least busy producer, the worker, the geyser, addon spot. Masks come from the DLL and are
also inputs to the network. Fixed 4 decisions per second.

## Proposal

Pluto's shape, with the selection folded into each action so there's no hidden selection state.

One decision per step (250 ms, i.e. up to 240 decisions a minute; a pro plays ~200 APM). Each
decision is a tuple of heads, sampled in order, each seeing the previous ones:

```
type      which kind of action                          (21)
argument  unit/building/research/ability/item type      (~300 for one race, ~900 all)
actor     anchor unit (pointer over own units)          only for unit orders
same_type only the anchor's unit type, or any type      {yes, no}
size      how many units nearest the anchor             1-16 or all
target    unit (pointer over all entities)  OR  point (coarse 8×8, then fine 16×16)
```

Heads a type doesn't use are skipped and contribute nothing to the loss.

### Selection (decided)

Pluto's "anchor + N nearest", but stateless: every action names its own anchor, same_type and N, so
there is no remembered selection, no separate select step and no control groups. The translator
takes the N own units nearest the anchor that can carry out the action (and match its type if
same_type). N=1 is exactly the anchor, which covers hero and single-unit micro. Pluto only takes
units within 128 px of the anchor; WC3 armies spread wider, so start with no hard radius (or a cap
around 1000 game units) and tune it from real games.

Why not AlphaStar's pointer (pick up to 64 units one by one): it only buys exact arbitrary subsets
("these 3 of 8 clumped archers"), which N-nearest approximates; AlphaStar learned it by imitating
humans, while Pluto shows anchor + N works from self-play. A pointer head can be added later
without changing the rest.

### Decided

| Decision | AlphaStar | Pluto | Us |
|---|---|---|---|
| When to act | learned delay head (~2.7/s) | fixed, every 6 frames (4/s) | fixed, every 250 ms (4/s) |
| Shift-queue | "queued" head on every action | never | never, for now (a yes/no head is easy to add) |
| Action shape | one flat list (~300: `Train_Marine`, `Build_Barracks`...) | type (20) × argument (233) | type × argument |
| Map points | one softmax over 256×256 | 8×8 region, then 16×16 cell | 8×8 region, then 16×16 cell |
| Build placement | exact cell, failures fail | exact tile; geysers/addons snapped | rough point, engine snaps (`auto_place`); no walling tricks |
| Who builds | network | code | network (worker as anchor) |
| Which building trains | network | code (least busy) | code (least busy) |
| Worker gold/lumber | network (SC2 rally auto-mines) | network | network (WC3 rally-on-mine auto-mines) |
| Selection | pointer, up to 64 units, per action | anchor + N nearest, remembered | anchor + N nearest, per action |
| Legality masks | SC2 API + static tables | own rules in the DLL | own rules in Python, allow when unsure |

### Action types

| # | Type | Uses | Becomes |
|---|---|---|---|
| 0 | no-op | — | nothing |
| 1 | move | actor, size, point | `move` each |
| 2 | attack-move | actor, size, point | `attack` with x,y |
| 3 | right-click unit | actor, size, target unit | `smart` (attack enemy, harvest mine/tree, repair, load, pick up item, follow) |
| 4 | attack unit | actor, size, target unit (own units allowed: denying XP) | `attack` with target |
| 5 | stop | actor, size | `stop` |
| 6 | hold position | actor, size | `cast holdposition` |
| 7 | train | argument | `train` in the least busy building that can make it |
| 8 | build | argument, actor (a worker, N=1), point | `build` by that worker, `auto_place` so the engine snaps to the nearest valid site (also Haunted/Entangled Gold Mine onto a mine) |
| 9 | research / upgrade | argument (incl. town hall upgrades) | `research` in the least busy building that can |
| 10 | set rally | actor (a building), size, point | `cast setrally` |
| 11 | cast, no target | argument (incl. autocast on/off), actor, size | `cast` on units that have it ready |
| 12 | cast at point | argument, actor, size, point | same |
| 13 | cast on unit | argument, actor, size, target unit | same |
| 14 | learn hero skill | argument, actor | `learn` |
| 15 | use item | argument (item type), actor (hero holding it, N=1), target unit or point if the item needs one (`item_targets.json`) | `use_item` with that slot |
| 16 | buy | argument (item, mercenary or tavern hero), actor (the buying unit, masked to units in range of a shop selling it) | `buy` at the nearest such shop |
| 17 | sell | argument (item type), actor (hero holding it, masked to heroes in range of a shop) | `drop_item` onto the nearest shop |
| 18 | drop / give item | argument (item type), actor (hero holding it), target unit (give) or point (drop on the ground: e.g. drop +hp/+mana items before healing, since WC3 keeps hp/mana as a percentage when the max changes, then pick them back up) | `drop_item` onto that unit or point; picking back up is right-click |
| 19 | revive | argument (hero type) | `revive` at the nearest free altar |
| 20 | cancel | actor | `cast cancel` on a building |

Hero and item play is the core of WC3 (focus the enemy hero, save your own, potions/scrolls, constant
buying and selling), so every item action names the hero doing it: with two heroes carrying potions,
"which hero drinks" is the decision. Focus fire is right-click/attack with N=all and the enemy hero as
target; pulling a hero back is move with N=1.

Example: "send my 6 grunts near the base to attack-move to the enemy expansion" is one decision:
type=attack-move, actor=one of the grunts, size=6, point=(coarse cell 5, fine cell 137). The
translator picks the 6 own units nearest the anchor that are of a type that can attack, and issues
6 `attack` orders.

"Train a peon" is type=train, argument=`opeo`; the translator picks the Great Hall or Stronghold
with the shortest queue. The network never picks the building, same as Pluto.

"Build a burrow" is type=build, argument=`otrb`, actor=peon 12, point=(region 18, cell 40). The
network picks the worker (decided): in WC3 pulling a gold worker vs a lumber worker vs the nearest
one is a real economic choice, not a no-brainer. The actor mask allows only workers.

### Masks

Each head outputs a score for every option; illegal options are set to −∞ before the softmax, so they
get probability 0. Masks follow earlier heads (the argument mask depends on the sampled type, etc.),
are stored with each step so the learner recomputes the same probabilities, and are also fed to the
network as inputs (AlphaStar and Pluto both do this).

Where legality comes from (decided): our own rules, like Pluto. Pluto's DLL has its own predicates
and cost tables over raw game memory and doesn't use BWAPI's checks; AlphaStar got action-type
availability from the SC2 API, which WC3 doesn't have. We compute masks from the binary observation plus
`reference.json`, in C next to the features (`wc3hook/masks.c`: in the game hook for live play, as a library
through `training/policy/native.py` for recorded observations). Rule: when unsure, allow. A wrongly allowed option costs one
wasted step; a wrongly forbidden one can never be learned. Reading the engine's own command-card
check is a possible later upgrade if refused orders turn out common.

Implemented in `wc3hook/masks.c` (the tech tree generated from `reference.json` by `tools/feature_tables.py`;
the argument vocabulary in `training/policy/masks.py`):

- **type**: unit orders when there are units; train/research/build when something is affordable (with a
  30-gold/lumber margin for income during the step), unlocked (`requires`; an upgrading Town Hall counts as
  its next tier; the 2nd and 3rd hero need a Keep and a Castle; the first hero is free) and has a building
  that makes it (90% built counts); cast when an ability is ready with the mana; learn with a skill point
  and the hero level; items when carried.
- **argument** given type: the same lists. The vocabulary is every unit, structure, item, upgrade and
  ability type plus every order id in the engine's range (851970-852800), 2,000-odd entries.
- **actor**: own units that can do it; units inside a burrow or mine too, and towers and uprooted Night
  Elf ancients for unit orders; any own unit as a buyer (the game names whichever is near the shop).
- **target**: any row. The rows include the 48 visible trees nearest our workers and buildings (targets for
  gathering and rallies) and neutral buildings seen before (mines, shops).
- **point**: any cell.
- Neutral units under our control (mercenaries, charmed creeps) may cast any order no known ability gives:
  their spells use creep orders (`creepheal`) the reference does not tie to their abilities.

Two things the replays taught the masks:

- An action often becomes possible during its 250 ms step (a skill point, gold or a cooldown arriving). The
  data set writer gives such a label to the first step whose observation allows it (within one second),
  which is also the first step a policy could have taken it.
- Game data names some orders differently from the engine (`absorbmana` vs `absorb`) or not at all
  (Devour Magic, Cold Arrows, creep spells). `training/replays/type_orders.json`, built from what every
  unit type cast in the 675 replays (`python -m training.replays.check --type-orders`), adds those orders
  to the types that use them; a neutral type no replay showed casting may give any unexplained order.

Checked on the whole data set (`python -m training.cloud.verify`: 1,350 files, 4.9M steps, 1.95M actions):
every label's type, argument, actor and target is allowed by its step's masks. On average 18 of 21 types
are allowed, and 3-7 arguments for build, train, research and learn.

### Automatic no-brainers

Only choices with no strategy in them go to the translator:

- train/research use the least busy building that can (Pluto does the same).

Everything economic stays with the network (decided): which worker builds, and sending workers to
gold or lumber (right-click a mine or tree), as in Pluto. The one allowed shortcut is WC3's own:
a town hall's rally point on a gold mine makes new workers mine by themselves.

## Replay labels

Pro replays become (observation, action) pairs by playing them with observations
(`training/replays/convert.py`, `python -m training.convert_replays`). The DLL records every order a player's
command gives each unit, as commanded (origin `player`; script and engine orders are tagged apart), so
each command is one action: its units give the anchor (nearest their middle), same_type and size, its
order the type and argument. Shop purchases come from `item_sold`/`unit_sold`. A command that changed
nothing is dropped: the units were already doing exactly that (repeat right-clicks: ~8% of commands) or
production never started (full queue, no gold).

`training/tests/test_round_trip.py` checks every type survives translate → play → capture → convert.

Checked on all 675 in-sync replays (`python -m training.replays.check`: 2M labels, both players):

- translating a label back gives the player's own order, target and point every time; 87% also command
  exactly the player's units. Size is an exact count (1-16, or all), not Pluto's buckets {1, 2, 4, 8, 12,
  all} (74% on the first replays, against ~90% with exact counts). The anchor is the commanded unit nearest
  the group's middle, or another of them whose nearest units are exactly the group. Of the 13% left, half
  are off by one unit and a fifth pick up a nearby worker; the rest are groups that aren't anyone's nearest
  (three wounded grunts out of a clump, a control group across the map), which only a unit-by-unit pointer
  (AlphaStar) could express. Leaving workers out of army selections would fix ~1%, not worth another head.
- production goes to another building that can make it 0.05% of the time (the translator's least-busy pick).
- dropped as no-ops: repeat clicks (smart 100k, attack 26k, move 25k...) and production never started
  (21k); labels whose actor was not yet in the step's first observation (0.04%).
- left out as designed: own shops' sales (they come from item_sold: 9k), moving items between slots (3k),
  patrol and attack-ground (330).
- 23% of steps with an action have two or more (a pro's burst within 250 ms); the extra ones wait for the
  next step.

`training/tests/test_round_trip.py` covers every type and each race's special orders (haunt, entangle,
uproot, eat tree, root, burrow, battle stations, unsummon, item pickup and aimed items).

## Size of the space

For Orc only: 21 types, ~120 arguments, ≤100 own units, ≤256 targets, 64 + 256 point cells.
AlphaStar's per-step choice is much larger (64 sequential unit picks and 65,536 points). The heads are
cheap: most of the model cost is in reading the observation, not in these logits.

## Later, if needed

- More than one decision per step (a "done" token, as AlphaStar's pointer uses) if 4/s proves too few
  for hero micro: a save like "potion + blink out + focus their hero" is 3 decisions, 0.75 s.
- A delay head (AlphaStar) to act less often when nothing happens; saves inference, not game time.
- Control groups à la Pluto; probably unnecessary with anchor+size selection.

Left out on purpose: patrol, moving items between slots, camera, chat. Shift (queued) is a head: after a build, pros Shift-click the mine 61% of the time, and the same order unqueued cancels the build. Resigning is not an action: like Pluto (value ≤ −0.95 for 10 s), code
ends hopeless games to save compute.
