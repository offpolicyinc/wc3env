# Combat: fights instead of whole games

A whole game runs 10-20 minutes before it says anything about a change to the trainer. A fight says it in under a
minute. Combat mode reuses the whole self-play loop (actors, inference, learner, league, dashboard). The only
difference: each episode is one fight between two identical Orc armies (three heroes each, plus units) on a flat map.

## Two modes, one loop

```
 training/selfplay/
   actor.py  inference.py  learner.py  league.py  batches.py  reward.py      shared by every mode
   modes/
     __init__.py   Step, Episode (the interface), get(name)
     game.py       --mode game    (default) whole games on Echo Isles
     combat.py     --mode combat  fights on the flat arena
```

The actor, learner and launcher take `--mode game|combat`. A mode supplies everything that differs:

| | game | combat |
|---|---|---|
| map | Echo Isles (`--map` overrides) | FlatArena |
| matchups | every pair in turn (or `--races`) | Orc vs Orc |
| plans | a pro plan for 90% of learning players | none (the empty plan) |
| ladder opponent | Warcraft's AI (`ai_agents`), easy/normal/insane, each handicapped 0.05 → 1.0 | scripted attack-move (`Episode.commands`), handicapped 0.35 → 1.0, then outnumbering us 1.1-2x; both players computer slots |
| episode start | the handicap on Warcraft's AI | a fight staged in the worker (`VectorSession.stage`) |
| rewards | `reward.Shaping` + win | army value traded + win |
| shaping weight in the learner | fades with the league's win rate | always 1 |
| episode over | Warcraft's result or the league's time limit | 30% rule held 2 s, 150 s limit, or Warcraft's result |
| map reloaded | at the time limit | after every fight |

An `Episode` is made at each episode's start and has four hooks: `start` (returns the first observations, or None if
it sent a staging step), `staged`, `commands` (scripted players' orders) and `step` (returns a `Step`: rewards per
learning player, over, outcome, minutes, reload). The actor has no `if combat` anywhere.

```
python -m tools.prepare.arena                                         # once: the map and its map data
python -m training.bench.combat_balance --fights 40                   # local: is a fight fair?
python -m training.cloud.selfplay start --name cb1 --mode combat --init runs/bc12/last.pt ... -- <learner flags>
```

## One fight

```
 FlatArena (Echo Isles flattened: no trees, cliffs, water, mines or creeps)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  [hall P0]                                                   [hall P1]   │  town halls stay (no buildings
 │                                                                          │  = a lost melee game)
 │                      random centre c, random direction d                 │
 │            siege   casters  ranged  melee │ melee  ranged  casters siege │
 │   P0:       ocat    oshm     ohun    ogru │ ogru   ohun    oshm    ocat  :P1
 │                                   ◄─gap─► c ◄─gap─►                      │
 │            player 1's army = player 0's mirrored through c               │
 └──────────────────────────────────────────────────────────────────────────┘
```

Staging a fight: one message to the game's worker, `VectorSession.stage`, carrying a list of debug ops and then the
step that starts the fight. The ops, in order:

1. Remove every unit of the last fight, and the start's workers.
2. Set both players' resources to 0.
3. Give both players every Orc upgrade.
4. Spawn both armies, each unit at full mana. Each hero is set to its level, given its items, and learns its skills,
   one a step before the fight's first observation (the agent never sees a hero with points left).
5. Set the scripted side's handicap.

Locally this took 0.03-0.1 s (a little more with the skill steps). The actor isn't blocked: the staged step comes back through `recv` like any other.

## One episode

```
 stage ──► the policy (and, in ladder games, the scripted side) decides every 250 ms
             │
             ├─ one side's army value < 30% of the other's for 2 s ──► that side lost: +1 / -1 on the win channel
             ├─ 150 game seconds and still undecided        ──► a loss for both (-1)
             └─ Warcraft ends the game (a whole side died)  ──► its result
 then: the map loaded again, the next fight staged on it
```

**Army value:** each living unit's gold + lumber times the square root of its share of hit points. Why the root
(Lanchester): a group's fighting strength goes roughly as its total hit points times its total damage, and a unit deals
full damage until it dies, so two Grunts at 50% (1.41 Grunts) beat one at 100% (1.0). With the plain share (until
2026-10-12) both were worth one Grunt: spreading damage paid as much as killing, so focus fire and saving hurt units
earned nothing directly. A Grunt at 10% is now 0.32 of a Grunt, so killing it pays 0.32 of its price. A hero counts its price times
1 + 0.25 per level above 1 (a guess: a level 5 Blademaster ~1,050, about five Grunts). The value comes from each
side's own observation.

Locally, scripted against scripted, fights with heroes lasted 39-150 game seconds (median about 75), a second or two
of real time each at full speed.

**Reward per step** (besides the win): the army value traded that step, as a fraction of the army's starting value V.
`dealt = (enemy value lost) / V`, `taken = -(own value lost) / V`, in the critic's existing "dealt" and "taken"
channels. Example: a side starting at 4,000 kills a 200-value Grunt and loses a 100-value Headhunter in one step:
dealt +0.05, taken -0.025. Summed over a fight this is the final value margin, so it is at most ±1, the same size
as the win. It is potential-based shaping (the potential is the value margin): it can't change which policy is best
and can't be farmed (taking damage and healing it back nets zero). The learner
weights it 1 throughout (`Combat.shaping`), unlike whole games' shaping, which fades as the league wins.

## Decisions

| Decision | Why |
|---|---|
| Orc mirror only | Our full games are Orc vs Orc. |
| Three heroes per army: 3 of the 4 Orc heroes (Blademaster, Far Seer, Tauren Chieftain, Shadow Hunter), each at a random level from 1 to 10 | Heroes decide Warcraft fights. Random levels cover early and late fights. |
| Each hero's skills learned at staging: a random legal build, one skill per level (basic skill k needs hero level 2k − 1, the ultimate taken at 6) | Each fight has a different hero setup. Learning skills is a macro decision, so in a fight it's set, not chosen. |
| Each hero carries 3-6 random items: one healing item (Potion of Healing, Healing Salve, Potion of Greater Healing), the rest from 38 shop consumables and permanent items (potions, scrolls, wands, rings, orbs, stat items) | Items are part of fighting: potions, scrolls and wands to use, stats to fight with. No Town Portal, Staff of Sanctuary, invisibility or summons: they leave the fight or change it too much. |
| Random units every fight: a food budget of 8-40 (besides the heroes) filled with uniformly drawn types from 11 Orc units (melee, ranged, casters, air, siege) | SMACv2's lesson: fixed scenarios get memorized as openings, randomized ones need the policy to read the situation. |
| Same army and mirrored positions for both sides, around a random centre and direction, with each unit nudged randomly | Fair by construction. The randomness moves the fight around the map so no side or spot is favoured. |
| Rows by role (melee in front, heroes just behind, then ranged, casters, siege at the back) | A plausible army shape. The jitter keeps it from being one fixed formation. |
| Lost below 30% of the other side's army value, held for 2 s | Once one side has under a third of the other's value the result is settled and the rest is cleanup. The 2 s hold keeps one burst (a Chain Lightning on a clump) from ending a fight that could still turn. |
| Undecided at 150 game seconds: a loss for both | warcraftsim's self-play learned to run away while draws were free. 150 s, not 90: with heroes, 90 s cut off 14 of 40 scripted fights, and the decided ones took up to 88 s. |
| Win/loss plus the army value traded each step | The win alone is one number per 75 s fight; the trade says which steps mattered. Potential-based, so it adds no new optimum (above). |
| The learner the local player (slot 0), the opponent a computer slot (slot 1) | The opponent's units cast their spells on their own, as Warcraft's computer players' do; the learner's cast only what it orders. The goal is a policy that casts its own spells and beats a competent opponent. cb1-cb5 had both computer slots (below): fair, but the learner's spells were Warcraft's. In self-play the slot-1 seat plays but is not learned from. |
| Ladder games: Warcraft's attack-move as the opponent (every 3 s its units attack-move to the centre of our army, picking their own targets), one ladder level (`attack-move`) | A fixed baseline that never changes: the question "does the policy fight better than Warcraft's own attack-move?". |
| Ladder stages: the opponent handicapped 0.35, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95, then even (1.0), then outnumbering us 1.1x to 2x in steps of 0.1 in unit food (same heroes; the last unit a coin flip, so each stage's extra food is right on average: a tenth more of a small army is less than a Grunt) | The curriculum. 0.35 first: bc12 won 5 of 5 at 0.05. Past even, more units rather than tougher ones: the handicap is a hidden damage scale the policy can't see, extra units it can, and they're real Warcraft. Attack-move against itself from the smaller side at 1.5x: 2 of 12 won, so hard but possible. |
| A stage passed at 80 wins in a block of 100 fights (blocks one after another, in start order), or 18 of its first 20 | Blocks, not a window rolling every fight: a stage sees hundreds of fights, and a rolling window checked after each one is soon passed by a lucky stretch (cb3 left 0.95 at 60-70%). 100, not 50: cb5 left 0.95 on one block of 41 of 50 while its blocks ran 22-35 (~60%), one lucky block in ~48. A player truly at 65% passes a block of 100 0.1% of the time (of 50: 1.6%), at 70% 1.7% (8%), at 80% 56%, at 85% 93%. The 18 of 20: stages far too easy don't take 50 fights. Applies to whole games too. |
| Every fight on a freshly loaded map (2026-10-19) | The game clock is one of the model's inputs: each fight then starts where a whole game does, instead of at some point of a 10-minute clock shared by several fights. Loading the map takes ~1 s (a new process at the memory budget 2-3 s), and the other games step on meanwhile (`VectorSession.send_reset`). Before, fights followed each other in one game (restaging 0.1 s) and the map was loaded again past 10 game minutes. |
| Only fight orders legal: noop, move, attack-move, smart, attack, stop, hold, the three casts, use item (`FIGHT_TYPES`); the rest masked before each decision | Sell, drop, revive and the like were legal on most steps (revive on 85% with no altar on the map) and only waste steps, or exploration. |
| An exploration floor (`--explore-floor 0.005 --explore-tries 50000`): every order type but noop has at least 0.5% chance wherever legal, until the run has chosen it 50,000 times, then PPO alone decides | Rare but valuable orders never got tried: cb5 cast a targeted spell on 0.04% of steps (about once in 7 fights) and focus-fired on 0.05%, too few for PPO to learn whether they pay. 0.5% is ~1.8 tries per 1.5-minute fight per type, at most ~5% of steps for all of them (warcraftsim: 7.5% random actions took 54% wins to 18%). A floor, not a push: it acts only while a type is below it. 50,000 tries: ~45 minutes of training. The learner logs each type's share (`share_*`), its tries (`tries_*`) and how many are still floored. |
| The floor re-armed at each promotion (each type's tries back to 0), and `--explore-min`: once a type's tries are done, at least that chance for good | cb6's floor came off every type by update 330; hold and stop then fell to 0.3% of steps and cast_point from 9% to 1%. What did not pay against a weaker opponent may against a stronger one, so each stage gets its own 50,000 tries per type; the minimum (0.1%: ~1% of steps over the 10 types) keeps any type from dying out. |
| A unit a spell made (ward, wolf, trap, Mirror Image illusion) worth its spell's mana at the shop's price (4/3 gold a mana: a Potion of Mana is 200 gold for 150), shared by the units one cast makes; mana costs read from the casters' own abilities each step. Learn allowed in fights | Wards and summons cost no gold, so they were worth 0: killing an enemy Healing Ward or Serpent Ward paid nothing, and an illusion counted as a whole hero. A Healing Ward is now ~267, a Serpent Ward 40, a Spirit Wolf 50, a level-3 Mirror Image illusion ~44. A hero levelling up mid-fight gets a skill point it could not spend. |
| UPGO beside PPO (`--upgo 1`, AlphaStar's and DI-star's weight): a second clipped term whose advantage follows what came after a step only while each next step did at least as well as the critic expected, then takes the critic's value | A good action is not charged for a later mistake it did not make: a Chain Lightning at step 10 is not blamed for a hero walking into the enemy at step 30. Per channel, as GAE; scaled by the advantages' spread but not centred (its sign is its point). |
| Every upgrade, full mana, no resources | Casters can cast from the first second, and nobody can train or build mid-fight. |

## Measured: player 1 had an edge, now fixed

`training/bench/combat_balance.py` has Warcraft's attack-move fight itself, 40 fights:

| Slots | Player 0 won | Player 1 won | Draws |
|---|---|---|---|
| Player 0 the local (human) slot, player 1 an agent slot (as whole games have it); no heroes | 1 | **36** | 3 |
| Both computer slots without an AI (combat mode); no heroes | 18 | 18 | 4 |
| Both computer slots, with heroes, 150 s limit | 16 | 22 | 2 |
| Same, with the 2 s hold (30 fights) | 14 | 15 | 1 |

(16 against 22 out of 38 decided fights is within chance.)

The cause: wc3env loads every agent slot other than the local player as a computer slot (with its AI suppressed), and
Warcraft has computer players' units use their spells on their own. Player 1's Shamans, Witch Doctors and Spirit
Walkers fought by themselves; player 0's did nothing unless ordered. Unit stats and upgrades were identical on
both sides (checked).

Combat mode made both players computer slots for cb1-cb5. That was fair, but it cast the learner's spells for it,
so it never had to learn them (2026-10-12: cb5 ordered a targeted spell on 0.04% of steps). Now the opponent keeps
its casting and the learner has none: the even stage is the 36-to-1 matchup, and beating it takes casting. In
self-play, as in whole games, only slot 0 is learned from.

## Files

- `training/selfplay/modes/combat.py`: the army draw, positions, staging ops, army value, the 30% rule, the scripted
  attack-move, the rewards (`Combat`, `Fight`).
- `training/selfplay/modes/game.py`: whole games, as self-play had them before the split.
- `training/selfplay/actor.py --mode combat`, `learner.py --mode combat`.
- `src/wc3env/vector.py`: `VectorSession.stage` (ops and a step in one message; a spawn may carry `level`, `mana`,
  `items` and `learn` for each unit it makes).
- `tools/prepare/arena.py`: the map, now with its map data (`src/wc3env/data/maps/(2)FlatArena.json`, which the
  policy's features read).
- `training/cloud/selfplay.py --mode combat`: copies the map into each games sandbox and passes `--mode` to the actors
  and the learner.
- `training/bench/combat_balance.py`: the fairness check above.

Tested locally: learner and inference on CPU, one actor with 2 games for 900 steps, with heroes. 10 ladder fights
ended and were reported to the league, the learner made 6 updates from the fight chunks, and no game failed.
**Not yet run on Modal.**
