# Observations

The API reads engine state using world coordinates and object IDs, independently of the
camera and display resolution. Fog filtering belongs to the observing player.

See [observation.json](../examples/observation.json) for a real example. Object ids are integers;
type/ability ids are four-character strings. `units` and `visible_enemies` share this shape:

```json
{"unit_id": 15419, "type_id": "hpea", "owner": 0, "x": 4404.2, "y": 3010.1,
 "hp": 220, "max_hp": 220, "mana": 0, "max_mana": 0,
 "structure": false, "hero": false, "level": 0, "buffs": ["Bblo"]}
```

`buffs` lists the buff ids on the unit, any unit in view, enemies included: `Bprg` purged, `Bblo` bloodlust, `BOae` an endurance aura. Their remaining time is not reported.

Own units add their current order and abilities; own structures also add their production. Enemy orders, production and abilities are never reported.

```json
{"order": {"name": "harvest", "target_id": 11527, "x": 4544.0, "y": 3584.0}, "abilities": []}
{"order": null, "state": "constructing", "state_seconds": 35.0, "queue": [], "queue_seconds": 0.0}
{"order": null, "state": null, "state_seconds": 0.0, "queue": ["hfoo", "hfoo"], "queue_seconds": 20.0}
{"order": null, "abilities": [{"ability_id": "AHbz", "level": 1, "mana_cost": 75, "cooldown_seconds": 6.0, "cooldown_remaining": 0.0}]}
```

| Own-unit field | Meaning |
|----------------|---------|
| `order` | `null` when idle, else `{name, target_id, x, y}`. `name` is the order string (`move`, `attack`, `harvest`, `repair`, ...); a build order's name is the structure's type id (`hhou`). `target_id` is a unit, item or destructable id, or `null` for a point order |
| `state` | Structures only: `constructing`, `upgrading` (to another structure, such as Town Hall to Keep) or `null` |
| `state_seconds` | Total seconds that construction or upgrade takes, not the time left; pair it with the `construct_start`/`upgrade_start` event time |
| `queue` | Structures only: queued unit and research type ids, in progress first, at most seven |
| `queue_seconds` | Total seconds for the item in progress; it began at the latest `train_start`/`research_start` for that structure |
| `abilities` | Every owned unit, including structures: every ability the unit has learned, with its `level`, live `mana_cost`, `cooldown_seconds` and `cooldown_remaining`. Internal abilities (movement, inventory, hero attributes) are listed too; tell them apart by id. Readback does not establish castability, pathing or immunity. Completed research is not listed: count `research_finish` events (binary observations list it: `research`) |

A unit inside something is absent from `units` and listed in `inside` instead: a worker in a gold
mine, a Wisp in an Entangled Gold Mine, a Peon in a Burrow, a builder inside the structure it is
building (Orc, Night Elf), transport cargo. It cannot be commanded until it comes out. `cast` with
order `cancel` removes a structure's last queued item or stops its construction or upgrade.

| Field | Meaning/shape |
|-------|---------------|
| `protocol_version` | 1 |
| `observer` | Observing player slot |
| `sequence` | Observation counter per player, starting at 0 |
| `game_time_seconds` | Current simulation time |
| `player` | `{gold, lumber, food_used, food_cap}` |
| `players` | Runtime player categories, relations, controllers and sharing flags; [metadata](#observation-metadata) |
| `map` | World bounds in engine coordinates; [metadata](#observation-metadata) |
| `score` | The observer's 25 engine score counters; [metadata](#observation-metadata) |
| `units` | Own units on the map; units inside mines, buildings and transports are in `inside` |
| `inside` | Own units inside a mine, building or transport: `{unit_id, type_id, x, y, hp, order}`; `order.target_id` is the mine a gatherer works |
| `visible_enemies` | Other visible units, including neutrals; observer's fog applies |
| `items` | Visible, living, unowned ground items: `{item_id, type_id, x, y}`; consumed tomes/runes are excluded as soon as their life reaches zero |
| `inventory` | `{unit_id, slot, type_id, charges}` entries; slots 0-5 |
| `destructables` | Visible living objects: `{id, type_id, x, y, hp, resource, invulnerable}`; `resource` is `lumber` or null, without guaranteeing harvestability |
| `events` | Visible/owned events since this player's previous observation |
| `events_lost` | Visible events overwritten since this observer last read |
| `orders` | Every order given to the observer's units since its previous observation, oldest first: `{unit_id, name, target_id, x, y, origin, queued, game_time_seconds}`, plus `item_type_id` for giving or dropping an item; `name` as in a unit's `order`; `target_id` for a target order, `x, y` for a point order, else null. `origin` is who gave it: `player` (a player's command: the network, a replay or `act`), `script` (JASS: AI scripts and map triggers) or `engine` (on its own: a worker returning its load, a unit acquiring a target) |
| `orders_lost` | Orders overwritten since this observer last read (each player keeps the latest 2048) |
| `chat` | Lines the person at the game window typed into the chat box since the previous observation (Enter, text, Enter); only in the local player's observation and only in a visible background window; printable ASCII |
| `result` | Empty string, `victory`, `defeat` or `draw`; replay EOF can end a session without results |

Events snapshot values and visibility when fired. Each configured player has its own cursor.
Each observer retains its latest 1024 visible events; hidden events cannot evict them. Overflow is reported once in `events_lost`.
Secondary references hidden from the observer have id `0` and type `""`; primary ownership
does not reveal a hidden attacker, summoner or buyer.

Text summaries preserve nonzero `events_lost` even if no events remain. They label the
observer's units `own` and use supplied player relationships; older observations fall back
to player IDs. Slot IDs alone do not establish neutrality, hostility or alliances.

| Event kind | Fields |
|------------|--------|
| `death` | `unit_id, type_id, owner` before defeat transfers |
| `construct_start`, `construct_cancel`, `construct_finish`, `upgrade_start`, `upgrade_cancel`, `upgrade_finish` | `unit_id, type_id` of structure (its type when the event fired) |
| `train_start`, `train_cancel` | `unit_id` of building, `type_id` of ordered unit |
| `train_finish` | `unit_id` of building, `trained_id, type_id` |
| `research_start`, `research_cancel`, `research_finish` | `unit_id` of building, `type_id` of upgrade |
| `hero_level` | `unit_id, level` |
| `hero_learn`, `spell_effect` | `unit_id, ability_id` |
| `summon` | `unit_id` of summoner, `summoned_id, type_id` |
| `item_pickup` | `unit_id, item_id, type_id` |
| `item_use` | `unit_id, type_id` |
| `item_sold` | `unit_id` of shop, `buyer_id, type_id` |
| `unit_sold` | `unit_id` of shop, `buyer_id, type_id` of the hired mercenary or tavern hero |
| `attacked` | `unit_id` of victim, `attacker_id` |

The host wrapper adds `ticks_skipped`: `max(0, int(elapsed_seconds) - 1)` in realtime, else 0.

Regenerate the example with `python tools/example_observation.py`.

## Binary observations

`GameConfig(observation="binary")` returns `wc3env.binary.BinaryObservation` objects instead of
dictionaries (`pip install wc3env[binary]` for numpy). The DLL writes fixed-size records into the
shared memory mapping `Local\wc3hook-obs-<pid>`, as BWAPI's client reads its game data: nothing
is formatted or parsed, and a session step observes every agent slot in the same round trip as the
step. Computer players are not observed; `session.game.rpc.observe_binary(slot)` reads one on request. Layouts are in [`wc3hook/obsbin.h`](../../wc3hook/obsbin.h) and
[`binary.py`](../../src/wc3env/binary.py); arrays are copies that stay valid after later steps.

| Attribute | Contents |
|---|---|
| `player`, `sequence`, `game_time_seconds` | As in JSON; `sequence` is shared with JSON observations |
| `gold`, `lumber`, `food_used`, `food_cap`, `result`, `events_lost`, `orders_lost`, `score` | As in JSON |
| `time_of_day` | 0 to 24; day lasts from 6 to 18. Melee games start at 8 |
| `units` | Every own unit, including those inside a mine, building or transport (flag `INSIDE`), and every other visible unit: `unit_id, type_id, owner, x, y, hp, max_hp, mana, max_mana, flags, level, order_id, order_target, order_x, order_y, state, state_seconds, queue_seconds, armor, damage_min, damage_max, attack_period, move_speed, facing, resource`. `flags` combines `OWN`, `STRUCTURE`, `HERO`, `INSIDE`, `ILLUSION` (own illusions only: a player cannot tell enemy ones apart, so an illusion of a hero is a `HERO` like the real one) and `DEAD` (the observer's dead heroes, listed for revival with position, level, hero stats and items; every other row is alive). Order and production fields are set for own units only; `order_target` is `0xffffffff` for a point order; `state` indexes `STATES`. The combat numbers are what the unit's info panel shows, after upgrades, items, auras and attributes: armor, the first weapon's damage range and seconds per attack (0 without an attack), move speed and facing in degrees. `resource` is the gold left in a mine |
| `abilities` | Own units' learned abilities: `unit_id, ability_id, level, mana_cost, cooldown_seconds, cooldown_remaining` |
| `buffs` | `unit_id, buff_id` for every listed unit |
| `queue` | Own structures' production queue: `unit_id, slot, type_id`, slot 0 in progress |
| `inventory` | `unit_id, slot, type_id, charges`, for every listed unit with an inventory: a player can click an enemy hero and see its items |
| `heroes` | Every listed hero: `unit_id, xp, skill_points, strength, agility, intelligence`; attributes include item and aura bonuses; `xp` and `skill_points` are the observer's heroes only (0 for others) |
| `research` | The observer's upgrades with a level above 0: `type_id, level` |
| `items` | Visible ground items: `item_id, type_id, x, y` |
| `destructables` | Visible living destructables: `id, type_id, x, y, hp, flags` (`LUMBER`, `INVULNERABLE`) |
| `events` | `kind` (engine event id; `EVENT_KINDS` names it), `unit_id`, `other_id`, `type_id`, `value`. `unit_id`, `other_id` and `type_id` are the JSON event's fields in order (`other_id` is `trained_id`, `summoned_id`, `item_id`, `buyer_id` or `attacker_id`; `type_id` is also an `ability_id`); `value` is a death's owner or a hero level |
| `orders` | `unit_id, order_id, kind, target_id, item_type, x, y, origin, queued, time_ms`: `item_type` is the type of the item a give or drop moves (0 otherwise); `kind` is `IMMEDIATE`, `POINT` or `TARGET`; `target_id` is `0xffffffff` and `x, y` are 0 where the kind has none; `origin` indexes `ORIGINS` (`engine`, `player`, `script`) |

Ids are integers; `fourcc()` turns a type, ability, buff or build-order id into its four characters.
Order ids are the engine's (a build order's id is the structure's type id). `players`, `map` and `chat`
are JSON only: read them with `session.game.rpc.observe(player)` when needed, for example after reset.
Unit ordering, positions and values match the JSON observation of the same state; the combat numbers,
`heroes`, `research`, `time_of_day`, dead heroes and other players' inventories are binary only.

## Observation metadata

Every observation includes `players`, `map`, and the observer's `score`. These are
engine facts, independent of window size, map filename, and race.

`players` lists occupied slots, the observer, and runtime neutral slots. Each entry
has `id`, `kind` (`player` or `neutral`), `relation` (`self`, `ally`, `enemy`, or
`neutral`), `team`, `active`, `controller` (`agent`, `human`, `computer`, or `none`),
`shares_vision`, and `shares_control`. Relations are relative to the observer;
the sharing flags describe what that player grants the observer. Alliances can
change during a match. Neutral players can also be enemies or allies. Empty
ordinary slots are omitted; slot numbers alone do not imply a category.
Text summaries use these relationships when supplied and plain player IDs for
older observations. `visible_enemies` keeps its historical name: it contains
visible units owned by anyone else, including allies and neutrals.

`map.bounds` contains `min_x`, `min_y`, `max_x`, and `max_y` in world coordinates.
Bounds describe the engine's world rectangle, including its border; they do not
promise navigable ground. Only four numbers are cached, read without creating an
engine object; the cache clears on reset.
Observing creates no game objects: listed units get a JASS handle (created once per unit, reused), which is
not an engine object and leaves object ids, and so replays, unchanged.

Visible living destructables additionally have `resource` (`lumber` or null) and
`invulnerable`. Lumber classification comes from the loaded object-data tree
target flag, including custom type IDs, rather than a list of known tree IDs.
A resource label identifies a potential target, not a guarantee that a particular
unit can harvest it: worker abilities, invulnerability, orders, and access still
matter. Crates, gates, and scenery are not automatically lumber sources.

`score` is a dictionary of the 25 engine `PLAYER_SCORE_*` counters; the ordered
names are exported as `wc3env.protocol.SCORE_FIELDS`. It includes production,
kills, razing, hero/item counters, gathered/transferred resources and score totals.
Only the observer's counters are exposed. They reset with the map and follow the
engine's scoring rules, including its initial-unit counts; they are not a custom
reward function. The fake server supplies zero scores because it does not model
production and combat scoring.
