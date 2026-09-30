# Actions

Each action is `{unit_id, command, arguments}`. Host validation rejects malformed/non-finite
arguments and unowned units before sending any player's batch. Target widgets and shops must
be visible; acting IDs must refer to units. Actions are delivered in list order, including
repeated commands for the same unit. There is no action-count limit; the complete RPC request
must fit the transport's 65,535-byte frame. Additional `act` calls append to the same player's
buffer without advancing game time in stepping mode. Buffers grow as needed and clear on reset.
Coordinates must be finite with magnitude at most 1,000,000; this is a numeric safety bound,
not the map's playable boundary.

Python accepts `Action` objects or dictionaries (omitted `arguments` means `{}`). Argument
snapshots preserve JSON types: dictionaries with string keys, lists, valid Unicode strings,
booleans, null, signed 64-bit integers and finite floats. Tuples, non-string keys and other
Python objects are rejected rather than converted. Nested containers count toward the
64-level limit of the complete RPC request. Caller mutations after submission do not change
the submitted snapshot.

| Command | Arguments | Meaning |
|---------|-----------|---------|
| `move` | `x, y` | Move |
| `stop` | | Stop |
| `attack` | `x, y` or `target_id` | Attack-move or attack target |
| `smart` | `target_id` | Right-click: gather, pick up, follow |
| `harvest` | `target_id` or `x, y` | Gather; target a tree for lumber |
| `build` | `type_id` and `x, y` or `target_id`; optional `auto_place: bool` | Build at/on the target, or search near a point |
| `train`, `research` | `type_id` | Train unit/hero or research upgrade |
| `learn` | `ability_id` | Learn hero skill |
| `cast` | `order` (its name, or its numeric id), optionally `x, y` or `target_id` | Cast by order |
| `use_item` | `slot`, optionally `x, y` or `target_id` | Use inventory slot 0-5 |
| `drop_item` | `slot`, optionally `target_id` or `x, y` | The unit walks over and gives the item to the target unit (a shop buys it for half its price) or drops it at the point; no target drops it at its feet. Sent as a player's drag of the item (the replay's `0x13` record) |
| `select` | | Select unit |
| `buy` | `shop_id, item_type_id` | Buy for `unit_id`; selects buyer/shop automatically |
| `revive` | `target_id` | An altar revives its own dead hero; the hero's `unit_id` comes from its `death` event |

Rejection reasons: `not_your_unit`, `unknown_unit`, `unknown_command`, `bad_arguments`,
`queue_full` (buffer allocation failed), `no_build_site` (native site search failed). Acceptance means queued for delivery; Warcraft can
still refuse an order under its normal resource, prerequisite and production queue rules.

## Queued orders and control groups

`move`, `stop`, `attack`, `smart`, `harvest`, `build`, `cast` and `use_item` accept
`arguments.queued: true` to append an order using Warcraft's Shift behavior. The default
is false. A normal unit order replaces that unit's pending orders, even within the same batch.
Training and research retain the engine's production queue behavior and do not accept
`queued: true`: multiple training orders in one batch fill that production queue in the same step.

```python
actions = [Action(unit_id, "move", {"x": 100, "y": 200}), Action(unit_id, "move", {"x": 500, "y": 200, "queued": True})]
groups = session.groups(0)  # also session.view(0).groups
groups.assign("scouts", unit_ids)
actions = groups.actions("scouts", "move", {"x": 500, "y": 200})
```

Control groups are named Python collections that expand into ordinary actions. They do
not manipulate Warcraft's keyboard hotkey groups. Assignment requires currently observed
own units and preserves order while removing duplicate IDs. `members(name)` temporarily
filters absent units (including loaded workers); their stored IDs remain available if they
reappear. Reset clears all groups, and unknown names raise `KeyError`. Normal ownership,
snapshot and transport limits apply to generated actions.

Executed special-order fixtures in `tests.e2e.test_actions` cover Dreadlord Inferno
(`cast`, `order="dreadlordinferno"`, point target, learned `AUin`) and Sacrifice (the acolyte
casts `requestsacrifice` at a Sacrificial Pit); `tests.e2e.test_placement` covers Entangle
(the Tree of Life casts `entangle` at a gold mine). Tests verify the resulting Infernal, Shade or entangled mine;
normal ability, placement and resource prerequisites still apply.

## Native building placement

`build` with `auto_place: true` treats `x,y` as an anchor. On the game thread, the environment
calls Warcraft's own local placement search for that worker and building type, then queues the
ordinary build order at its selected coordinates. `auto_place` omitted or false retains exact-point
build semantics. This argument is valid only on `build`; it must be a JSON boolean.
`target_id` issues the native build-on-object order (for example, an Acolyte haunting a visible
neutral Gold Mine), matching Blizzard's targeted mine construction path. It cannot be combined
with `auto_place: true`. The target must be visible; the engine decides whether it is suitable.

Successful searches appear in the act reply's `placements` list as `{index, x, y}`. Session step
info groups these lists by player under `placements`; the input actions remain unchanged. A failed
search rejects that action with `no_build_site`. A selected site is not a promise of construction:
resources, prerequisites, subsequent orders and changing game state can still prevent execution.
Observe unit orders and construction events in subsequent steps to determine the outcome.

`x,y` is a hint, not a site: on open ground the chosen site is usually within about 150 of it, but
the solver keeps mining lanes and the area around halls clear, so an anchor there can move several
hundred units.

The engine's search sees only standing objects. The environment therefore also treats as occupied
(using the structure's pathing-texture footprint) the sites of the player's workers' current build
orders and of earlier builds in the same `act` call; when the solver's pick overlaps one, it searches
again from anchors stepped outward (up to three footprint widths) and rejects with `no_build_site` if
none is clear. An unqueued build ignores the same worker's own earlier site, which it replaces. The
engine's order queue is not read, so the environment remembers the Shift-queued build sites it handed
out and keeps them occupied until that worker gets an unqueued order, dies, or 60 game seconds pass. This invokes the native site solver, not Blizzard's economy controller: it does not choose
workers, issue gathering orders or retry failed builds. Use a target build for a Haunted Gold Mine;
mine selection and expansion planning are not supplied by the site solver.

## Native encoding

`wc3hook/act.c` encodes W3G records; delivery and player identities are described in
[design](../design.md#time-and-command-delivery).

Captured from UI actions using `trace 1`. Numbers are hexadecimal unless stated otherwise.
Packets are limited to 1,000 bytes and split at record boundaries.

| Record | Meaning |
|--------|---------|
| `16` mode count (idA idB)* | Select: mode 1 adds, 2 removes |
| `1a`; `19` type idA idB | Pre-subselection; subgroup |
| `10` flags order ff*8 | No target; training/research use the type id as order |
| `10` 42 ability ff*8 | Learn hero skill |
| `10` 60 (852008+slot) itemA itemB | Use inventory slot; 852008 is decimal |
| `12` flags order ff*8 x y tA tB | Point/target order; ff*8 means ground |

Object id pairs are at `+0xc/+0x10`; the RPC uses the first half. Order names map through
`tools/prepare/data/orders.json`. Each player's previous selection is stored as ids and cleared before
selecting another unit, preventing mixed selections from rejecting orders.
Queued unit orders set flag `0x0001`, matching the
[W3G action format](https://github.com/scopatz/w3g/blob/master/w3g_actions.txt).
Live tests verify traversal of both waypoints and cancellation by an ordinary stop.

`buy` selects the buyer, then the shop, and issues the item type as an order. Stock and range
rules still apply. For lumber, target a tree: point harvest does not select one.
