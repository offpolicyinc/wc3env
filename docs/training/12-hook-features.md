# 12. Features and masks: one implementation, in C

The model's inputs (unit rows, the remembered tree grid, scalars, the last decisions) and the legality masks are
computed by one implementation, `wc3hook/features.c` and `wc3hook/masks.c`, used two ways:

```
                       wc3hook/features.c + masks.c
                       (tables from reference.json: tools/feature_tables.py)
                          │                                  │
   compiled into the game hook (32-bit)          compiled as a library (64-bit, Windows or Linux)
   live play: self-play actors, evaluation,      recorded observations: data set conversion, replay
   play against the AI                           checks, benchmarks (training/policy/native.py)
   the block arrives with each observation       Features(static).step(obs, last_action)
```

Both give the same dict (`wc3env.binary.parse_features`). Data set conversion uses the library because a replay's
label is only known from the next observation's orders: it calls the same C in the same order the hook does.

## Using it

Live (the hook):

```python
config = GameConfig(..., observation="binary", features={slot: native.hook_spec(static) for each observed slot})
block = obs.features                                   # after reset or step
session.step(actions, decisions={slot: native.hook_decision(action)})   # non-noop actions only
# VectorSession: session.send(i, actions, decisions)
```

`hook_spec` gives the map's bounds, both races and its starting tree grid (`rpc.features`); a new game drops it and
the session sends it again. A decision is folded into the player's next features, as a noop is not.

Recorded observations (the library):

```python
state = native.Features(static)  # one per player per game
block = state.step(obs, last_action)  # in order; last_action None for a noop or the first step
masks = native.chosen_masks(block, kind, argument)  # the masks a choice was made under, as the data set stores them
```

The library is built on first use when missing or older than its sources (`training/native/build_features.bat` on
Windows, `cc` on Linux: images that convert data or run benchmarks need a C compiler). Its C keeps scratch in static
buffers, so `Features.step` takes a lock: one call at a time per process.

## The block

Written after the player's observation in the hook's shared memory (the step reply gives `features_offset` and
`features_size`); the observation itself, its header and version are unchanged. A header (magic `WFT1`, version,
size, row count, scalar count, the number of argument rows and of actor rows, the last 4 decisions' types and point
cells, the type mask), then, each 4-byte aligned:

| Array | Shape | |
|---|---|---|
| `units` | rows x 69 float | features.COLUMNS |
| `unit_type`, `unit_owner`, `unit_order`, `unit_queue` | rows int | embedding ids |
| `unit_inventory`, `unit_buffs`, `unit_abilities` | rows x 6, 8, 5 int | |
| `unit_id`, `unit_cell` | rows | which game object; its map cell |
| `trees` | 128 x 128 bytes | the remembered tree grid (at most 4 a cell) |
| `scalars` | 1,216 float | resources, food, time, races, research, unit counts |
| `argument_types`, `argument_bits` | k, k x 354 bytes | the argument mask of each allowed type that takes one |
| `actor_keys`, `actor_rows` | K, K x 32 bytes | the actor mask of each (type, argument) with legal actors |

~37 KB a player-step, ~5 KB compressed; computing and writing it ~0.1 ms in the game process.

## Tests

- `training/tests/test_native.py` (`WC3_E2E=1`): in a live game, the hook's blocks equal the library's.
- `training/tests/test_inference.py`: the inference process's choices on these features score the same in the
  learner.
- Replay checks (`python -m training.replays.check`): a pro's choice must never be masked out.

The C was first checked equal to the Python it replaced, bit for bit, on 35,218 player-steps (recorded games, 3 pro
replays with every action type legal somewhere, live games through the hook).

## Changing the features or masks

1. Change the C (and `features.COLUMNS` / sizes in `features.py`, `binary.py`'s block layout if the block changes).
2. `python tools/feature_tables.py` if reference data or id orders changed.
3. `wc3hook\build.bat` (~17 s; the DLL ships in each run's code tarball, no image rebuild).
4. `python -m training.tests.test_native --record` (new digests: the change is now the accepted output).
5. Convert the data sets again (`python -m training.replays.dataset ...`): files from before the change hold the old
   features.
