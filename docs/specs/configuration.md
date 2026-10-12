# Configuration

Match setup and pools share `GameConfig`.

Games start with sound on. `WC3_SOUND=0` makes automation silent; the real-game test
harness sets it automatically for tests, scenarios and replay checks. An explicit
`GameConfig(sound=True/False)` or `game.launch(sound=True/False)` overrides that
default for one launch. This leaves Windows volume and saved Warcraft preferences unchanged.

## Offline match setup

`GameConfig.setup` opts into configuring the match before starting units and AI
are created. Without it, `PlayerConfig` continues to validate the loaded setup.

```python
from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

config = GameConfig(
    map="(4)TwistedMeadows.w3x",
    players=(
        PlayerConfig(0, "human"),
        PlayerConfig(1, "orc"),
        PlayerConfig(2, "undead"),
        PlayerConfig(3, "night_elf", "computer"),
    ),
    setup=MatchSetup(seed=42, randomize_starts=True),
    render=False,
)
with GameSession(config) as session:
    observations = session.reset()
    observations, done, info = session.step({0: [], 1: [], 2: []})
```

With setup enabled, named races select the initial race. `random` resolves to a
seeded choice among the four melee races; an omitted race uses the engine's
choice. `control="computer"` starts the built-in AI, including on the default
local-player slot. `agent` suppresses startup AI while retaining a valid offline
slot type. Normal melee startup can still issue initial worker gather orders.

Configured slots must have map-defined starting locations. Additional defined
slots can be activated as computer slots and claimed by agents; slots absent
from the map fail with `unsupported_config`. Unlisted slots retain their loaded
state. The lobby may move the slots it knows off their scripted start locations; an activated
slot whose location is then occupied takes the lowest free map-defined one, so every player
starts alone. This does not create human network peers or enable LAN/Battle.net play.

`randomize_starts=True` applies a seeded permutation to the map's defined starting
coordinates. It randomizes the available locations, including ones for unlisted
slots, and does not preserve team adjacency. The map's priorities and other
custom logic still apply. Maps that define locations after assigning players,
omit standard configuration, or use unsupported location indices fail explicitly.

`seed` accepts integers 0..2147483647. If omitted, a fresh seed is selected for
each session (or each process launched directly by a pool). `info()` and `create_game()` return the effective seed and start
option in `match_setup` (null without setup). Record that effective seed to replay
a run. Reset reuses the same setup and seed, including process recycling and recovery;
use a new session/config to change it.
Pools pass setup independently to each process, so a shared explicit seed repeats
across copies while omitted seeds are selected independently.

The pool CLI accepts `--configure-match`, `--race SLOT:RACE` (repeatable), `--seed`,
and `--randomize-starts`. Any race, seed or start option enables setup; use
`--configure-match` alone to apply controller changes. Race slots must appear in
the pool's configured agent/computer list.

The DLL seeds the engine during configuration and again at the initial 1000 ms
hold. Race selection and start permutation use separate deterministic generators.
Live tests cover repeated initial races/positions and an executed combat damage
trace across resets. This is evidence for those cases, not a promise of full-game
determinism across custom scripts, executable versions, or realtime scheduling.
Custom scripts can subsequently reseed the engine or issue their own orders.

Implementation is tied to the verified executable: it applies races after the
lobby copies/resolves preferences, before melee startup. It never changes a race
after starting units exist. Setup remains subject to the launch executable hash
check. `tests.e2e.test_match_setup` checks all four races, computer AI execution, four
independent players, seed repeats, start variation, and unsupported-slot rejection.

## Pool configuration

Implemented. `StepPool` advances independent Warcraft processes concurrently. Each game
uses the same `GameConfig` as `GameSession` and may contain one or more agents.

### API

```python
StepPool.launch(n, speed, render=None, offscreen=False, spin=False, *, config=None)
StepPool.launch_configs(configs, speed, *, offscreen=False, spin=False)
```

`launch()` repeats one configuration and delegates to `launch_configs()`. The latter takes
a nonempty sequence and snapshots it as `pool.configs`, preserving order across games,
statistics and results, including failed instances.

```python
from wc3env.pool import StepPool
from wc3env.session import GameConfig, PlayerConfig

configs = [
    GameConfig(map="(2)EchoIsles.w3x", step_ms=250, render=False),
    GameConfig(map="(2)SecretValley.w3x", players=(PlayerConfig(0), PlayerConfig(1)), step_ms=250, render=False),
]
pool = StepPool.launch_configs(configs, speed=64)
try:
    for i, stats in enumerate(pool.stats):
        if stats.error:
            print(i, stats.error)
    results = pool.step_all()
    print(pool.run(steps=200).text())
finally:
    pool.close()
```

For repeated copies: `StepPool.launch(8, speed=64, config=configs[0])`.

### Rules

| Setting | Contract |
|---|---|
| `map`, `players` | Resolve each map before launch; forward its agent slots to launch and its complete player list to `create_game` |
| `render`, `window_mode` | Per game; [rendering constraints](../design.md#rendering-and-input) apply |
| `setup` | Per-game [races, controllers, seed and starting locations](#offline-match-setup); opt in with `MatchSetup` |
| `max_episodes_per_process` | Session reset policy; pools do not reset episodes or recycle healthy instances automatically |
| `mode` | Every configuration must use `stepping` |
| `step_ms` | All games agree; used when `step_all()` or `run()` omits `ms` |
| Explicit `ms` | Overrides one call without changing configurations; validate before any game advances |
| `speed`, `spin` | Pool-wide tuning, applied after each map loads |
| `offscreen` | Optional placement; does not change input policy |

Existing `launch(2, speed=64, render=False)` calls retain their defaults. Supplying both
`config` and a legacy `render` value is an error; put rendering in the configuration.
`StepPool(games)` accepts already configured games, exposes `configs=None`, and defaults
to 1000 ms when `ms` is omitted. A round requests equal simulation time, not synchronized
wall time. Agent decisions use each game's RPC; submit actions before stepping.

### CLI

```powershell
python -m wc3env --instances 4 --map "(2)SecretValley.w3x" --agent-slot 0 --agent-slot 1 --stepms 250 --speed 64
```

The CLI shares one setup across games. Repeat `--agent-slot` and `--computer-slot` for the
complete requested player list; without either, use agent 0 and computer 1. Duplicate
slots and lists without an agent are rejected. `--window-mode` accepts `background` or
`interactive`; defaults are 250 ms steps and presentation off unless `--render` is supplied.

### Failure behavior

Validate every configuration, map and tuning value before launching. The engine then checks
loaded slots, controllers and races; unlisted active players retain their setup. Configuration
does not create arbitrary lobbies. Opt-in match setup can override races/controllers and
activate slots with map-defined starting locations.

A process-launch failure closes all earlier launches and raises. After launch, setup/tuning
failures retire only that instance; healthy games continue. Step failures and stalled replies
also close the failed game. `step_all()` returns its exception that round, then `None` in
later rounds. Statistics count actual advancement; always close the pool when finished.

Coverage: `python -m unittest tests.e2e.test_pool` checks routing,
map compatibility, cleanup and stable failure ordering.
