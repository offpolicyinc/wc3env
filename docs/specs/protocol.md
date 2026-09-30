# RPC protocol

One JSON request/reply per line over the instance's named pipe. `RpcClient` serializes round
trips and skips non-JSON diagnostics. Version 1 is shared by the DLL and test fake server.

## RPC

```json
{"protocol_version": 1, "id": 7, "method": "step", "params": {"ms": 250}}
{"protocol_version": 1, "id": 7, "ok": true, "status": "in_game", "result": {"game_time_ms": 1250, "frames": 4, "elapsed_ms": 250, "reason": "target"}}
{"protocol_version": 1, "id": 8, "ok": false, "status": "in_game", "error": "bad_params", "detail": "ms must be a positive multiple of 25"}
```

Requests require `protocol_version`, integer `id`, string `method` and optional object `params`.
Replies echo version/id and include `ok`, `status`, and either an object `result` or `error`
plus human-readable `detail`. Replies preserve request order. A call shares one 120-second deadline across writes and all
reply/diagnostic reads; close cancels pending I/O. JSON numbers must be finite, request lines
must be shorter than 65,535 bytes, and nesting must not exceed 64 levels. Parsed requests
are limited to 4,096 values, including object keys. Strings must contain valid Unicode;
native string arguments cannot contain embedded NUL. Integer fields require integer JSON
notation within signed 64-bit range; `1.0` and `1e0` are not integer fields.
DLL replies also include `timing_ms` outside `result`; see [profiling](#step-profiling).

States: `launched` (awaiting configuration), `in_game` (running), `ended` (results or replay EOF;
final observations readable).

| Method | Params | Valid states | Result |
|--------|--------|--------------|--------|
| `info` | | Any | `exe_hash` (SHA-256), `game_version`, `dll_version`, `mode` (stepping/realtime/null), `game_time_ms`, `instance` (pid), `map` |
| `create_game` | `map`, `players`: nonempty list of `{slot, control, race?}`, `mode`: stepping/realtime | launched, ended | `game_time_ms`, `map`, `players`; enters in_game |
| `reset` | | in_game, ended | `{}`; reloads and holds at 1000 ms; enters launched |
| `save_replay` | `path`: absolute, under 260 bytes | in_game, ended | `{game_time_ms}`; writes the episode's native `.w3g` |
| `step` | `ms`: multiple of 25 in 25..60000; `observe`: optional list of distinct player slots | in_game, stepping only | `game_time_ms`, `frames`, `elapsed_ms`, `reason`; with `observe`, `observations`: `[{player, offset, size}]` [binary observations](observations.md#binary-observations) written after the step |
| `observe` | `player`: slot, default 0; `format`: `json` (default) or `binary` | in_game, ended | [Observation](observations.md), or for `binary` `observations`: `[{player, offset, size}]` |
| `act` | `player`, `actions` | in_game | `rejected`: list of `{index, reason}`; `placements`: resolved `{index, x, y}` for native build searches; others queued |
| `debug` | `op`, `args`: object | in_game | Op-specific; see Debug below |
| `quit` | | Any | `{}`; process exits after replying |

`GameSession.step(actions, ms=None)` and `WC3Env.step(actions, seconds=None)` advance by
`config.step_ms` unless a length is given. Uninterrupted steps advance exactly the requested time. Results, replay EOF or stalls can end them early;
`reason` is `target`, `game_over`, `replay_end` or `stalled`. `GameSession` faults on stalls or unexpected
short steps; reset replaces the suspect process. Successful step info includes `elapsed_ms`, `step_reason`, and per-player `placements` lists (empty when no sites were resolved). Queuing an action guarantees neither execution nor next-turn timing.

`reset` and `create_game` from `ended` reload offline games in the same process. Orders, events,
results and staging clear; speed, wait floor and render settings persist. Failed reloads return
`bad_status`; `GameSession` closes the failed process. Its `reset()` also configures and observes.
By default a session replaces the process before episode 33 to bound engine retention.
`GameConfig.max_episodes_per_process` accepts a positive integer or `None` to disable that
limit. Replacement preserves the match seed and successful `RpcClient.debug` speed,
wait-floor and rendering settings; groups and episode observations always clear.
The raw `reset` RPC remains an in-process reload without this host policy.

`save_replay` runs the engine's own save: it closes the recorder and moves `Replay/TempReplay.w3g` to the
path, as leaving a game does for `LastReplay.w3g`. Recording ends there, so save once, when the episode is
over; a second call in the same episode, a replay playback or the fake server answer `bad_status`. A reset
starts a new recording. `debug` staging is not recorded, so a replay of an episode staged with `debug` will not
match it. `GameSession.save_replay` also writes the match setup, AI difficulty and AI slots the game started
with to `<name>.w3g.json`; launching the replay applies them, so a `MatchSetup` or `ai_agents` episode replays.

`.w3g` playback requires stepping, the installed map and the original agent-slot configuration.
Steps stop at the recorded duration; final observations remain readable, with empty results
if the recording ends without an outcome. `act` is rejected. Session reset relaunches the replay;
raw RPC reset rejects it. Match-setup/AI overrides are rejected (they come from `<name>.w3g.json`); debug state
changes are not recorded.
Unexpected simulation teardown returns `bad_status`, never wrapped elapsed time.

Configuration: map must match the launch path or filename; slots must be unique integers 0-15
and active. `control` is `agent` or `computer`; agent slots must match launch configuration
and all agents require offline mode. Startup AI is suppressed without changing slot types.
Omit race to retain it; named races (`human`, `orc`, `undead`, `night_elf`, `random`) must match
loaded setup by default. Opt-in [match setup](configuration.md#offline-match-setup) applies races, controllers,
map-defined slots and seeded starts before startup; `match_setup` in `info` and
`create_game` records the effective seed. `exe_hash` is the launch-verified executable
SHA-256. See [compatibility](../design.md#compatibility).

| Error | Meaning |
|-------|---------|
| `bad_version` | Protocol version missing, invalid or not 1 |
| `bad_request` | Non-object JSON or invalid/missing id or method |
| `unknown_method` | Unrecognized method |
| `bad_params` | Invalid, missing or out-of-range parameter; includes acting for a non-agent |
| `unsupported_config` | Requested map/player setup cannot be supplied |
| `bad_status` | Invalid lifecycle state or failed game-thread/reload operation |
| `not_stepping` | Step requested in realtime mode |

Payload contracts: [observations](observations.md) and [actions](actions.md).

## Debug

Test/staging API, separate from agent actions. Invalid arguments return `bad_params`.

| Op | Args | Effect/result |
|----|------|---------------|
| `speed` | finite `factor` >0 and <=2048 | Clock multiplier |
| `waitfloor` | integer `ms` in 0..1000 | Minimum real duration of scaled waits |
| `render` | `on` | 0 skips drawing and presentation once the match is held |
| `spawn` | `type_id, player, x, y, n=1, columns=4, spacing=64` | Grid with 1..500 units, 1..500 columns and spacing 0..10000; returns `unit_ids` |
| `kill`, `remove` | `unit_id` | Kill or remove unit |
| `level` | `unit_id, level` | Set hero level |
| `give` | `unit_id, type_id` | Give item |
| `hp`, `mana` | `unit_id, value` | Set value; hp raises max if needed |
| `item` | `type_id, x, y` | Create ground item; returns `item_id` |
| `resources` | `player, gold, lumber` | Set resources |
| `ai` | `player, paused` | Pause computer AI |
| `order` | `unit_id` | Returns current executed `order_id` |
| `invulnerable` | `player, on` | Apply to all own units now and after each step |
| `alliance` | `player, other, kind` (0..9), `enabled` | Set alliance flag |
| `destructable` | `type_id, x, y, invulnerable=0` | Create destructable; returns `id` |

## Step profiling

`GameSession.step()` and `WC3Env.step()` include two timing fields in `info`:

- `timings_ms`: host wall time for validation, action submission (`act`), simulation
  advancement (`step`), all observations (`observe`), and the complete call (`total`).
  Waiting to acquire the session lock is excluded. Phase RPC times include transport.
  The `step` phase is zero in realtime mode; the simulation runs concurrently there.
- `rpc_timings_ms`: records for each RPC in that step, with its method, player when
  applicable, and the measurements described below.

Standalone `RpcClient.call()` and its convenience methods expose `last_timing_ms`:

| Field | Measurement |
|---|---|
| `total` | Host time for JSON encoding, request/reply, and JSON decoding; excludes lock wait |
| `server` | DLL time from request parsing through reply construction, before sending |
| `game_thread` | Time spent executing that request's game-thread jobs, including observation construction |
| `overhead` | `max(0, total - server)`: host, transport and scheduling overhead combined |

The DLL uses its original unscaled performance counter. Server time includes waits
for game-thread scheduling and simulation advancement; game-thread job time does
not measure every simulation update or CPU utilization. These fields overlap and
must not be added as independent costs. Overhead is a residual, not a pure pipe
latency measurement, and may clamp tiny clock/rounding differences to zero.

Older servers and the fake server omit native timing: `server`, `game_thread`, and
`overhead` are null, while host `total` is still measured. Transport failures also
leave unavailable fields null. Raw conformance calls clear the previous timing.
Native reply `timing_ms` lives outside `result`, so observations remain gameplay
data without performance-dependent fields.

The scenario runner additionally totals agent decision time and the session/RPC
phases in its JSON report. Agent time excludes staging and simulation. Use these
measurements to identify a bottleneck before changing step sizes or pool size.

## Difficulty and local overlays

These helpers are part of `wc3env.rpc.RpcClient` and the standard hook.
They use the existing serialized, game-thread `debug` dispatcher and require
`in_game` status.

```python
difficulty = game.rpc.ai_difficulty(1)
game.rpc.overlay("An application-supplied status", x=-0.30, y=0.60, seconds=2)
```

`ai_difficulty` returns `{"ai_difficulty": 0|1|2}` for easy, normal or insane.
The low-level process launcher accepts `ai_difficulty=None|0|1|2`; the default
preserves map settings. An explicit value configures unclaimed melee-AI players
before their startup script and its native controller difficulty. The launch
setting survives map reset. Agent slots still have their startup AI suppressed,
except slots listed in `GameConfig.ai_agents` (launcher: `ai_agents`). Those need
`GameConfig.setup`: they load as computer slots, their melee AI plays them, and
`step()` orders act alongside it. `GameConfig.computer_agents` loads agent slots
as computer slots without an AI. Warcraft has a computer player's units use their
spells on their own (Heal, Slow, Storm Bolt), which a local player's units never
do, so identical armies fight unequally unless both slots are computer slots.
The AI's group controller reissues its own
orders within about three seconds, so an order that must hold is resent after
the AI changes it.
Native logs include both configuration readback and the original AI getter.

`overlay` displays local UI text and clears the previous text panel. It does not
send chat. The caller chooses its layout; the environment has no required number
of rows or agent labels. Text is limited to 4,096 UTF-8 bytes without embedded NUL.
Coordinates must be finite within -1..1, and duration within 0..3,600 seconds.

`wc3env.game.launch` also accepts `background_visible=True` with
`window_mode="background"`. This shows the otherwise offscreen window while
retaining input isolation. It defaults to false. These options are passed to the
child process only, so one launch cannot change another's configuration.

Wire operations are `debug` with `op` equal to `ai_difficulty` or `overlay`, and the
helper's arguments in `args`. Source lives in `wc3hook/difficulty.c` and `wc3hook/overlay.c`.
