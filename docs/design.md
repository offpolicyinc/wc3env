# Design

Warcraft owns the simulation. Python owns agents and processes; an injected DLL exposes
game state, commands and time control. [Architecture](architecture.md) has the overview
and diagrams. [Roadmap](roadmap.md) owns unfinished work.
Contracts: [RPC](specs/protocol.md), [observations](specs/observations.md),
[actions](specs/actions.md), [match setup and pools](specs/configuration.md).

## Compatibility

Supported: offline Warcraft III on Windows x64 with 64-bit Python 3.11+.
Battle.net's **1.29.2.9232-legacy-tft** package contains the x86 **1.29.2.9231** executable.
The launcher verifies this SHA-256 before injection:

```text
3f2ed0120d80578bf07e4423296dade1adfb959d59a2d20a7584224559570eed
```

This pairing was checked on 2026-09-17 against local `.build.info`, the executable and
[Blizzard's metadata](https://us.version.battle.net/v2/products/w3-legacy-tft/versions).
See the [Legacy announcement](https://us.forums.blizzard.com/en/warcraft3/t/warcraft-iii-legacy-the-frozen-throne-129-now-available/38037)
and [9232 patch notes](https://us.forums.blizzard.com/en/warcraft3/t/warcraft-iii-legacy-the-frozen-throne-129-now-available/38037/4).
Other binaries need a verified port of the offsets and calling conventions in
[`wc3hook.h`](../wc3hook/wc3hook.h), not a changed version label.

Maps accept absolute paths, install-relative paths or unique filenames under `Maps/`.
The loaded map establishes active slots, controllers, races and starting state;
configuration validates those facts by default. Opt-in `MatchSetup` applies race/controller
choices, activates map-defined slots and seeds starting locations before melee startup.
Paths must fit the engine's 260-character buffers,
including generated replay paths. Map paths must round-trip through the Windows code page
because Warcraft reads `-loadfile` as ANSI. DLL injection uses UTF-16.

## Ownership and lifecycle

```text
agent -> WC3Env -> GameSession -> RpcClient -> named pipe -> wc3hook.dll -> Warcraft III
StepPool -> independent games, each with its own process, pipe and simulation clock
```

- `GameSession` owns launch, reset, close and one clock. It validates every agent's batch,
  advances once and observes each player. `PlayerView` reads cached observations;
  `WC3Env` wraps a session for one agent.
- RPC round trips are serialized. Game-object reads and mutations run on the game thread,
  during updates or the frame pacing wait. Between frames, jobs temporarily install cached
  component TLS; arbitrary wait callbacks are unsafe.
- A short-held ownership lock lets close detach and cancel a process during a blocked
  round trip. A concurrent launch owns its result until publication and disposes it if
  close wins. Retaining the process handle prevents PID-reuse mistakes.
- Each launch gets unique writable data for logs, replays and custom files. Hooks isolate
  named mutexes/events and redirect Documents before the main thread resumes.
- `StepPool` accepts one or several `GameConfig` values through one startup path. It
  preserves instance order, closes failed games and measures actual simulated time.

The game holds at **1000 ms** before `create_game` validates setup and selects stepping or
realtime. States are `launched -> in_game -> ended`; final observations remain readable.
Each player's result is retained; the session ends when all agents have results.

Reset calls `RestartGame(false)` (`0x0a5620`) to reload the same map in-process. Unregister
added agent identities first: retaining them stalls teardown. Check membership before
unregistering. Clear actions, selections, events, results, staging and game/TLS pointers;
preserve hooks, pipe, speed, wait floor and rendering settings. Ignore teardown results and
register identities at the new hold. Failed reloads or suspect connections need a fresh process.
`GameSession` also recycles before episode 33 by default;
configure `max_episodes_per_process` to change this limit. Seed and RPC tuning survive the
replacement. Raw RPC resets remain in-process. See the [measured matrix](compatibility.md).

Stepping replay playback stops at the recorded duration and preserves final observations;
EOF does not invent player results. Actions come from the recording. Session reset relaunches
the replay: `RestartGame` is a map reload, not a replay restart.

Moved native reallocations use the current matching arena (`allocator.c`), following normal
`SMemAlloc` selection. This prevents old full arenas from repeatedly reserving new ones.
On a failed free-space search, check the remaining size bins and merge adjacent free blocks
before growing. Native allocation, copying, freeing and group locks remain intact. UI teardown
unregisters its two forgotten input handlers; at the new hold, native rehashing reclaims retired
connection subscriptions after dispatch unwinds. Live text caches retain their buffers.

## Time and command delivery

QPC, GetTickCount, FILETIME and the `rdtsc` helper (`0x396c80`) share a virtual clock.
Seqlock readers avoid blocking timer calls inside other libraries. Integer conversions
divide before multiplying to avoid overflow. Speed scales finite waits; frozen waits keep
a 1 ms floor. RPC deadlines use real time. The hook requests 1 ms timer resolution and opts out
of Windows power throttling: at the default 15.6 ms tick, the frozen game thread and the turn
producer each woke up to one tick late, which made a 250 ms step cost 15-25 ms of waiting for
about 1 ms of simulation. A step or a posted job also sets a wake event that the frame loop's
pacing wait includes, so the game thread starts at once.

`GameUpdate` (`0x1aefd0`) consumes 25 ms turns. Stepping writes pending time and passes zero
elapsed time to bypass the lag clamp. Publish the producer budget after actions flush.
The offline host (`0x557be0`) emits that budget, at most 400 ms per packet; account time
before transmission and reset the total with its native counter. Warcraft retains sequencing,
delivery and acknowledgements. Realtime uses native scheduling. Receiver-side credit trimming
caused stalls; 1x settling only hid queued backlog.

A step completes after `GameUpdate` unwinds. The stall guard measures five seconds without
progress, not total duration. Replies report elapsed time and completion reason; sessions
reject unexplained short steps, and pools retire stalled games.

Actions use Warcraft's command parser. Offline agents send zero-time packets through
`0x1aab40`, splitting at record boundaries. `QueueLocalAction` (`0x1ae190`) stamps the local
wire identity and remains the UI path. Register missing identities through the native table
(`0x1a5820`, `0x1b1990`); suppress startup AI without changing human/computer slot types.
Selection history stores IDs, not pointers, and clears before another unit is selected.
Acceptance means queued; Warcraft still applies range, resource and ability rules.

## Observations and events

Enumerate objects (`0x3f4ba0`) without allocating JASS handles; live numbers (abilities, armor,
damage, hero stats) come from natives on each listed unit's handle, created once per unit and reused
(no memory growth over eight 20-minute episodes). RPC IDs use the object's
`+0xc` ID; internal references also need its `+0x10` salt. Lookup has no fixed object-count
cap. Exclude hidden, loaded and Locust units and apply each observer's fog. Old map scripts
map neutral JASS IDs 12–15 to internal slots 24–27.

Observing must never create or remove game objects, not even a temporary one: the engine reuses
object IDs, so one extra object shifts every later ID, and a replay's recorded orders name units by
ID (a ladder replay then diverges within seconds: its players' orders miss their units). Handles
for existing units are harmless.

Events come from detours on the per-kind fire functions (`0x0b95f0`..`0x0bcad0`, one per kind),
read from their arguments at entry. Nothing is registered on the game's behalf: a registered kind
makes the engine build an event object, an agent that shifts later object IDs. Spell effects are
read one level up (`0x28ccd0`, the casting unit and the ability), since the fire function there
runs only when a trigger is registered. Snapshot values and visibility
before objects change; retain no game pointers. Death ownership precedes defeat transfers;
summon's trigger unit is the summoned unit. Each observer has a bounded visible-event
history: hidden activity affects neither retention nor loss counts. Never infer death from
disappearance. Agents select resources from visible destructables and type IDs, without a
first-base or nearest-tree shortcut.

Orders come from two places. A player's command (network, replay or `act`) gives each unit its
order through `0x2c9360`, whatever the engine then does with it (issue, Shift-queue, issue later,
refuse): its detour records the command as given (origin `player`). The unit-side functions that
fire the issued-order events (`0x28b8b0`, `0x28be70`, `0x28d270`) see every order a unit actually
gets; the call chain (the exe keeps frame pointers) says where it came from: inside `0x2c9360`, the
command already recorded; inside the JASS interpreter (`0x4d75b0`), a script; else the engine
itself. A unit's orders are its owner's only. Playing a ladder replay, 97-99% of its order actions
come back as player commands within a turn; the rest reached units that had just died or were
refused, so a replay whose commands stop reaching units has lost sync (`tools/check_replays.py`).

Player categories, alliances and own scores come from runtime natives. Resource labels use
loaded object data, including custom tree types. World bounds cache four numbers per episode,
copied by the routine `GetWorldBounds` uses (`0x3036f0`), without its rectangle. See [metadata](specs/observations.md#observation-metadata).

## Rendering and input

| Window mode | Behavior |
|---|---|
| `background` (default) | Offscreen, absent from taskbar/Alt-Tab; never requests focus or captures the mouse; all input swallowed |
| `background` + `background_visible` | A normal window that never takes focus or the cursor on its own and never pauses; click it to control the game, click away and it keeps running |
| `interactive` | Normal Warcraft input and focus behavior, including pausing when the window loses focus; use realtime for human play |

Window policy is fixed per process and survives resets. Background mode requires a window;
`launch(window=False, window_mode="interactive")` allows fullscreen. `game.resize()` and
`game.offscreen()` reposition without requesting focus.
Resize requests are asynchronous and can be clamped by Warcraft's minimum window size.

`render=False` skips the frame's paint handler (`0x559330`) once the match is held, so no
scene, UI or text is drawn; that drawing cost most of the game thread even without
presentation. The simulation never depends on drawing: a minimized multiplayer client takes
the handler's own no-draw path and stays in lockstep, and observations match render-on runs
step for step. `GxPresent` still runs, at most 30 times a real second, for backend cleanup and
FPU control restoration; its **Direct3D 9 Present** is suppressed and GPU work is completed
through an event query. Map loading still paints: the match does not start its clock until
the loading screen has drawn. Leaving GPU work pending retained buffers; skipping cleanup
retained textures.
A window and graphics device are still required. `debug("render", on=0)` uses the same
hook. Other backends must keep `render=True`; requesting render-off exits
with `render off requires the Direct3D 9 backend` in the hook log.

## Build and validation

Real-game tests skip when no supported installation is configured; CI runs only
host and native tests. The MSVC build produces the x86 hook, checks required hook installation, runs native
regressions and copies the DLL into the Python package. yyjson owns strict JSON parsing;
checked accessors enforce native ranges and string capacities. Vendor sources stay pinned
and unmodified. Wheels include the DLL and project/vendor licenses, never Warcraft assets.

```powershell
wc3hook/build.bat
python -m unittest discover -s tests/unit -t .
python -m unittest discover -s tests/e2e -t .
python -m tests.e2e
python -m pip wheel --no-deps . --wheel-dir dist
python tools/check_wheel.py dist
```

Host/native CI needs no game; the wheel check runs outside the checkout. Live tests cover
commands, fog/events, results, resets, timing and pool isolation on supported stock maps.
The fake checks contracts and lifecycle, not combat or native delivery. A complete seeded
melee match repeats across reload and process replacement; a recorded two-agent movement
fixture matches playback. The [compatibility matrix](compatibility.md) bounds this evidence.
Run artifacts stay ignored; deliberate regression fixtures are tracked. Pending work belongs in the roadmap.

Native tools: `tools/disasm.py`, `imports.py`, `natives.py`.
Pipe probes: `where`, `profile`, `watch`, `scan`, `dump`, `trace`, `pktlog`, `steplog`
(timeline of the next n steps: request, turn production and delivery, finish; in the hook log).
`tools/bench.py` and `tools/profile_game.py` measure rollout throughput, memory and where the
game spends CPU, on Windows and under Wine.
Native calls use 32-bit argument slots, float pointers for real arguments and bits for real
returns. Staging may allocate VM handles (`0x077710`, `0x4e72c0`); object enumeration itself must not.
MPQ extraction needs StormLib; see [tools](../tools/README.md).

## Agent integration and Linux workers

[`wc3agent/`](../wc3agent/README.md) is a separate Python project. Its current brief-driven
Jev controller owns context formation, memory, tactics and scheduling. Its Warcraft
policy consumes prepared reference data and supplied observations;
model transport lives in `models/jev.py`. The environment
imports no agent code. Policy evaluation is separate from engine validation.

Combat menus expose candidate actions and geometric movement choices without health-based tactical gates.
Micro receives targets, recent HP changes, individual allies and estimated summon timing. Explicit
delegation determines which units it controls throughout combat, movement, scouting and recovery.
Direct macro orders take control back and invalidate older micro replies, except a cast or item
use, which keeps the unit delegated while micro waits for it to go off. Summons join their summoner's group. Loot collection and spell
interruption are controller choices; the dispatcher does not add corrective orders or a gameplay
legality model. Native acceptance and later observed execution remain distinct.

Agent build commands use Warcraft's native site search through `wc3hook/placement.c`, without
starting an AI controller. The selected point becomes a normal recorded build order; `step()`
returns its coordinates, and later observations confirm execution. The engine owns site selection and pathing;
the hook only rejects picks that overlap sites already promised to pending build orders. See [action semantics](specs/actions.md#native-building-placement).

The shared hook provides configurable melee-AI difficulty and optional local overlays; see
[difficulty and overlays](specs/protocol.md#difficulty-and-local-overlays).
[`tools/prepare/`](../tools/README.md) extracts a reusable JSON reference and builds
scenario maps. `tools/prepare/gamedata.py` and `mpq.py` read the archives;
`STORMLIB_DLL` selects its archive library. Its [runner](../wc3agent/README.md#running) validates
the prepared artifacts, launches the standard environment DLL, supplies inputs to the
agent and records sessions. Runtime agent and
runner code require neither StormLib nor preparation modules.

[`docker/`](../docker/README.md) packages the same environment under Wine on Linux.
Its `environment` target contains the game, Wine, Windows Python and `wc3env`;
the optional `agent` target adds the policy, runner and prepared data. Builders
generate the hook and prepared inputs on Windows, then `docker/prepare.py` combines
them with allowed files from their own installation. No game files, prepared data
or images are distributed. Activation files stay outside the image and are
supplied through a read-only `/run/wc3-license` mount; retained output goes to a
mounted `/sessions` directory.

The host must support Wine's x86 syscall entry. `docker/platform_probe.py` checks
that before Wine starts, and `docker/smoke.py` verifies real engine control.
