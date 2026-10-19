# What we take from warcraftsim

[RobinKa/warcraftsim](https://github.com/RobinKa/warcraftsim) trains on the same game build as us (1.29.2.9231 under
Wine). As of 2026-10-18 (commit e3f9f97) it wins ~85-89% of its games against the insane AI on Echo Isles, on one desktop
(Ryzen 5950X, RTX 3090) at ~1,600-2,000 agent steps/s. Gains quoted as "his" come from his
[optimization log](https://github.com/RobinKa/warcraftsim/blob/master/docs/reference/optimization-log.md) and
[experiments](https://github.com/RobinKa/warcraftsim/blob/master/docs/experiments.md). Only what he does and we did not
is listed. Every speed change is measured here before and after (`training/bench/local_games.py`: the same seeded games
under each variant in alternating rounds, CPU per game-step of the games' own processes, and every step's state hashed
against the baseline, so a change that alters the game shows).

## Optimizations (speed)

### 1. Game-code cuts in the hook

**What:** faster versions of hot game functions that give the same results, ported from his `shim/cuts.c` into
`wc3hook/cuts.c`; the same executable, so his addresses are ours. `GameConfig.game_cuts` (bits, `WC3HOOK_CUTS`):

| Bit | What it does |
|---|---|
| 1 | Remembers the map-version answer behind `GetPlayerId` (the game copied 14 KB of map info to read one field, every call) |
| 8 | Skips decoding texture pixels once the match is held with render off (nothing is drawn) |
| 64 | A* path search: each cell's "is it free" check remembered for one search (90% were repeats) |
| 128 | Two soft-float helpers as single CPU instructions (`lzcnt`, `sqrt`), each checked in-process against the original first |
| 1024 | A* computes a neighbour's heuristic only when it is used |

**Status:** done, measured 2026-10-18 (`runs/bench/hook-cuts.json`: 32 games of Warcraft's AI against itself on Echo
Isles, 10 game minutes, 3 alternating rounds). `game_cuts=1225` (bits 1, 8, 64, 128, 1024) took **4.1% less CPU per game-step
early in the games and 4.6% late, 7% less wall time per step**, and every step's state was identical to the baseline
(230,400 of 230,400). Smaller than his 16-28%: his figure is the game's main thread alone (ours counts every process of
the games: Wine, wineserver, the Windows Python workers), and part of his came from bits we have no use for (16, 32: his
JASS harness; 256: below). The hook built here with MSVC (msvc-wine: `wc3hook/build.sh`) costs what the handoff DLL
did with the cuts off (5.39 against 5.41 ms early). **Shipped:** the actors play with `--game-cuts 1225` by default
(`--game-cuts 0` turns it off). The final build checked again against the handoff DLL: identical at every step (38,400 of
38,400), 3% less CPU over 5 game minutes. The repo's real-game tests (`tests/e2e`) pass with it but for
`test_replay`'s stepping-mode check, which fails the same way with the handoff DLL (not from these changes).

### 2. Paint every N steps instead of never

**What:** with render off our hook never runs the game's paint handler once the match is held. The paint also rebuilds a
unit list the game's fog-of-war pass walks every turn; never rebuilt, he saw that pass grow to 17% of the game's main
thread by 350 s of game time. Bit 256 paints once every `WC3HOOK_PAINT_EVERY` steps (default 100: 25 s of game time).

**Status:** measured, left off. Over 10 game minutes painting every 100 steps cost 3% more CPU than never (5.56 against
5.41 ms early, 7.13 against 7.06 late), with the same states at every step; CPU per step rose ~30% from early to late in
every variant alike (armies grow), so no fog-pass growth showed by 600 s, where his showed by 350 s.

### 3. Wine with fsync (GE-Proton 10-34)

**What:** Proton's Wine keeps Windows events, mutexes and waits in shared memory with Linux futexes instead of a round
trip to `wineserver` each (his -28% CPU a step). Before this nothing gave our games fsync: the image sets `WINEFSYNC=1`,
but WineHQ stable ignores it.

**Status:** **the home default** (2026-10-19): `~/.config/wc3env/wine-env.sh` sets `WC3_WINE` (GE's `files/bin`, which
`wine_workers` uses), `WINEFSYNC=1` and `WINEPREFIX=~/.wine-wc3-ge`. Modal's image stays on WineHQ stable (GE untested
there; Proton 11's games died at map reloads on Modal, [8-decisions.md](8-decisions.md)). Measured 2026-10-18, `runs/bench/fsync.json`, 32 games x 10 game minutes x 2
alternating rounds, then 3 map reloads of every game, all with the cuts:

| | CPU per game-step, early / late | Wall per step | Steps identical | Failed reloads |
|---|---|---|---|---|
| WineHQ stable 11 | 5.09 / 6.64 ms | 8.6 ms | (baseline) | 0 of 192 |
| stable, audio drivers off | 5.15 / 6.68 | 8.7 | 153,600 of 153,600 | 0 of 192 |
| **GE-Proton 10-34, fsync, audio off** | **4.27 / 5.63 (-17% / -16%)** | **7.8 (-10%)** | 153,600 of 153,600 | 0 of 192 |

What GE needed here (all done on this machine):
- A prefix GE's own Wine made (Wine updates a prefix of another version in place): `~/.wine-wc3-ge`, Windows Python 3.11
  with numpy **1.26** (GE's C runtime lacks `crealf`, which numpy 2 calls), wc3env installed editable, `C:\wc3` -> the lean
  `~/games/wc3-run`, vkd3d's DLLs copied in, `UseEGL=N` and the lowest video options (`training/bench/local_games.py` sets these).
- 32-bit GL: `libgl1:i386 libglx0:i386` (GE runs the game as a 32-bit process; WineHQ 11's wow64 mode used 64-bit GL).
- No `WINEARCH=wow64` (GE's Wine 10 rejects it) and **Wine's audio drivers off**: with them the game hangs before its
  first frame in GStreamer's MP3 codec setup. `wine_env` does both for any `$WC3_WINE` build.

### 4. Learner pauses for inference on a shared GPU

**What:** `--update-pause-ms`: after each gradient step the learner waits for the GPU, then sleeps that long, so the
inference process's batches are not queued behind the update (his: rounds during an update 39 -> 6 ms, ~1,770 -> ~2,155
steps/s). Matters at home, where both share the 3090 Ti.

**Status:** built, measured: **no effect with MPS** (below: within 0.5%). MPS lets the two processes' kernels run side by
side, which is what the pause worked around. The flag stays (default 0) for a GPU shared without MPS.

Measured 2026-10-19: self-play, 96 games (6 actors x 16, combat, cb9's flags), NVIDIA MPS on for the learner and
inference (the other session's E2: +11%), the cuts on; steady-state decided steps/s (`~/wc3/tools/rate.py`, from 120 s),
2 alternating rounds of 6 minutes, 0 failed games and 0 dropped chunks in every arm:

| Arm | Round a | Round b |
|---|---|---|
| `--update-pause-ms 8`, `--nice 0` (as run so far) | 1,798 | 1,783 |
| `--update-pause-ms 0` | 1,798 | 1,792 |
| `--nice 10` (checked: all 96 game processes at niceness 10) | 1,802 | 1,797 |

### 5. Games niced

**What:** `--nice N` runs the games' Wine processes at lower priority, so the learner, inference and actors get the CPU
first when it is full.

**Status:** built, measured: **no effect at 96 games** (table above): the CPU is not full (RAM caps the games, and the
decision path is the limit). Keep at 0; worth a look again if the games ever fill the CPU.

## Training

### 6. More gradient steps per update

**What:** he takes 64 gradient steps per 8,192 agent steps (2 epochs, minibatches of 16 sequences x 16 steps); we take 4
(1 epoch, 128 chunks of 64 steps, minibatch 32). cb9's update moved the policy little (update KL ~0.002, clip fraction
0.06) while the learner waited ~2/3 of each update for games. Flags exist (`--epochs`, `--minibatch`); the experiment is
in [todo/learner-tuning.md](todo/learner-tuning.md).

**Status:** to run (a training experiment, judged by win rate per game played).

### 7. Move the anchor to a recent checkpoint between runs, with a weaker pull

**What:** the KL term holds the policy near its anchor (`--init`). He moved the anchor to the run's recent best between
runs (+11 points, twice) and quartered its weight (+10). Moving it every N updates within a run was worse. We do it by
hand only (cb9's anchor is cb7, by accident). The rule: when a run plateaus, the next run's `--init` is its best
checkpoint, `--start` the same, and `--kl` lower.

**Status:** a procedure, no code; to apply at the next plateau.
