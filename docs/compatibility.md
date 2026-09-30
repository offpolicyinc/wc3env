# Compatibility evidence

The launcher supports the executable hash in [design](design.md#compatibility).
These measurements describe tested fixtures on one machine, not universal engine limits.
Measured 2026-09-17/18: Windows 11 build 26200, x64 Python 3.14.6, Intel Graphics driver
32.0.101.8724, desktop 1920x1200, system DPI query 96. Other physical GPUs, drivers,
desktop scaling settings and Windows versions remain untested.

| Case | Result |
|---|---|
| Echo Isles and Secret Valley, two agents | Passed observation, movement and exact stepping |
| Twisted Meadows, four agents; Twilight Ruins, eight agents | Passed independent observations, movement and exact stepping |
| Human, Orc, Undead, Night Elf | Correct starting workers/buildings; seeded random selection and built-in computer AI also tested |
| 640x480, 1280x720, 1920x1080, presentation on/off | Passed gameplay at every requested size |
| Request 320x240, presentation on/off | Engine applied 640x360; gameplay passed at that actual size |
| 512, 1,024 and 1,536 added flying units | Full own-unit observation and movement of the last spawned unit passed |
| 2,048 added flying units, presentation on/off | Passed observation, last-unit movement and exact advancement after allocator fix |
| Pools of one, two and four processes | Passed 40 rounds of exact advancement and cleanup |
| 360 raw resets before GPU completion fix | After reset 10: used address space grew 123.6 MiB; 17 mostly empty 5.5 MiB graphics ranges remained |
| 360 raw resets with GPU completion | After reset 10: used address space grew 10.7 MiB; no accumulated 5.5 MiB ranges; native arenas stayed at 83.4 MiB |
| 120 session resets, recycling every 32 episodes | Passed across four processes; final-window median was about 10 MiB above the warm baseline |
| Complete seeded Echo Isles match, idle Human versus Orc AI | All 458 sampled states and the 458.4-second outcome matched across reload and a fresh process |
| Two-agent movement replay | All 51 sampled states matched the recording; EOF preserves observations and ends the session |

The original 2,048-unit failure exhausted the process's 2 GiB address space: moved reallocations
kept reserving arenas from an old full arena. Selecting the current matching arena fixes that
case. These are short movement workloads, not a portable unit limit or combat benchmark.

The old free-space search skipped usable bins; merging holes alone did not fix it. Over resets
10–120, native reservations previously grew 104.3 → 147.4 MiB; the corrected search stayed at
83.7 MiB. Tiny allocations pinning old arenas were live UI text caches, not abandoned data.
Remaining growth traced to Direct3D heap capacity and Intel graphics-buffer mappings.
Suppressing `Present` left GPU work pending and accumulated mostly empty 5.5 MiB ranges.
Render-off now waits for GPU completion through an event query. Heap caches still retain
capacity; the 32-episode recycling default remains.

## Reproduce

```powershell
wc3hook/build.bat
python tools/check_compatibility.py --output runs/compatibility.json
python tools/check_compatibility.py --only window --soak-resets 0 --output runs/windows.json
python tools/check_compatibility.py --only reset-soak --soak-resets 120 --output runs/reset-soak.json
python -m unittest tests.e2e.extended.allocator tests.e2e.extended.determinism tests.e2e.test_replay
```

The harness records binary hashes, revision, hardware, actual sizes, simulation timing and
memory after every case. Failures exit nonzero; clamped sizes are `limited`, skipped cases
remain untested. `--army-sizes` selects larger stress cases; no Warcraft assets enter reports.

The tracked replay fixture contains recorded commands, not map assets. Arbitrary custom maps,
other graphics backends and LAN require separate validation. LAN create/join and network
command delivery are not implemented by the current offline API.

## Rollout throughput

Measured 2026-09-28 with `tools/bench.py` on an Intel Core Ultra 5 325 laptop (8 hybrid cores,
Intel Graphics, Windows 11 build 26200) that kept about 2.6 cores busy with other applications.
Workload: Echo Isles, Insane melee AI playing an agent slot against the Insane computer, 250 ms
steps at speed 2048, render off, both players observed every step, one Python process per game.

| Case | Before (hook at `e45a6ae`) | After |
|---|---|---|
| One game, first 6 game minutes | 8.7x realtime, 30 ms per step | 26-33x realtime, 5-7 ms per step |
| Eight games, machine saturated | 45x realtime aggregate, 181 steps/s | 117-145x realtime aggregate, 470-580 steps/s |
| CPU per game (exact cycle counts, minutes 2-6) | | about 4.3 ms game and 1.0 ms host Python per step: 45-47 game seconds per CPU second |
| 60 in-process resets, no recycling | private 219 -> 222 MiB, used address space 2649 -> 2668 MiB | private 220 -> 233 MiB, used address space 2649 -> 2674 MiB |
| In-process reset | 1.7 s | 1.3-1.6 s, mostly map loading |
| Binary observations (`observation="binary"`), one game, minutes 0-8 | | 38x realtime, 5.1 ms per step, 68 game seconds per CPU second (JSON in the same run: 30x, 7.1 ms, 47) |
| `VectorSession`, eight games, minutes 0-5, agent observed | | binary: 350x realtime aggregate, 1,399 steps/s; JSON: 231x, 924 steps/s |

The gains came from skipping drawing when rendering is off, 1 ms timer resolution with a wake
event for the game thread, and querying fog directly for destructables in observations. Both
players' observations stayed identical to the original hook, and to render-on runs, for all
3,600 steps (15 game minutes) of a seeded match. Throughput under Wine has not been measured;
`docker/` has a `bench` mode for it.

## Linux / Wine

The [Docker worker](../docker/README.md) passed observations, worker movement, four exact
1,000-ms steps, same-process reset, close and an agent probe on Ubuntu 24.04 (kernel 6.1),
Wine 11, Windows Python 3.11.9, Xvfb and Mesa llvmpipe with 4 vCPUs / 8 GiB RAM, without
privileged mode. Files under a mounted `/sessions` survived container removal. These short
checks do not establish throughput; model-driven runs, render-off, repeated resets and long
workloads are untested there. Windows results do not establish Linux compatibility.

Modal VM sandboxes (kernel 7.2.6, AMD EPYC, 8 CPUs, no `/dev/ntsync`) passed the smoke test and
[`modal_run.py`](../docker/modal_run.py) benchmarks on 2026-09-28, with the workload above and
binary observations. Per game in an episode: 46-54x realtime. One prefix (one wineserver) did not
scale past four games (135x with four, 127x with eight); four prefixes of two games reached about
380x while playing and about 50 game seconds per CPU second (container cgroup time), with 21% of it
in wineserver. Launch took 13-20 s and an in-process reset 2.5 s.

GCP spot `n2d-standard-8` VMs (8 vCPUs = 4 AMD cores, kernel 7.0, Ubuntu 24.04) through
[`gcp_run.py`](../docker/gcp_run.py) on 2026-09-28, binary observations, 10-minute games:

- `bench`, four prefixes of two games, two episodes each: 92.5x aggregate including launches and
  resets with `/dev/ntsync` passed into the container, 88.7x without it; wineserver used about 20%
  of the CPU either way, so this Wine build does not appear to use ntsync.
- `rollout` ([`vector_rollout.py`](../docker/vector_rollout.py)): `VectorSession` from native Linux
  Python driving Wine workers, eight games in groups of two, as a trainer would run it. 17 s
  startup, then 522 steps/s = 130x aggregate realtime, about 14 game seconds per vCPU second.

## Ladder replays

`wc3env.w3g` reads classic replays (players, map and checksum, every command); `launch(map=replay)`
plays them back in stepping mode with observations. warcraft3.info lists 700 Battle.net 1v1 replays
tagged 1.29 (tournament and ladder games of 2018; `tools/fetch_replays.py`). Checked on 2026-09-29 with
`tools/check_replays.py`, each played to its end:

| Result | Replays |
|---|---|
| Play to the end in sync (152.6 hours of games) | 609 |
| Map not installed (community maps: Fields of Ruin, older 1v1 Flood Plains, Ancient Isles, Irresistible Mind, Nomad Isles, Violet Outpost, Fertile Creek) | 53 |
| Does not load: recorded before 1.29.2 (uploaded April 2018; the map checksum differs on every map) | 36 |

In sync means the units each player selects exist in the playback: the median miss rate is 0.75% and
the 95th percentile 3.7%, units that died within the checked second. Where warcraft3.info's parser
detected a winner (320 of the 609), the playback agrees in 317; the 3 others miss almost nothing, so
the reference is likely wrong there. 25 replays have a minute with 20% or more misses (big fights,
many deaths within a second); the 12 of them with a winner reference all agree. A few replays name a stock map under another file name
(`TSLV.w3x`, `floodplains1v1_LV.w3x`, `Download/thetworivers_LV.w3x`); a copy of the stock map at that
path plays them. The playback's result is empty when the recording ends in the same second as the game.

