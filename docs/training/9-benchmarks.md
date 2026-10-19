# 9. Benchmarks: where self-play's time and money go

The current numbers for each part of self-play on Modal, with how each was measured (2026-10-07, bc10: 11.9M
parameters, 250 ms steps, Echo Isles). The design they come from is in [10-actor-design.md](10-actor-design.md);
what is left to do in [todo/](todo/README.md). The current numbers come first; the run-by-run history (sp13 to sp25)
is kept below them.

## Prices (Modal list prices)

| Resource | Price | Billed on |
|---|---|---|
| H100 | $3.95 / h | the container's lifetime |
| Sandbox CPU | $0.142 / core-h | what the games use (above the request too) |
| Sandbox memory | $0.024 / GiB-h | the request (sp12's bill matched its 32 GiB per sandbox, not the ~22 used) |
| Function (container) CPU | $0.047 / core-h | 3x cheaper than a sandbox's |
| Function memory | $0.008 / GiB-h | |
| Region pinning | 1.75x for a narrow region (`us-east`), 1.15x for a broad one (`us`) | GPU, CPU and memory alike |

## The game under Wine (sandbox)

`training/bench/wine.py` and a waiting test (a scratch script): one sandbox, 64 games, 2 per Windows Python worker, no
model.

| | CPU per game-step | Cores busy |
|---|---|---|
| Stepped flat out | **7.2 ms** | ~16 |
| Each step then 150 ms of waiting (today's loop) | **12.5 ms** | |
| Each step then 350 ms of waiting (sp12's loop) | **13.9 ms** | |
| All 64 waiting, none stepping | | **0.5** (mostly winedevice.exe) |

- One sandbox runs ~2,000 game-steps/s with nothing waiting on it.
- Waiting used to cost ~0.15 of a core per game (64 waiting: 9.7 cores): with the clock frozen between steps, every
  timed wait of every game thread was cut to 1 ms, and two threads per game also waited 1 ms at a time on their own.
  Frozen waits now sleep at least 100 ms and end when the clock starts (wc3hook/clock.c). Against sp12's 350 ms loop,
  a step costs ~5x less CPU than it did.
- Memory, 64 games: 21.8 GiB used (0.34 GiB a game, flat from 5 s to 150 s): Warcraft III 287 MiB each (17.9 GiB),
  the Python workers 55 MiB each x 32, wineserver 15 MiB x 32, other Wine services ~0.6 GiB. With the lowest video
  settings (texture, model and animation quality, particles, lights, shadows, occlusion, spell filter 0; 16-bit;
  640x480): 246 MiB a game, 19.2 GiB (0.30 a game). Played to the end against Warcraft's AI (~490 game-seconds): flat
  to 400 s, then at most 20.4 GiB (0.32 a game) as armies grow and games end. Self-play sandboxes now use the lowest
  options and request 0.32 GiB a game + a fifth + 1 GiB (25.6 GiB for 64 games instead of 32).
- Wine prefixes are 1.6 GB: they are hard-linked, not copied (`wc3env.vector.link_prefix`).

### Proton's Wine with fsync (2026-10-10): ~20% less CPU a game-step

`training/bench/wine_sync.py`: 64 no-op games, 250 ms steps, 50 ms of waiting after each round (the loop's tunnel and
inference), 4 cores requested as in self-play, the variants alternating in one sandbox.

| Wine | CPU per game-step | A round's work besides the wait |
|---|---|---|
| WineHQ stable 11.0 (until now) | 9.34, 9.44 ms | 29.7, 30.4 ms |
| Valve's Proton 11 Wine (Kron4ek's wow64 build), `WINEFSYNC=1` | **7.24, 7.53 ms** | **21.8, 23.3 ms** |

- fsync keeps Wine's events, mutexes and waits on futexes (`futex_waitv`, in Modal's 7.2 kernel) instead of a
  wineserver round trip each. The same Proton Wine without fsync was no better than stable (32 games: 7.92 against
  8.05 ms; with fsync 6.86). warcraftsim measured 28% on its machine; our frozen-clock waits had already removed
  part of what fsync saves.
- Sandbox CPU is billed on use: ~$10 of a 512-game run's ~$25/h becomes ~$8.
- The same games: Warcraft's AI against itself on 4 seeds, 1,200 steps each (5 game-minutes, 30-45 units a side by
  the end), every step's time, economy, score and unit rows agree between the two Wines (`training/bench/wine_parity.py`).
- Only a wow64 build runs here: Modal's kernel runs no 32-bit processes (a 32-bit futex call fails with ENOSYS and
  glibc aborts). GE-Proton 10's wow64 build page-faults Warcraft III (fsync on or off); GE-Proton 11 has no wow64
  build; Wine 11.18 staging-TkG never started fsync. ntsync (Wine 11's own) needs `/dev/ntsync`, not in Modal's kernel.
- 64 games hold all 256 of Xvfb's default connections, with either Wine: a game that cannot connect never starts
  (Proton's Wine needs a few more, so its games hung at start). Xvfb now takes 2,048 (`docker/with-display.sh`).
- Turning off Wine's HID/USB/Bluetooth bus drivers (warcraftsim's other fix) saved nothing measurable here.

## Features and masks (in the game hook)

The hook computes each observed player's features and masks after its observation, in the game process
(wc3hook/features.c, masks.c; the same C as a library through `training/policy/native.py` for recorded
observations). On 3,200 recorded steps (`training/bench/record_obs.py`):

| Per player-step | |
|---|---|
| Features and masks (C) | **0.093 ms** (Python did the same in ~1.15 + ~0.5 ms) |
| Writing the block | 0.008 ms |
| The block | 37.5 KB; **4.9 KB** zlib-compressed (the observation itself: 1.0 KB) |

At 512 games with both players observed, the blocks are ~19.5 MB/s through the tunnels (the observations alone
were ~4.2 MB/s).

## The tunnel (sandbox → container)

Set by where Modal places the two, not by the tunnel: a CPU function serving 1 KB replies through `modal.forward`, a
sandbox asking, 300 round trips each.

| Sandbox | Container | Round trip p50 |
|---|---|---|
| us-east | AWS ap-southeast | **269 ms** (nothing pinned) |
| us-east | Azure us-east | 13.5 ms (nothing pinned) |
| us-east | AWS us-east | **7.4-7.8 ms** (both `region="us-east"`) |
| Azure us-west | Azure us-west | 35 ms |

In a step (`training/bench/relay.py` and self-play's `net`: the tunnel both ways, zlib at its ends), by where the
GPU container is pinned (2026-10-08):

| Container | Sandbox | Tunnel per step |
|---|---|---|
| us-east (GCP) | us-east | **~10-14 ms** |
| us-east (GCP) | us-central | ~27-34 |
| eu-west (GCP) | eu-west (OCI) | ~42 |
| us-central (GCP) | us-central | ~50 |

- The tunnel goes through Modal's relays (`r43x.modal.host`), on AWS near the container (us-east-1 for us-east,
  Ireland for eu-west): a shared region name is not a short way. So the container stays pinned to us-east, and
  sandboxes are not followed (tried: `--region follow`, dropped).
- Unpinned, the container and the sandboxes can land on different continents; sp12's 50-85 ms was likely that.
- A direct connection to the container's private address fails (sandboxes have no `i6pn`).
- Bandwidth: one connection ~1.4 MB/s, 32 in parallel 49 MB/s (`training/bench/tunnel.py`, run unpinned).

### A step's way there and back, hop by hop (2026-10-08)

`training/bench/relay.py`: one games sandbox as self-play makes it (2 bridges x 32 games, features from the hook),
a CPU container pinned to us-east in the actors' place, no-op steps held 70 ms each (self-play's pace). Host Azure
us-east, sandbox us-east; ms per step from `info["hops"]` (wc3env.vector), median (p90):

```
host's socket -> reply back (h_round)   15.2 (17.5)
  tunnel both ways, zlib at its ends    10.4 (12.2)   net = h_round - b_span
  in the sandbox (b_span)                4.6  (5.8)
    the game's step under Wine           3.5  (4.6)
    pipes into and out of Wine           0.5
    the bridge's own work                0.4
    waiting for the game's thread        0.1
```

- The bridge and the pipes into Wine cost ~1 ms: removing them would not pay.
- The tunnel is ~10 ms between two clouds in us-east (7.4-7.8 with both pinned to one: ~3 ms for 1.75x).
- sp15's ~25 ms "transit" from us-east was this ~15 plus replies waiting in the actor's queue while it was busy
  (now `h_queue` in the actor's report).

## The actor (container CPU)

Per game per step, the CPU's part of a decision now that features and masks arrive ready (recorded observations,
the model's reply canned so only the actor is timed):

```
parsing the features block          0.074 ms
the model's inputs                  0.046 ms
decisions from the choices          0.033 ms   (the chosen action's masks, the game's commands)
recording the step                  0.045 ms
total                              ~0.20 ms  -> ~6 ms per 32 games (it was ~40 ms with Python features and masks)
```

## The inference process (H100)

Two processes on the second H100, each deciding for half the actors (`training/selfplay/inference.py`): a batch is
whatever requests came in within 2 ms. The latest policy's step (the model, all 10 heads, the critic, each game's
memory in tables on the GPU) is compiled (torch.compile: the transformer blocks and SSM layers, encode_step, the
heads) and replays as a CUDA graph at fixed sizes: seats 64/128/256, unit rows 96/128/256, the window always 64
steps (`policy/agent.py`'s Model). The 9 common shapes are made at the first batch.

GPU time of one step, 58 seats at 96 unit rows (`training/bench/step_parts.py`):

| | Kernel by kernel | CUDA graph | Compiled + graph |
|---|---|---|---|
| Whole step | ~20 ms wall (~15 of it launching ~1,070 kernels) | 3.9 ms | **1.8 ms** |
| of it: encoding (map, units, 4-layer step transformer) | | 2.3 | 1.1 |
| heads | | 0.95 | 0.26 |
| memory + time attention | | 0.44 | 0.16 |

- The GPU work, not the CPU, was the limit once the step was a graph: two batches on two streams of one process do
  not overlap (measured with a one-off bench: 4.7 ms alone, 9.5 each together), and two inference processes kept the
  GPU ~100% busy (sp21: a batch 15 ms, requests ~11 ms unread behind the one before).
- A request's time from sent to answered, games paced as in a run (`training/bench/inference.py`, 512 seats, 4
  requests an actor): **8.6 ms median, 14.2 p90** compiled; ~15 graph only; one process 23-28 ms.
- Making the 9 compiled shapes: 385 s cold in a run (742 s on 8 cores), **90 s** with torch.compile's cache
  (`/dataset/cache/inductor.tar` in the Volume, unpacked to local disk at the container's start, packed back after
  8 minutes; read from the Volume directly it was as slow as none). Snapshot policies are not compiled.
- Heads are drawn as argmax p/Exp(1) (exact, graph-safe; multinomial syncs).

## The learner (H100)

One PPO update of 128 chunks x 64 steps (8,192 steps) in 4 gradient steps of 32 chunks; the transformer blocks and
SSM layers compiled (unit rows rounded to 64, 96, 128 or 256 so a few fixed-size programs cover every batch); the
next batch prepared on a thread. Timed alone on the GPU with chunks grown to mid-game unit counts (53 a step, 92 at
most, as in sp16-18; the recorded games only reach ~40): `training/bench/profile_learner.py`.

```
copy to the GPU (pinned)                        0.015 s
reference pass (the clone's distributions)      0.16
4 gradient steps                                0.44
update                                          0.62 s -> ~13,200 steps/s   (1.03 s, ~7,900, before 2026-10-08)
GPU busy                                        97% of it
```

What took it from 1.03 to 0.62 s:

| Change | Update | Why |
|---|---|---|
| embedding tables' gradient as a one-hot matmul | 1.03 → 0.81 | PyTorch sums each index's rows on one thread, and most rows are padding (index 0): 72 → 21 ms a minibatch (`bench/embedding_grad.py`); with the two below |
| the map's 19×19 max-pool as two 1-D passes; batches in pinned memory | (in the above) | the same result, 8× cheaper; copies no longer wait |
| attention on cuDNN's kernel | 0.81 → 0.735 | with the key mask PyTorch took its memory-efficient kernel: 11.0 → 5.9 ms a call (measured with a one-off bench) |
| no waits for the GPU inside an update | 0.735 → 0.70 | statistics read once at the end, masked means: 274 waits → 4 (more in a run, where threads compete) |
| a 96-row size | 0.70 → 0.62 | ~90-unit steps padded to 128 rows: 193 tokens where 161 do |

- In sp18 (before the last two rows) an update's work was ~0.8 s and the learner waited ~0.6 s of each update for
  chunks: the games, not the learner, set the speed.
- What is left: attention 0.13 s, dtype copies ~0.06, the one-hot gradient's float32 matmul ~0.05, the bags'
  forward ~0.03, the reference pass 0.16 (needed for the KL to the clone).
- Compiling the whole forward pass fails (Inductor mixes float32 and bfloat16 in one matrix product).

## Learner and actors agree

At the same weights (`training/bench/drift.py`, 24,576 steps), |learner - actor| log-probability: median 0.0004,
99th percentile 0.03, largest 0.11-0.12 (ratio 1.12-1.13), none past PPO's 0.2 clip. With the inference step
compiled: p99 0.039, largest 0.15 (ratio 1.16), none past the clip. The "ratio drift" the learner
logs above 1 comes from its later minibatches, after the weights moved. `training/tests/test_inference.py` holds the
inference process's choices to the learner's scores.

## Rates

| Part | Steps or decisions / s | How |
|---|---|---|
| The games, 8 sandboxes, nothing waiting on them | ~16,000 | offline |
| The learner, compiled (0.62 s an update) | ~13,200 | offline |
| Inference, 2 processes, paced as in a run | 10,500 decisions/s at 8.6 ms a reply | offline |
| The learner in a run (sp28: ~0.62 s of work an update, packed attention; sp24/sp26: ~1.05-1.2 s) | ~13,000 | real run |
| **sp24 (inference compiled + CUDA graphs, event-driven actors, 512 games; 7 of 8 sandboxes in us-central)** | **~7,000-7,300 decided** | real run |
| sp18 (inference as CUDA graphs, two H100s, 512 games) | ~6,500 decided | real run |
| sp15 (two H100s, two inference processes) | ~4,840 decided | real run |
| sp13 (one H100, one inference process) | ~2,900 decided and learned | real run |
| sp12 (the old layout: actors with models of their own, 512 games) | 1,460 decided, 1,390 learned | real run |

## A full-size run: sp13 (2026-10-07)

sp12 resumed for ~13 minutes at full speed: 8 sandboxes x 64 games, 16 actors, the H100 container pinned to
us-east (`--region`), the sandboxes unpinned (3 us-east, 4 us-central, 1 eu-west), features from the hook.

| Sandbox region | Per actor (32 games) | Per game | Loop per step |
|---|---|---|---|
| us-east | ~195 steps/s | 6.1 steps/s | ~165 ms |
| us-central | ~190 | 5.9 | ~170 ms |
| eu-west | ~120 | 3.8 | ~265 ms |

- **~2,860-2,930 steps/s decided, and learned at the same rate** (10 updates of 8,192 steps every ~28 s); 0 failed
  games. 2x sp12.
- The inference process was 56-63% busy at ~210-220 games a batch; the learner waited for chunks between updates.
  So neither is the limit: **the loop is**. 512 games / ~0.167 s = ~3,070 steps/s, as measured. More games per H100
  raise it, until the inference process (ceiling ~8,000) or the learner (~8,000) is reached.
- Policy lag: 6-9 updates. The learner updates every ~2.8 s and the inference process pulls weights every 20 s.
- An unpinned launch put the container in ap-south against sandboxes in the US (~200-270 ms round trips): stopped
  after a minute.

The loop each game waits through (game step + tunnel + actor + inference) sets the speed: steps/s = games running /
loop time. ~165-170 ms with the container and the sandboxes in the US (sp13), ~350 ms in sp12.

### Inference compiled, actors event-driven (sp22-sp25, 2026-10-08): ~7,000-7,300 steps/s

The same 512 games, with the actors ending answers as they come (47ae50f), requests stamped on their way, and the
inference step compiled. Per actor (32 games), steps/s:

| Sandbox region | sp18 | sp21 (graph only) | sp22 | sp24 (compiled) |
|---|---|---|---|---|
| us-east | 441 | 446 | ~480 | ~555 |
| us-central | 369 | 381 | ~410 | ~443 |
| eu-west (made again 4 times, stayed) | | | ~176 | |

- A us-east actor's step (sp22): game 28 (tunnel ~14, Wine ~7), queued ~4, decide ~23 (request → answer ~17:
  unread ~6, gathering 2, batch ~7, back ~2.5). In sp21 the request took ~30 and decide ~37.
- Inference batches ~6 ms median (sp21: 13), ~46 seats, 96 unit rows in ~98% of them.
- With every sandbox in us-east, ~16 x 550 = ~8,800 steps/s (estimate); the usual half-and-half ~8,000.
- The learner keeps up: an update every ~1.05 s (~7,800 steps/s) once compiled; its first update ~6 minutes after
  the start with the compile cache, ~11 without.

### Inference as CUDA graphs (sp18, 2026-10-08): ~6,500 steps/s

sp15's layout with the inference step as CUDA graphs and 2 ms of gathering, and the learner's first speedups
(8fe2e81, f8bbdf5); 4 sandboxes in us-east, 4 in us-central. ~7 minutes at full speed.

| ms per step | us-east sandbox | us-central |
|---|---|---|
| game: sent → back (Wine step ~8, transit ~13) | ~28 | ~40 (transit ~27) |
| queued (the actor's round) | ~8 | ~8 |
| decide: begun → sent (request → answer read ~30) | ~36 | ~38 |
| **loop** | **~72** | **~86** |

- **~6,530 steps/s decided** (sp15: 4,840), 290-340 a second per actor.
- The answer is ready ~14 ms after the request (batch 9, gathering 2.6, queued 2.2) but read ~30 ms after: it waits
  for the actor's next round. The actor ending answers as they arrive is the next cut.
- The learner kept up (an update every ~1.5 s, ~0.6 s of it waiting for chunks); policy lag 7-8, a few chunks
  past --max-lag dropped.

### The fixes (sp15, 2026-10-07): ~4,800 steps/s

Two H100s (`--gpu H100:2`: the learner on one, 2 inference processes on the other), far sandboxes made again (all 8
in the US), weights packed once a version, the game's worker timing its step. ~5 minutes at full speed:

| ms per step | us-east sandbox | us-central |
|---|---|---|
| game: sent → back | ~32 (Wine step **~7**, waiting in the worker 0.3, **transit ~25**) | ~70 (transit ~62) |
| queued | ~7 | ~7 |
| decide (inference request → reply ~55) | ~61 | ~60 |
| **loop** | **~100** | **~140** |

- **~4,840 steps/s decided** (sp13: ~2,900); inference batches 30 ms median, 35 at p90 (no learner tail), 55% busy.
- The learner is now the limit: an update every ~1.75 s (~4,700 steps/s; 1.0 s offline), so chunks queue and the
  policy lag rose to ~21 updates.
- Transit (tunnel, relay, pipes into Wine) is most of the game side: ~25 ms from us-east, ~62 from us-central.

### Where the loop goes (sp14, 2026-10-07: sp13's layout with per-stage timings)

Mean ms per game step, from the actors' and the inference process's reports (sp14 had a few minutes without the
learner first: its share is the difference).

| | us-east sandbox | us-central | eu-west | us-east, no learner |
|---|---|---|---|---|
| game: commands sent → observation back (tunnel, bridge, Wine) | ~48 | ~68 | ~118 | ~48 |
| queued: observation → its decision begun (the actor's round) | ~8 | ~6 | ~7 | ~8 |
| decide: begun → commands sent (inference reply, the actor's round) | ~108 | ~97 | ~128 | ~68 |
| **loop** | **~165** | **~170** | **~255** | **~125** |

- Inference, ~95-110 ms from request to reply (65 without the learner): its batch 41 ms median but 106 at p90 with
  the learner training on the same GPU (45 without), 11 ms gathering, ~8 ms queued after being read, and ~40 ms
  unread while the batch before runs. Two processes on one GPU take turns on it (no MPS): a batch caught behind the
  learner's kernels waits, and every request queued behind that batch waits too.
- The game side, ~48 ms from us-east, is ~30 ms more than the tunnel (~8) and a Wine step (~7-12) explain: not
  yet split (the bridge, two games per Windows Python worker, the relay).
- Weight pulls: ~2.0 s each (pickled and zlib-compressed per call by the learner), on the inference process's own
  thread since ef188cd; loading them still stops the batches ~65 ms, every ~5 s. Policy lag stayed ~6.5: a chunk
  is 64 steps, ~10.5 s of play (~4 updates) from its first step's version.

## Home 3090 Ti (2026-10-18)

One box: Ryzen, 32 threads, 60 GB RAM, one RTX 3090 Ti for both the learner and inference; games under WineHQ stable
11 + Xvfb + llvmpipe, two per prefix (`run.py --wine`). Combat mode, cb9's flags, every game against the ladder at
stage 0.85, 6 actors x 16 games (96: RAM's limit) unless said. Decided steps/s are the actors' game steps a second
after 2 minutes' warm-up (the learner's "steps/s" is a running mean from the start: it reads ~20% low), A/B arms of 6
minutes alternating; `runs/selfplay/*/log.txt`.

| Setup | Decided steps/s | Notes |
|---|---|---|
| 32 games, no model (`docker/vector_rollout.py`) | 4,583 | CPU: 28 of 32 threads |
| 16 games (2 x 8) | ~420 | too few games |
| 96 games | 1,577 / 1,574 | decisions ~40 ms of a ~58 ms step, game steps ~11 ms; GPU 99%, RAM 47.5 GB |
| + NVIDIA MPS (learner and inference kernels side by side) | 1,743 / 1,747 (+11%) | inference batch p90 30 -> 20 ms |
| + game-code cuts (`--game-cuts 1225`) | ~1,790 | with MPS, `--update-pause-ms` and nice made no difference (robinkahlow's A/B) |
| + less padding (48 seats x 64 unit rows, was 64 x 96) | 1,943 / 1,880 vs 1,797 / 1,798 (+6%) | batch p50 10 -> 8-10 ms |
| + TCP_NODELAY on the training links (`training/selfplay/transport.py`) | 2,243 / 2,243 vs 1,850 / 1,862 (+21%) | inference's gather 9 -> 2.5 ms, decisions 36 -> 25 ms; the learner now waits 0.8 s an update, not 2.4 |
| 128 games | - | out of memory: the kernel killed the desktop's processes |

- The GPU is the limit at 96 games: the learner trains ~1.8 s of every ~4.8 s update and waits the rest, inference
  runs ~38 batches a second of ~10 ms each, and both contend. Leaving a batch in flight while the next one's requests
  are read (pipelined inference) was slower (~1,400-1,450): each batch then finished ~28 ms after launch, not ~11.
  Reading a request is cheap (p50 0.1 ms, 70 KB), but 5% of reads waited ~40 ms on Nagle's algorithm and delayed ACKs
  (send_bytes writes the length and the body apart): 88% of the reading time, until TCP_NODELAY.
- RAM sets the game count: ~0.4 GB a Warcraft process (LOW_VIDEO set in the prefix; 0.36 before) and growing with
  every staged fight until the process recycles, ~17 GB the training processes. Run inside
  `systemd-run --user --scope -p MemoryMax=50G` so an OOM kills only the run.
- One GPU for both needs the learner's `--micro 4` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (otherwise
  learner 10 GB + inference 12.7 GB ran out).
- cb9 won 0.66-0.79 of these games at 0.85 (Modal: ~0.58-0.67 over its run).

## Cost

- **Estimate for a 512-game run now: ~$25/h, ~$0.95 per million steps** (no bill yet for this layout): sandbox CPU
  ~$10 (~9-11 cores a sandbox at $0.142 / core-h), sandbox memory ~$5 (the request, 25.6 GiB per 64 games), two H100s
  ~$8, the container's CPU and memory ~$2.
- sp12's busiest hour, billed: **$32**: sandbox CPU $21.73, memory $7.82, the H100 $2.79. The CPU came from waiting
  games burning cores (fixed since) and memory from the 32 GiB requested per sandbox (now sized to the games).

## Lessons

- Measure where the time goes before optimizing: the GPU was ~1% of a decision when it looked like the bottleneck,
  and a "33 ms per step under Wine" estimate from throughput was wrong (the cores were idle).
- Python per game per step is what costs: the features, the masks and per-row sampling were most of a decision until
  they moved to C and to one batched GPU pass.
- Many processes on one GPU take turns: one inference process instead of one model per actor.
- Read steps/s from the change between reports, as soon as there are two, and stop a test then.
- From Git Bash on Windows, set `MSYS_NO_PATHCONV=1` (else `/dataset/...` and `/tmp` arguments become Windows paths).
- Every launch gets an automatic stop of its own run (`training/cloud/guard.sh`); stopping everything
  (`training/cloud/stop_all.py`) also stops other sessions' tests.
