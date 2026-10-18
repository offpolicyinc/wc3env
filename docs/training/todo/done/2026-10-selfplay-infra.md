# Self-play infrastructure: the list as of 2026-10

The layout and why: [10-actor-design.md](../../10-actor-design.md). Time per part and the run-by-run measurements:
[9-benchmarks.md](../../9-benchmarks.md) and [selfplay-anatomy.html](../../selfplay-anatomy.html). Cost: [9-benchmarks.md](../../9-benchmarks.md#cost).

## To do

**Measure at full scale (first)**
- [x] A full-size run (sp13, 9-benchmarks.md): ~2,900 steps/s decided and learned, 2x sp12; loop ~165 ms per step
      in the US. The loop is the limit: the inference process was ~60% busy, the learner waiting for chunks; actors
      steady at ~190 steps/s each with the features blocks (no tunnel saturated).
- [ ] A real bill for the current layout (the estimate is in [9-benchmarks.md](../../9-benchmarks.md#cost)).
- [x] Where the loop goes (sp14): inference ~108 ms of 165, a third of it batches waiting behind the learner on the
      shared GPU; the game side ~48.
- [x] The fixes measured (sp15: two H100s, two inference processes, sandboxes in the US, weights packed once):
      ~4,840 steps/s, loop ~100 ms.
- [x] The learner in a run: ~0.62 s of work an update with packed attention (sp28), as offline; it was ~1.05-1.2 s.
- [ ] More games per container now that the learner (~13,000 steps/s) and inference (GPU 1 ~1/3 busy) have room.
- [x] Drop old chunks instead of training on them: chunks more than 8 updates old are dropped (`--max-lag`; policy
      lag was ~21 in sp15).
- [ ] More games (speed = games / loop time), until the learner is ~90% busy; or sandboxes in us-east only.
- [x] Weights every 1 s on the inference process's own thread, packed once per update.
- [x] Relaunch sandboxes that land outside the US (482a4ae).
- [x] The game side's transit, hop by hop (283ccaf, `training/bench/relay.py`): from us-east the tunnel ~10-14 ms,
      the sandbox's bridge and pipes into Wine ~1 ms; the rest of sp15's ~25 was replies waiting in the actor.
- [x] The actor by events (47ae50f): answers become commands as soon as they are in; blocked ~5 -> 0.2 ms.
- [ ] One H100 for learner and inference, or two A100s instead of two H100s: one short run to compare.

**Cost**
- [x] Sandbox memory, billed on the request (0e2e95a): the lowest video settings in each games sandbox, and a request
      sized to the games (0.32 GiB a game + a fifth + 1 GiB: 25.6 GiB per 64 games instead of 32, -20%).
- [ ] 4 games per Python worker instead of 2 (-4% memory): measure its effect on stepping first.
- [x] Tunnel placement: the container stays pinned to us-east (the tunnel per step: us-east to us-east ~10-14 ms,
      us-central to us-central ~50, eu-west to eu-west ~42; following the sandboxes tried and dropped). Sandboxes
      land where Modal puts them (remade only on another continent): forcing them to us-east was declined.
- [ ] The rest of a sandbox's CPU: sp12 was billed ~19 cores a sandbox, the games explained ~11-12. The bridge's
      own work is ~0.4 ms a step (relay bench); compare a sandbox's billed cores with what its games use.

**Speed (once the full-scale run shows the limit)**
- [x] Inference: two processes; the step compiled and replayed as CUDA graphs at padded sizes (made at the start in
      ~90 s with torch.compile's cache in the Volume).
- [x] Inference off the learner's GPU (`--gpu H100:2`): batches 30 ms median, 35 at p90 (sp14 on a shared GPU:
      41 and 106). MPS tried and set aside (a small batch next to a busy process: 17 ms taking turns, 50 under MPS).
- [x] Learner: GAE from the values the actors recorded, not a value pass of its own (`--actor-values`).
- [x] Learner: the next batch built on a thread, in pinned memory, while the GPU trains the current one.
- [x] The features block: the tree grid only when it changed.

**Training**
- [x] Ladder counting without a guessed delay: actors report each ladder game's begin, end or drop; the league
      counts a game once every game begun before it has ended (a silent one stops blocking after 10x the time limit).
- [ ] Continue from sp12 (`runs/sp12/last.pt`) once counting is fixed.

**Model**
- [ ] The time attention has no positions (it sees the last <=64 steps as a set): add them (e.g. rotary).
- [ ] A bigger model (~50M, bf16): compare its cloning loss with bc10's first; the GPU's math is mostly idle.

**Housekeeping**
- [ ] Data set conversion on GCP or Modal: its images need a C compiler for the features library (or ship the
      built `.so`).
- [ ] A PR to main for the wc3env changes (RemoteWorker, link_prefix, VectorSession.debug, the hook's features
      block, step decisions).
