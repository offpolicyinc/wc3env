# 5. Infrastructure

## Where each piece runs

| Piece | Needs | Where |
|---|---|---|
| Games | CPU (Wine) | CPU VMs; ~130 game-seconds per wall second per 8 vCPU on GCP |
| Inference (actors choosing actions) | GPU, batched | next to the games |
| Learning (gradient steps) | GPU | one GPU |
| Snapshots, logs | storage | a GCS bucket |

Throughput to plan around, at 4 decisions per game-second:

- one 8-vCPU spot VM: ~520 steps/s, ~$0.07/hr;
- 256 vCPUs (32 such VMs): ~16k steps/s, ~1.4 billion steps/day, ~$2.2/hr;
- one L4 learning on a ~10M-parameter model: very roughly 10-30k steps/s.

So a single L4 roughly matches 256 vCPUs of games. AlphaStar used ~50k steps/s per learner;
TStarBot-X trained for 57 days. We can do on the order of 10⁹ steps per day for a few dollars an hour.

## How the references run

- **AlphaStar**: games on CPUs, inference batched on separate TPUs, one learner per agent; actors
  pull weights every 10 s; batches of 512 chunks × 64 steps.
- **TStarBot-X**: 13k CPU cores for games, 48 GPUs just for inference (one server batching requests
  from 6,720 games), 96 GPUs for learning.
- **DI-star small scale**: one PC, 8 GB GPU, batch 2-4 chunks of 16-32 steps. Runs, but slowly.
- **Pluto** (inference only): int8 CPU inference in a separate process, ~20 ms per step.

## Stages

The plan as first written, kept for its reasoning. What runs now: [7-selfplay.md](7-selfplay.md) and
[10-actor-design.md](10-actor-design.md); its cost: [9-benchmarks.md](9-benchmarks.md#cost).

**A. Laptop** (now). `VectorSession` with 8 games on Windows (~350x realtime), torch on CPU or the
laptop GPU. For writing and debugging actions, features, trainer. Nothing here needs the cloud.

**B. One GCP box** (after the quota is approved). A `g2-standard-32` spot VM: 32 vCPUs + one L4,
everything in one process tree: `VectorSession(wine_workers())` for ~24 games, inference and learning
on the L4, checkpoints to GCS. No networking to build, no egress. Roughly $0.6-0.8/hr spot.
Quota needed: GPUs (all regions) ≥1 (requested) and CPUs (all regions) ≥32; us-central1 already
allows 1 L4 and G2 counts against its general CPU limit (200).

**C. Scale out.** Add N2D spot VMs running only games + a small actor process. Two ways to do their
inference:

- *central GPU inference* (SEED/TStarBot-X style): actors send features to the L4 box, get actions
  back. Keeps a single copy of the weights, but every step crosses the network (~10-50 KB per step,
  same zone so free, needs low latency);
- *CPU inference on the actor VMs* (IMPALA style): each VM batches its own games on CPU and only ships
  finished chunks. Fine for a ~5M model; costs CPU that Wine would otherwise use.

Start with CPU inference on actors (simpler, tolerant of latency and preemption); switch if the
model grows. (On Modal it went to central GPU inference: one process deciding for every actor in the H100's
container, 10-actor-design.md.) Either way:

- actors are stateless; a preempted spot VM just disappears and a new one joins;
- weights: the learner writes them to GCS (or serves them over HTTP); actors poll every ~10 s;
- chunks: actors push to the learner over TCP (a plain socket with length-prefixed pickles, like
  `vector.py`, or ZeroMQ);
- the learner checkpoints to GCS every ~15 minutes; the learner itself can run on-demand (not spot)
  if preemption restarts become annoying (~3x the price for that one VM).

`docker/gcp_run.py` builds the image and keeps the license in Secret Manager; `training/cloud/gcp.py` runs
evaluations and recordings on it (self-play on GCP VMs was tried: [8-decisions.md](8-decisions.md), Tried and dropped).

## Modal

Self-play (training/cloud/selfplay.py) and cloning run on Modal, with GPUs needing no quota. Sandboxes bill CPU at
3x the function rate, so only the games run in sandboxes; actors, inference and the learner share one container
with two H100s ([10-actor-design.md](10-actor-design.md)).

## Cost sketch

First estimates, kept for their reasoning; the current estimate is in [9-benchmarks.md](9-benchmarks.md#cost).

| Run | Setup | $/hr | Steps/day |
|---|---|---|---|
| Debug | laptop | 0 | ~30M |
| First real run | 1 × g2-standard-32 spot | ~0.7 | ~100M |
| Scaled | L4 box + 16 × n2d-standard-16 spot | ~3 | ~1B |

Set a billing budget alert before stage C.
