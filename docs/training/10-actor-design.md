# 10. The decision path: game, actor, inference, learner

How one step of one game becomes a decision and then training data, which machine does each part, and why it is
split this way. The numbers (with how each was measured) are in [9-benchmarks.md](9-benchmarks.md); the whole pipeline on one page in
[selfplay-anatomy.html](selfplay-anatomy.html); features and masks in the hook in
[12-hook-features.md](12-hook-features.md).

## The rule

Work done **once per game** goes where it is cheapest per game: in C next to the game, or in the actors' light
Python. Work done **once per batch** goes on the GPU, in one process. Nothing runs "for each game, in Python" on the
GPU's side, and no two processes take turns on the GPU for small batches.

## The layout

```
 SANDBOX ×8 (Wine; CPU billed on use;       GPU CONTAINER (Modal function: 48 cores, 2 H100s: --gpu H100:2)
 made again on another continent)           ┌──────────────────────────────────────────────────────────────┐
 ┌───────────────────────────────────┐      │ ACTORS ×16 (CPU only, 32 games each)                         │
 │ 64 games                          │ obs  │   parse the block, build the model's inputs     ~0.1 ms/game │
 │ each step, in the game process:   │ +    │   even actors ──► inference 0    odd actors ──► inference 1  │
 │   the game: 250 ms of game time   │ block│        │ inputs                        ▲ choices             │
 │   the hook: features + masks      │ ───► │        ▼                               │                     │
 │     (C, ~0.1 ms per player)       │      │ GPU 1: INFERENCE ×2 (processes; ~46 games a batch each)      │
 │                                   │      │   model, 10 heads, critic; each keeps its games' memory      │
 │ applies the commands; folds the   │ ◄─── │ ACTORS: choices → commands + decision; record the step       │
 │ decision into the next features   │ cmds │        │ 64-step chunks                                      │
 └───────────────────────────────────┘ +dec │        ▼                                                     │
                                            │ GPU 0: LEARNER: PPO update per 128 chunks                    │
                                            │        └── new weights every 1 s ─► both inference processes │
                                            └──────────────────────────────────────────────────────────────┘
```

With one GPU (`--gpu H100`) the learner and the inference processes share it, as in sp13-sp14.

## One step

1. **The game** advances 250 ms under Wine and writes the binary observation into shared memory.
2. **The hook** computes that player's features and masks from it, in C (the unit rows, the remembered tree grid,
   scalars, the last decisions; the type mask, each allowed type's argument mask, each (type, argument)'s actor
   mask), folding in the decision the actor sent with the step before, and writes the block after the observation.
3. **The tunnel** carries observation and block to the game's actor (zlib, ~6 KB a player-step) through Modal's
   relays on AWS near the container: ~14 ms there and back from a us-east sandbox, ~25-50 from us-central. In the
   sandbox, the bridge and the pipes into Wine add ~1 ms.
4. **The actor** parses the block and sends the model's inputs to the inference process: unit rows as float16, ids in
   16 bits, the tree grid only when it changed, the mask tables. It works by events, not rounds: observations are
   taken as they come, a request goes out ~3 ms after the first of them, up to 3 requests are in flight, and each
   answer is turned into commands as soon as it is in (`Agents.begin` / `end`).
5. **An inference process** (two of them, each serving half the actors) joins its actors' requests into one batch per
   policy (the requests that came in within 2 ms, ~46 games), padded up to 64/128/256/512 games and 96/128/256 unit rows. At
   each size the step is compiled (torch.compile) and replayed as one CUDA graph: it draws the map on the GPU from the
   unit rows, runs the trunk in bfloat16 and the 10 heads in float32, each head sampled for all games at once with the
   masks looked up on the GPU, and the critic. The 12 graphs are made at the start (~90 s with torch.compile's cache,
   kept in the Volume as `/dataset/cache/inductor.tar`). It keeps each game's SSM state and attention window in tables
   on the GPU, and returns the memory a chunk started from. Request to answer: ~17 ms in a run.
6. **The actor** turns the choices into game commands, records the learning player's step (features, choices, the
   chosen action's masks, log-probability, value), and sends the commands and the decision back with the next step.
7. **The learner** takes 64-step chunks, shipped as soon as each is done; every 128 it makes one PPO update (using the
   values the actors recorded for GAE, the next batch built on a thread in pinned memory, chunks more than 8 updates
   old dropped) and packs the weights once, which each inference process pulls every 1 s on a thread of its own.

## Why each piece is where it is

- **Features and masks in C, in the hook.** In Python they were ~1.6 ms per game per step on one core, most of a
  decision; in C ~0.1 ms, beside a game step of ~7 ms. Replays for cloning go through the same C (as a library),
  so there is one implementation. The masks are tables of every choice that can come up, so the GPU can sample all
  ten heads without asking the CPU between heads.
- **Few inference processes, not one per actor.** With an actor per GPU process, 16 CUDA contexts took turns and
  each of the ~10 GPU→CPU syncs per step waited for the others; one process sees big batches and holds one copy of the
  model.
- **Two of them, not one.** A batch is ~2/3 CPU work (joining requests, packing inputs, answering) and ~1/3 GPU,
  and a request that arrives during a batch waits for it to end. With two, one prepares while the other's batch is on
  the GPU: a request is answered in ~8.6 ms offline, ~23 with one process.
- **The step as a compiled CUDA graph.** Kernel by kernel, a step was ~20 ms, ~15 of it launching ~1,070 kernels; as a
  graph 3.9 ms of GPU, compiled 1.8 ms. A graph fits one array size, hence the padded sizes.
- **Inference on a GPU of its own.** Sharing the learner's GPU, a batch caught behind a learner update waited: sp14's
  batches took 41 ms at the median but 106 at p90 (45 without the learner), ~35 ms of a 165 ms loop. MPS made it
  worse (a small batch next to a busy process: 17 ms taking turns, 50 under MPS). On its own GPU, sp15's batches took
  30 ms median, 35 at p90.
- **Actors on the container's CPU, not the sandboxes'.** Container cores cost a third of sandbox cores, and the
  actors are now light (~6 ms per 32 games).
- **Each game's model memory on the GPU.** Gathered and stored with one indexing operation per batch; the time
  attention has no positions, so a window's steps sit at the front of their row with the rest masked.
- **The learner on a GPU of its own.** Its math is ~4% of an H100: it is not short of compute, it just must not make
  inference wait. A second H100 costs $3.95/h; two A100s (~$5/h for both) are the cheaper option, untested.

## What limits it now

The games' loop: ~58 ms a step from us-east sandboxes, ~72 from us-central (game, tunnel, actor, inference), so 512
games make ~7,000-7,300 steps/s. Inference has room (GPU 1 ~1/3 busy), and so does the learner: ~0.62 s of work an
update in a run (~13,000 steps/s), then it waits for chunks. The loop does not contain the learner: games never wait for it;
when it falls behind, chunks wait in its buffer and those more than 8 updates old are dropped. Open items:
[todo/](todo/README.md).
