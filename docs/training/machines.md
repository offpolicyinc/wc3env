# What runs where

Three jobs. All on Modal (profile peter-nico). Wine (the game) only runs in Modal's VM sandboxes, and those
cannot have a GPU. So anything that needs both games and a GPU is two machines joined by a network tunnel.

## 1. Cloning (learning from pro replays): one GPU machine, no games

```
  Modal volume wc3-dataset            one H100 container
  (converted replays)  ───────────▶  training/train/cloning.py
                                      reads replays, trains the model
                                      writes runs/<name>/last.pt
```

No game is played. `python -m training.cloud.train start --name bcN`

## 2. Self-play training: one GPU container + 8 game sandboxes

```
  ONE CONTAINER (us-east, 2 x H100, cheap CPU cores)          8 SANDBOXES (Wine, CPU only)
  ┌───────────────────────────────────────────────┐          ┌──────────────────────────┐
  │ learner  (GPU 0)  updates the model           │          │  Warcraft games          │
  │    ▲ games' steps + rewards                   │          │  (each: our model as     │
  │    │                                          │  tunnel  │   player 0, and player 1 │
  │ 16 actors (CPU)  run the game loop:   ◀───────┼──────────┼─▶ = our model, an old    │
  │    send orders, get observations, rewards     │          │   copy, Warcraft's AI,   │
  │    │ ▲                                        │          │   handicapped or not)    │
  │    ▼ │ "what should I do?"                    │          └──────────────────────────┘
  │ inference (GPU 1)  the model picks orders     │
  └───────────────────────────────────────────────┘
```

- Games run in sandboxes because only they run Wine. Everything else is in the container, whose CPU is 3x cheaper.
- Each actor plays some games against the ladder (Warcraft's AI at easy, normal, insane, each first handicapped) and the
  rest against itself or old copies of itself. Against Warcraft's AI, player 1 is played inside the game itself:
  no model, no GPU for that side.
- `python -m training.cloud.selfplay start --name spN ...`

## 3. Evaluating a checkpoint against Warcraft's AI: one sandbox

```
  ONE SANDBOX (Wine, 8 CPU cores, no GPU allowed)
  ┌─────────────────────────────────────────────┐
  │ 20 Warcraft games  ◀──▶  evaluate.py        │
  │ (player 1 = Warcraft's     the model on the │
  │  AI, inside the game)      CPU              │
  └─────────────────────────────────────────────┘
```

- Simple, but the model on CPU is the slow part: 533 ms of every 544 ms round at 20 games (sp34, 2026-10-10), so a
  15-minute game takes ~33 minutes.
- A GPU would need setup 2's split (model in a GPU container, games in a sandbox, a tunnel between); not built for
  evaluation yet.
- `python -m training.bench.modal_eval runs/sp34/last.pt --label sp34 --games 20 --races orc orc --minutes 15`
