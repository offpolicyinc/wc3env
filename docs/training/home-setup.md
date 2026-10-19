# Training at home: the 3090 Ti machine

For the agent setting up the home machine (RTX 3090 Ti 24 GB, Ryzen 9500 as the user put it, 64 GB RAM). Until
2026-10-17 all self-play ran on Modal (`machines.md`); this moves it onto one box. Read
[7-selfplay.md](7-selfplay.md), [combat.md](combat.md) and [the checkpoint index](../../checkpoints/README.md) first.

## Where things stand

- Training is in **combat mode**: fights on `(2)FlatArena.w3x` against Warcraft's attack-move, its units
  handicapped (they take hits as if they had a share of their hit points; the ladder's stages). The best is
  **cb9**: ~0.58-0.67 of fights won at stage 0.85 (training games, update ~4,100).
- cb9 = cb8 resumed from update 2,200 with every channel's gamma 0.9999 (cb8: the fight channels at 0.995). Its
  KL anchor is **cb7's end** (an accident from before `--start` existed; see the checkpoint index).
- New since cb9 and never used in a run: `--start` (start apart from the anchor, Adam carried), evaluation games
  (`--eval-games`, `--evaluate`, `--eval-stage`), the lr default 4e-5, `--channel-gamma` (default dealt/taken 0.995).
- What is planned next: [todo/](todo/README.md) (curriculum, learner tuning). What warcraftsim (Robin Kahlow's,
  same game, one 3090) learned on the same problem: [warcraftsim.md](warcraftsim.md) and his repo's
  `docs/reference/optimization-log.md`.

## The USB drive

The user's USB stick (the Ubuntu 24.04 installer) has `wc3-transfer/` beside the installer's files:

| path | what | goes to |
|---|---|---|
| `Warcraft III (Legacy)/` | the user's 1.29.2.9232 install, with `roc.w3k`/`tft.w3k` (licence files: keep them out of git and any upload) | e.g. `~/games/wc3` (`WC3_GAME_DIR`) |
| `wc3-training-handoff/checkpoints/cb9.pt` | the latest: weights, Adam state, league | `checkpoints/cb9.pt` (`--start`) |
| `.../checkpoints/cb7.pt` | cb9's anchor | `checkpoints/cb7.pt` (`--init`, to continue cb9 as it was) |
| `.../checkpoints/bc12.pt`, `cbe05.pt` | the pro-replay clone; its policy with a fight critic | the anchor if the user wants it back at the clone |
| `.../data/plans.npz` | the plan bank (pro plans the policy is given) | `runs/plans.npz` (actor.py's default) |
| `.../data/cb9-games.jsonl` | cb9's game log, for comparison | anywhere |
| `.../maps/*.w3x` | FlatArena (combat) and the 3x Echo Isles | `build/maps/` and `<game>/Maps/FrozenThrone/` |
| `.../generated/wc3hook.dll` | the hook, built 2026-10-18 from this commit (MSVC; not in git) | `src/wc3env/native/wc3hook.dll` |
| `.../wc3env-training.bundle` | the repo offline (or clone branch `training` from GitHub) | |

The stick is FAT32: executable bits are lost, so `chmod +x` scripts if needed. Everything else of every run is on
the Modal Volume `wc3-dataset` (`runs/<name>/`), profile `peter-nico`.

## Setup (Ubuntu, everything local: the games under Wine)

The game is 32-bit Windows, so it runs under Wine; the learner, inference and actors run in Linux Python. The
recipe that already works is the worker image: [docker/Dockerfile](../../docker/Dockerfile) and
[docker/install.sh](../../docker/install.sh). Do what they do on the host, not in Docker:

1. NVIDIA driver + CUDA-enabled torch in a Linux Python 3.11+ venv (`torch` cu128 wheel; `numpy zstandard wandb
   wandb-workspaces`; `pip install -e .[remote]`).
2. Wine: GE-Proton 10-34 (16-17% less CPU a game-step than the image's WineHQ stable, identical games), unpacked
   under `~/wc3/ge/`, with 32-bit GL (`libgl1:i386 libglx0:i386`). `WC3_WINE` = its `files/bin` and `WINEFSYNC=1`
   (wine_workers adds what Proton's launcher sets for the games). A prefix GE's Wine made (`~/.wine-wc3-ge`),
   Windows Python 3.11 in it at `C:\Python311` with numpy 1.26 (not 2: GE lacks `crealf`) and this repo's `wc3env`
   (+ zstandard), and the game folder linked as `drive_c/wc3`. All of it is in `~/.config/wc3env/wine-env.sh`;
   details in [warcraftsim.md](warcraftsim.md). A virtual display (Xvfb, as `docker/with-display.sh`) unless the desktop's is used.
3. The hook from the stick into `src/wc3env/native/`, the maps into the game's `Maps/FrozenThrone/` and `build/maps/`.
4. `.env` from `.env.example`: `WC3_GAME_DIR` as Wine sees it. W&B: `wandb login` (project off-policy-inc/wc3).
   **Never print `.env`**: it holds API keys.
5. Check the environment alone first: `python -m wc3env.vector --wine ...` (see its `--help`) or `docker/smoke.py`'s steps.

The actors run their games under Wine with `run.py --wine` (`wine_workers()`, a prefix per two games; each actor its
own prefixes). The learner and inference share the GPU through NVIDIA MPS: a user service running
`nvidia-cuda-mps-control -d` with its pipe in `/tmp/nvidia-mps`, so every CUDA process uses it (+11% decided
steps/s, [9-benchmarks.md](9-benchmarks.md#home-3090-ti-2026-10-18)). Under it the learner gets 65% of the GPU's SMs
(`run.py --learner-gpu-share`), so inference's batches are not stuck behind its kernels (+10%, rsi/log.md E15).

torch.compile works on Linux (Triton ships with torch), as on Modal.

## Continuing cb9

```
python -m training.selfplay.run --init checkpoints/cb7.pt --out runs/selfplay/cb10 --mode combat \
    --actors 4 --games 8 --ai-games 8 --eval-games 1 -- \
    --start checkpoints/cb9.pt --start-stage 0.85 --warmup 0 --lr 4e-5 --kl 0.005 \
    --explore-floor 0.005 --explore-min 0.001 --upgo 1 --channel-gamma= \
    --evaluate checkpoints/cb9.pt --eval-stage 0.85
```

- These are cb9's flags (`--channel-gamma=` empty = every channel 0.9999). Its launch on Modal was
  `selfplay start --name cb9 --mode combat --init runs/cb7/last.pt --games 32 -- --hours 1 ...` with 16 actors.
- `--init checkpoints/bc12.pt` instead puts the anchor back at the clone (as cb1 had); a choice for the user.
- Check `run.py --help` and `learner.py --help` for the flags' current names before launching; verify, don't guess.

## Sizing the run (the first job)

At home the limit will be the CPU that runs the games (warcraftsim: games took 13 of 16 cores), not the GPU.

1. Start at 4 actors x 8 games; read the log's steps/s and the actors' `ms:` breakdown (games vs inference).
2. Raise games until the CPU is ~90% busy or steps/s stops rising; then actors (warcraftsim gained 13% from more
   actor processes with fewer games each: one Python GIL per actor).
3. Note the numbers in [9-benchmarks.md](9-benchmarks.md) (a "home 3090 Ti" section). Modal did ~6,500 decided steps/s.
4. The learner and inference share one GPU here (on Modal they had one each): MPS on, and the learner's `--micro 4`
   (or it runs out of memory beside inference).

## The old 1080 Ti machine

Not as a second GPU in this box (Pascal: no tensor cores, newer CUDA/torch builds drop it). Useful as a second
machine: evaluation games over the LAN, or more Wine/Windows games for the actors.

## Rules the user has set

- Concise docs, high level, with references; don't add work the user didn't ask for.
- Check configs and numbers (W&B, logs, code) before stating them.
- Checkpoints stay out of git; add a row to `checkpoints/README.md` when one becomes a start, an anchor or the best.
- Commit only when asked; the repo is public.
