# 7. Self-play

RL from the cloned policy (training/selfplay/). Actors, inference processes and a learner, joined over TCP, so the
same code runs on one laptop or on Modal. On Modal (`training/cloud/selfplay.py`): one container pinned to us-east
with two H100s runs 16 CPU-only actors, 2 inference processes on GPU 1 and the learner on GPU 0; the games run in 8
Wine sandboxes (64 games each), made again when they land on another continent. GCP VMs (`training/cloud/gcp.py`)
were tried for self-play and still run evaluations and recordings. The whole pipeline with its numbers:
[selfplay-anatomy.html](selfplay-anatomy.html); why it is laid out this way: [10-actor-design.md](10-actor-design.md).

```
sandboxes ×8 (Wine games)          container (us-east, 2 H100s)
  games ──obs + features──►  actors ×16 (CPU) ──inputs───► inference ×2 (GPU 1)
        ◄──commands────────         │         ◄──choices──        ▲
                                    │ 64-step chunks              │ weights, pulled every 1 s
                                    ▼                             │
                             learner (GPU 0): PPO + KL to reference; the league
```

## Pieces

| File | Does |
|---|---|
| policy/agent.py | Plays many seats in one batch per step: features, memory and attention window per seat, the heads in order under the masks and head_rules(); keeps what learning needs (choices as indices, masks, log-probability, critic value) |
| selfplay/inference.py | Decides for the actors on the GPU: joins requests that arrive within 2 ms into one batch per policy, runs the compiled step as a CUDA graph, keeps each game's memory; pulls the learner's weights every 1 s |
| selfplay/actor.py | Runs games (wc3env.vector), chooses opponents, records each learning player in 64-step chunks (the attention windows), ships them |
| selfplay/learner.py | Batches chunks, PPO, value warm-up, snapshots, the league; serves weights |
| selfplay/league.py | Opponents: the ladder (the built-in AI easy, normal, insane, each first handicapped: its units take hits as if they had 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85 of their hit points, then full; the next stage or rung at 80 wins in a block of 100 games), itself half the self-play games, else snapshots by PFSP; the games' time limit, 15 minutes growing by 5 (to 30) |
| selfplay/reward.py | ±1 for a win or a loss; −1 at the time limit, for both players and against any opponent; shaping channels (build, units, upgrades, dealt, taken) faded by the win rate: [rewards.md](rewards.md) |
| selfplay/transport.py | multiprocessing.connection with an auth key: chunks in, weights and snapshots out, results and actor logs in |
| cloud/gcp.py | GCP VMs: evaluate checkpoints against the AI, one VM each; record games of the AI against itself as training data; self-play on VMs (tried, not used now) |
| cloud/scripts.py | The shell scripts the worker image runs: actors, evaluation games, recordings |
| cloud/pull.py | Copies a bucket folder (recorded data) into the Modal Volume, from inside Modal |
| cloud/selfplay.py | Self-play on Modal: the two-H100 container (actors, inference, learner) and the games sandboxes, joined by TCP tunnels |
| play/evaluate.py | Many games against the built-in AI at once (--handicap: its hit points as a share, as a ladder stage), one JSON line per game as it ends; the summary table |
| play/diagnose.py | Army, orders at the enemy and time at the enemy's base per 5 minutes, pros (data set files) against a checkpoint |

## Decisions (each with the evidence behind it)

- **Start from cloning, keep one anchor**: a KL to a reference policy on the agent's own states (VPT, 0.2), beside a
  small entropy bonus (AlphaStar's: 1e-4 × each used head's entropy over ln(its options); with the KL alone sp7 halved the action type's entropy).
- **Value warm-up**: the first 50 updates train only the critic (SCC, Wolczyk; warcraftsim's run collapsed
  without it).
- **A critic that sees the opponent**: the core plus the opponent's scalars (resources, food, research, unit
  counts), training only (AlphaStar). Against the built-in AI too: its slot is an observed AI slot.
- **Small learning rate** (4e-5; VPT's 2e-5 until cb8), clip 0.2 with a dual clip of 3 on negative advantages (Ye et al. 2020:
  sp2 collapsed on one stale sample with a ratio of 525), 1 epoch, γ 0.9999, λ 0.95.
- **Start apart from the anchor**: `--init` is the KL's anchor, `--start` the weights a run begins from (another
  run's `last.pt`, its Adam state carried; warcraftsim's forks with a fresh Adam lost ~10 points in their first 20
  minutes). Before 2026-10-13 a run from another's `last.pt` took it as its anchor too.
- **Evaluation games** (`--eval-games` per actor, `--evaluate` checkpoints): games against the ladder's rung at one
  fixed stage (`--eval-stage`) that nothing learns from, the learning seat played by the current policy and each
  checkpoint in turn, each round of them on the same draws (a fight's armies and places). Training games can't
  compare policies: the weights change mid-game, the stage moves, the draws differ (warcraftsim's training games read
  52% where evaluation games read 36% for the same policy). The log's `eval <stage>:` part and `eval_won_<policy>`.
- **Plans**: 90% of players get a pro plan of their matchup from the plan bank, as cloning saw.
- **Exact log-probabilities**: the actor samples with the same head flow and masks the learner scores with;
  training/tests/test_selfplay.py checks they agree on a real game (to 2e-3).

## Checks so far

- Unit tests: GAE by hand, league promotion and PFSP, transport, head rules against the data set's labels,
  actor and learner log-probabilities on a real game.
- Local run (laptop, untrained model): 20 updates, 19 games, snapshots joined the league and were played,
  a failed game restarted on its own.
- Modal: an actor sandbox played 2 games at ~55 game steps/s with CPU inference, pulling weights as they changed.
- sp3 (3 hours, 760 updates, from sp2's update 1050), evaluated on GCP, 12 games of 30 minutes per checkpoint
  against the easy AI: 0 wins in 83 games, half defeats, half the time limit; gold per minute rose from 273 to
  408 but no checkpoint razed a building. The log's "win rate" of 0.20 was draws counted half.
- sp4 (3 hours on GCP, an on-demand L4): 0 wins; it massed workers (39 at minute 10) and banked 3,200 lumber it never
  spent, with 5 grunts and 6 move orders: the old shaping (gathering, food in use) was being farmed. The clone it
  started from (bc6w) has 6-12 workers and 0-4 army units at minute 10, against a pro's ~20 and ~8.

## Running it

```
python -m training.selfplay.run ...                                         # one machine
python -m training.cloud.selfplay start --name sp13 -- --hours 1            # Modal; log / stop --name sp13
bash training/cloud/guard.sh sp13 70 &                                      # its automatic stop
python -m training.cloud.gcp eval --name e1 -- gs://.../a.pt runs/b.pt      # then eval --name e1 --collect
python -m training.cloud.gcp record --name r1 --games 400 --vms 8           # then training.cloud.pull
python -m training.play.evaluate ckpt.pt --games 12 > e.jsonl              # locally; --table e.jsonl
python -m training.play.diagnose --pros runs/dataset3 --checkpoint ckpt.pt # how it plays against pros
```

What is left: [todo/](todo/README.md).
