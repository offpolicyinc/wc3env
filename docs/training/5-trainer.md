# 4. Trainer

## The loop, plainly

1. Actors play many games with the current weights, recording for every step: features, the sampled
   action (all heads), its log-probability, the masks, the reward, the memory state (SSM states + attention window) at chunk starts.
2. Steps are cut into chunks of 64 consecutive steps from one game (16 seconds of game); an actor sends a player's
   chunks four at a time, as one segment (64 seconds; the last one of a game shorter).
3. The learner's batch process computes each segment's advantages over the whole segment as it arrives (a step sees
   real rewards to its segment's end, then the critic's value: OpenAI Five's 256-step segments and 16-step samples),
   then draws each batch of 128 chunks at random from the oldest 512, so a segment's chunks train in different
   batches. The learner takes a few gradient steps on the batch and publishes new weights.
4. Inference picks up the new weights every 1 s and the games keep playing; they are never paused for training.

Because actors keep playing while the learner trains, some chunks were produced by weights a few
updates old (more since segments: a chunk waits for its segment to fill; --max-lag 24, was 8). PPO's clipping (or V-trace, AlphaStar's choice) corrects for that. TStarBot-X measured
PPO and V-trace as equally good; PPO is simpler and better known.

## Cloning first

training/train/cloning.py. 32 lanes, each reading one game from start to end, 64 steps per batch; each lane's
memory carries over to its next chunk (gradients stop at the chunk's start) and resets when it starts another
game. Loss: each head's cross-entropy on the pro's choice, plus 0.1 x the value head's on the game's result.
5% of replays are held out for accuracy. On an H100: ~15,500 steps/s with the step transformer packed (real tokens only)
and its blocks compiled, as the self-play learner (~6,100 padded: about half of the padded tokens were padding,
pro2 + bots3k, 2026-10-10); ~5 minutes and about $0.35 per pass over the 4.9M steps (the A100s, L40S and A10 cost the same or more per pass, and are slower).

## Algorithm

**PPO**, adapted to the multi-head action:

- log-prob of an action = sum over the heads that were used (skipped heads contribute 0);
- ratio and clipping on that sum (AlphaStar clips per head; try it if updates look unstable);
- value: separate head on the core output, trained on GAE / TD(λ) returns;
- entropy bonus 1e-4 × the heads' entropies a step uses, each over ln(its options) as AlphaStar's (0 to 1), beside a KL of 0.005 to the clone (0.02 until 2026-10-12);
- recurrent: re-run the memory core over each chunk from its stored start state (64 steps).

Add **UPGO** (AlphaStar) as a second policy-gradient term if learning from sparse wins is slow: it
imitates stretches where things went better than the value predicted. It's 20 lines.

During RL the policy is kept near the clone by a KL to it on the agent's own states (AlphaStar, VPT), not by a
cloning loss on pro replays: that was dropped while we check which self-play systems kept one (8-decisions.md).

Settings (training/selfplay/learner.py; why each: [8-decisions.md](8-decisions.md)): AdamW lr 4e-5, clip 0.2 with a
dual clip of 3, 1 epoch per batch of 128 chunks in minibatches of 32, value coef 10 (AlphaStar's, DI-star's; 0.5 until sp37), entropy 1e-4 (normalized per head), KL 0.005,
γ 0.9999, λ 1 for the policy's advantages and 0.9 for the critic's targets (AlphaStar's: V-trace ~1, TD(λ) 0.8; one λ 0.95 for both until sp40), grad-norm clip 1.0. Each batch's advantages rescaled to mean 0, spread 1 (the PPO libraries' default; sp40 ran them raw, as AlphaStar and DI-star, and its spread of 0.01-0.09 left the policy loss small against everything else).

## Opponents, in phases

| Phase | Opponent | Move on when |
|---|---|---|
| 0. (optional) Pre-train | none: supervised on cloned built-in AI games ([4-reward.md](4-reward.md)) | matches the AI's actions well |
| 1. Curriculum | built-in AI easy → normal → insane | >70% vs insane |
| 2. Self-play | 35% current self, 50% snapshots, 15% built-in AI | — |
| 3. (later) Exploiter | a second agent trained only to beat the main one | if the main agent gets exploitable |

Phase 2's snapshot mix is AlphaStar's "prioritised fictitious self-play" (PFSP) cut down: save a frozen
copy every few hours; pick opponents with weight (1 − win rate against them)², so the agent plays
more against snapshots that still beat it. Keeping built-in AI games in the mix is a cheap guard
against forgetting how to beat a normal opponent.

Pure self-play without a pool tends to cycle (A beats B beats C beats A); the pool is what stops it.
AlphaStar's full 12-agent league is out of budget and TStarBot-X shows it isn't needed to get strong.

## Evaluation

Every hour: 50 games against insane AI (win rate), and a round-robin between recent snapshots (Elo).
Watch a replay now and then; numbers miss degenerate strategies.

## Code shape

```
training/
  policy/      what the agent sees and does
    actions.py     the action space; translator to wc3env commands
    features.py    what the features are (columns, ids), the map's data, the map drawn from the rows
    masks.py       the argument vocabulary and orders
    native.py      features and masks of an observation, from their C (wc3hook/features.c, masks.c)
    model.py       the network and the cloning loss
  replays/     pro replays -> training data
    convert.py     a replay's commands -> actions
    dataset.py     replays -> files (features, labels, masks)
    check.py, check_dataset.py   checks over replays and files; type_orders.json
  train/       the trainer
    loader.py      files -> [B, T] chunks (the map's ingredients; the model draws the map on the GPU)
    cloning.py     behavioral cloning: lanes reading whole games with memory, held-out accuracy, checkpoints
  selfplay/    RL: actor.py, inference.py, learner.py, league.py, reward.py, transport.py ([7-selfplay.md](7-selfplay.md))
  cloud/       modal.py: the data set and checks on Modal; train.py: cloning on an H100; gcp.py: evaluation and recordings on GCP;
               scripts.py: their shell scripts; selfplay.py: self-play on Modal; pull.py: bucket to Volume
  play/        evaluate.py: games against the AI; diagnose.py: army and attacks against pros; vs_ai.py, watch.py
  tests/
```

For RL, actors, inference processes and the learner talk over TCP (`selfplay/transport.py`), on one machine or across
machines ([10-actor-design.md](10-actor-design.md)).
