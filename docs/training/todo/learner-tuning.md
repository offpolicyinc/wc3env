# Learner tuning: learning rate, KL control, sample reuse

The learner takes small steps and sits idle most of each update. Experiments to find out whether it can learn more
per game played.

## Where it stands (cb9, update 4,100)

| | value | reads |
|---|---|---|
| `--lr` | 4e-5 | from W&B: cb1-2 1e-5, cb3-7 2e-5, cb8-9 4e-5 |
| `update_kl` | ~0.002 | each update moves the policy very little |
| `clip_fraction` | ~0.06 | PPO's clip barely binds |
| `actor_kl` | ~0.005 | the actors' lag is a small part |
| `--epochs` | 1 | each chunk trained on once |
| gradient steps per update | 4 (128 chunks / minibatch 32) | |
| update time | 1.88 s, of it 1.26 s `time_wait`, 0.46 s `time_train` | the learner is ~2/3 idle, waiting for games |

The data comes in slower than the learner can use it, so every pass it skips is free.

## Experiments

Each against a baseline run from the same checkpoint and stage. Compare win rate by **games played** (sample
efficiency) and by **wall clock**, alongside `update_kl`, `clip_fraction`, `kl_to_clone`, the critic's explained
variance and the order-type shares.

**Learning rate**
- [ ] `--lr` 4e-5 (baseline), 1e-4, 2e-4. No clean comparison exists yet: cb8 (4e-5) started from cb7's end, not beside it.

**Adaptive learning rate from the KL**
- [ ] Hold `update_kl` near a target: the learning rate times 1.5 when below target/2, divided by 1.5 when above
      2x target (rl_games' adaptive schedule; PPO's adaptive-KL variant). Targets 0.005, 0.01, 0.02.
- [ ] Stop an update's passes early when its KL passes ~1.5x target (Stable-Baselines3's `target_kl`).

**Sample reuse (the learner's idle time)**
- [ ] `--epochs` 1 (baseline), 2, 4: twice the passes costs ~0.46 s more of an update that waits ~1.26 s.
- [ ] More, smaller gradient steps: minibatch 32 -> 16 (8 steps an update).
- [ ] Extra passes for the critic only (Phasic Policy Gradient): the value gets more fitting, the policy stays as it is.
- [ ] Chunks kept and replayed for a few updates, weighted by importance (V-trace/IMPACT), if extra epochs alone help.

**Watch for**
- Reused chunks are more off-policy each pass: `clip_fraction` and `actor_kl` rising, the KL to the clone running away.
- A larger step can drop rare order types (casts, moves) faster: watch the order-type shares, not only wins.

## References

- Andrychowicz et al. 2021, What matters in on-policy RL (epochs, learning rate, clipping at scale)
- Engstrom et al. 2020, Implementation matters in deep policy gradients
- Schulman et al. 2017, PPO (adaptive-KL variant); rl_games' KL-adaptive learning rate; Stable-Baselines3 `target_kl`
- Cobbe et al. 2021, Phasic Policy Gradient
- Luo et al. 2020, IMPACT (sample reuse in distributed PPO); Espeholt et al. 2018, IMPALA / V-trace
- Hilton et al. 2022, Batch size-invariance for policy optimization
