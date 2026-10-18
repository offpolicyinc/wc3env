# Checkpoints

The ones worth keeping: what each is, where it came from, how good it is. The files stay out of git (`.gitignore`);
this index is tracked. Every run's own files are on the Modal Volume `wc3-dataset` under `runs/<name>/`.

    python -m modal volume get wc3-dataset runs/cb9/last.pt checkpoints/cb9.pt

Use as `--init` (the KL's anchor) or `--start` (the weights to start from; a learner's `last.pt` brings its Adam
state too). Add a row when a checkpoint becomes a start or an anchor, or beats the one before it.

| file | Volume path | what | from | how good |
|---|---|---|---|---|
| `bc12.pt` | `runs/bc12/last.pt` | the clone: pro 1.29 replays, 60k steps | bc11 | every combat run descends from it |
| `cbe05.pt` | `runs/cbe05/last.pt` | bc12's policy with a critic fit to fights (policy frozen, stage 0.5, 0.5 h) | bc12 | the clone's play |
| `cb6.pt` | `runs/cb6/last.pt` | combat RL, lr 2e-5, UPGO, kl 0.005 | cb1 | ~0.62-0.70 at stage 0.7 |
| `cb7.pt` | `runs/cb7/last.pt` | combat RL, `--explore-min 0.001`, lr 2e-5 | cb6 | ~0.41-0.52 at stage 0.85 |
| `cb8.pt` | `runs/cb8/last.pt` | combat RL, lr 4e-5, fight channels' gamma 0.995 from its resume (update 2,200) | cb7 | ~0.55-0.62 at 0.85 (update 3,400) |
| `cb9.pt` | `runs/cb9/last.pt` | as cb8 from update 2,200, every channel's gamma 0.9999 | cb8 at 2,200 | ~0.58-0.67 at 0.85 (update 4,100); the latest |

Rates are training ladder games against Warcraft's attack-move at that handicap stage (`docs/training/combat.md`),
blocks of 200 from the learner's log; not evaluation games, so only a rough guide.

**The anchor until 2026-10-13:** a run started from another's `last.pt` with `--init` took that run's end as its anchor
too (`--start` did not exist): cb1's anchor was cbe05 (bc12's policy), cb6's cb1's end, cb7's cb6's, cb8's and cb9's
cb7's. Since then the anchor is whatever `--init` names, and the start is `--start`.
