# Next model: changes that need retraining

Ideas that change the network's inputs, outputs or shape, so current checkpoints (bc12, cb9) don't load into them:
each needs a cloning pass again (or a warm-started trunk and a short cloning pass) before RL. A retrain costs a
cloning run plus RL back to cb9's level, so these are batched into one next model rather than tested one by one,
and each is first checked where it is cheap (the cloning loss, an offline probe) before it joins the batch.
Moved here from [ideas.md](ideas.md) on 2026-10-18; ideas.md keeps what improves training of the current model.

| # | Idea | Moves | Cheap check first | Rests on | Status |
|---|---|---|---|---|---|
| M1 | Positions in the time attention (rotary); today it sees the last <=64 steps as a set | wins | cloning loss | Su 2021 (RoPE) | open |
| M2 | Larger model (~50M, bf16; the GPU's math is mostly idle) | cloning loss, wins | cloning loss vs bc10's first | Kaplan 2020; Hilton 2023 (RL scaling) | open |
| R1 | Moves relative to a unit (anchor x direction x distance); cb9 moves on ~0.2% of steps | move share, wins | share of pro moves it can express | [curriculum.md](../todo/curriculum.md) | open |
| R2 | Ids by what they are: frozen text vectors + base numbers + a learned residual from zero | rare-id learning | nearest neighbours of the text vectors | [representation.md](../todo/representation.md); RTFM (Zhong 2020); EMMA (Hanjie 2021) | open |
| R3 | Abilities and items as tokens; the argument head as a pointer | cast share, wins | held-out hero or item in cloning | [representation.md](../todo/representation.md); Pointer Networks (Vinyals 2015) | open |
| X1 | Brood War through Pluto, then WC3: text-aligned ids across games | WC3 start | Pluto's terms; pluto-re's outputs | [cross-game.md](../todo/cross-game.md); MAGENTA (Wang 2024) | blocked: R2, R3 |
