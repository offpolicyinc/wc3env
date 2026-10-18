# Cross-game: Brood War through Pluto, then Warcraft III (reach)

WC3 has little imitation data (1,350 pro games). Pluto (tscmoo, Brood War, won CoG 2026) plays as many strong games
as we run. The aim: a better WC3 bot by learning a lot of Brood War first, as a gamer brings fundamentals to a new
game. Both games in one schema (units as tokens, numbers in comparable units; [representation.md](representation.md)),
each id's vector made of three parts:

- its text (name, tooltip) through a frozen text encoder and a trained layer: what it is, the same words in both
  games. Frozen, so the two games' words stay aligned;
- its base numbers (hp, damage, range, cost) through a trained layer;
- a learned per-id residual from zero, small L2: what play teaches about this id that its text misses. Rare ids
  stay near their text; common ones learn their own.

What play teaches about stuns moves the shared layers, so every stun in either game; that is the channel from Brood
War to WC3. After [representation.md](representation.md) and [curriculum.md](curriculum.md).

## Why it might work now

- Multi-game agents so far gave the games no common meaning: Gato maps each game's actions to arbitrary tokens;
  multi-game Decision Transformer and SIMA share pixels and buttons. Transfer was small; Atari games share little.
- Text descriptions of entities have only been tried in grid worlds (RTFM, Zhong 2020; EMMA / Messenger, Hanjie
  2021): agents act on what a text says an entity does, but nothing there bears on real games. Entity tokens
  transfer across unit mixes within one game (UPDeT, Hu 2021).
- Recent: NitroGen (2026) clones 40k hours of gameplay video over 1,000+ games (pixels to gamepad) and does up to
  52% better than from scratch on unseen games, but is weak on RTS and MOBA. MARL-GPT (AAMAS 2026) is one
  transformer across SMAC, Google Research Football and POGEMA, offline, competitive per game; no unseen-game result.
  SIMA 2 (2025) improves in new games by self-play with Gemini setting tasks and rewards.
- MAGENTA (Tencent / NTU, DAI 2024) is the closest: entity tokens, a tokenizer and action heads per game (learned,
  nothing aligned across games), one shared transformer. Its Honor of Kings transformer under new tokenizers and
  heads learns SMAC and Neural MMO faster than from scratch (claimed 50-100% faster fine-tuning). One seed in HoK,
  three elsewhere, mixed against SMAC baselines, no zero-shot. That is our arm B with per-game input and output;
  it says a shared trunk carries something, not whether aligned meaning carries more.
- Nobody has tried text-described entities across two real RTS games. Text aligns what things are; the numbers
  (ranges, attack periods, damage and armor systems) still differ, so expect a better start, not zero-shot strength.

## What transfers, what doesn't

| Likely | Unlikely |
|---|---|
| Focus fire, pulling hurt units, kiting, surrounds, keeping the army together | Heroes, levels, items, most spells (Brood War has few casts, no heroes or items) |
| Worker saturation, no supply block, production kept busy | Timings, build orders, unit counters, creeps, upkeep |

## Pluto as the teacher

| | Data | Target | Work | Feasibility |
|---|---|---|---|---|
| 1 | Pluto's games | its sampled actions | Brood War harness, Pluto at scale, states written in the shared schema | clear |
| 2 | Pluto's games | its logits per head | hook the inference exe (pluto-re documents its inputs and outputs) | likely |
| 3 | the student's games (on-policy, as DAgger / kickstarting) | its logits per head | Pluto's DLL as an advisor in the student's game: it sees the student's player, its BWAPI commands are intercepted, its logits read | hardest |

- Logits teach more per state than sampled actions (second choices, uncertainty; lower-variance gradient); the
  student's own states fix the compounding errors of cloning the teacher's games. 1 is enough to start.
- Same action shape as ours (type -> argument -> anchor + N nearest -> 8x8 + 16x16 cell), so the KL is per head
  along the student's choices; argument logits at each candidate token's id, renormalised over the candidates both allow.
- Legality: the student's masks from BWAPI's checks (`canIssueCommand`, `canTrain`, `canUseTech`, `canBuildHere`).
  Pluto's own mask comes with its hooked logits (entries at -inf); log where the two disagree. No need to
  reverse-engineer its predicates.
- A unit-relative move ([curriculum.md](curriculum.md)) has no Pluto counterpart: distil location only on the
  absolute path, or Pluto's point as a hard label.

## To do

**Check first**
- [ ] Pluto's release terms: whether training on its outputs is allowed.
- [ ] pluto-re: which inference inputs and outputs are marked verified; whether per-head logits can be read.
- [ ] Brood War at speed: BW 1.16.1 + BWAPI under Wine (as our WC3 sandboxes) against OpenBW headless.

**Shared schema**
- [ ] The representation.md changes in WC3 first (text-embedded ids, abilities and items as tokens, pointer argument).
- [ ] Numbers in comparable units in both games: hp as a fraction and as seconds to die under typical damage, ranges
      and speeds per second relative to unit size, damage per second, positions in unit sizes.
- [ ] Brood War's static table: unit, building, tech and spell texts and base numbers from its data.

**Data**
- [ ] Tier 1: Pluto against itself and the usual bots, recorded in the shared schema; then tier 2, then 3.
- [ ] Optionally human Brood War replays as well (StarData, Lin et al. 2017).

**The experiment**

The hypothesis is how a gamer learns a new game: knowing roughly what a unit does from what it is (its
description), fundamentals carried from games already played, both refined by playing. The two need each other:
the text encoder knows "stun" only as a word until some game's play has grounded it, and fundamentals carry to a
unit only if it is recognised as the same kind of thing. So two factors, crossed:

| | learned ids | text ids |
|---|---|---|
| no Brood War pretraining | A: from scratch | T: the description alone ("read the manual") |
| Brood War pretraining (Pluto) | B: fundamentals alone (as MAGENTA) | C: both, the experienced gamer |

- [ ] The main path is C: Brood War from Pluto, then WC3 cloning and RL. Against the current pipeline (bc12's start)
      on WC3 combat mode: win rate by games played, and its first games before any WC3 RL.
- [ ] The other arms only if C's result needs explaining (which half helped).
- [ ] Micro first: Brood War combat scenarios against Pluto's play before whole games.

## References

- Reed et al. 2022, Gato; Lee et al. 2022, Multi-game Decision Transformers; SIMA Team 2024.
- Wang et al. 2024, MAGENTA: Multi-agent Multi-game Entity Transformer, DAI '24, doi 10.1145/3719545.3759428.
- NitroGen, arXiv 2601.02427 (2026); MARL-GPT, arXiv 2604.05943 (2026); SIMA 2, arXiv 2512.04797 (2025);
  LLM4Teach, arXiv 2311.13373 (IJCAI 2024, an LLM as kickstarting teacher).
- Zhong et al. 2020, RTFM; Hanjie et al. 2021, EMMA / Messenger; Hu et al. 2021, UPDeT.
- Schmitt et al. 2018, Kickstarting; Ross et al. 2011, DAgger; Agarwal et al. 2024, GKD (on-policy distillation).
- Baker et al. 2022, VPT.
- Pluto: <https://github.com/tscmoo/pluto>; reconstruction: <https://github.com/hwkim3330/pluto-re>.
