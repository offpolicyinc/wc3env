# Representation: ids by what they are, choices by pointer

Unit types, abilities, items and buffs enter the model as learned lookups by id, and the argument head chooses among
every id in the game by a fixed output row each. Nothing a rare id's vectors learn carries to a similar id. Give each
id a vector made from what it is (its text and numbers), and choose abilities and items the way units are already
chosen: by scoring the ones available against the query.

## Where it stands (training/policy/model.py, features.py)

| | today | reads |
|---|---|---|
| Unit type, owner, order, items, buffs, abilities | learned tables by id (`type_emb`, `item_emb`, `buff_emb`, `ability_emb`) | a rare id's vector barely trains; a new hero or item starts from nothing |
| A unit's abilities | up to 5 ids summed into one vector (EmbeddingBag); each slot's level and cooldown as separate numbers | the sum loses which id is in which slot, so which cooldown belongs to which ability is left to slot order |
| The argument head | MLP to one logit per id in the game (`argument_head`), then `argument_choice[id]` added to the query for the later heads | three unrelated vectors per ability: having it (`ability_emb`), choosing it, having chosen it |
| Actor and target | query · each unit's key | already pointer form |
| Location | query · each of 64 patches, then a softmax over the patch's 256 cells | absolute only; one cell is map width / 128 |
| Numbers (x, y with sin/cos, hp, cooldowns, ...) | linear into the unit's vector | stay as they are |

## To do

**Ids by what they are**
- [ ] A static table, built once from the game's data and frozen: per id (and per level, for abilities), the
      text vector of its name and tooltip from a frozen text encoder, plus its fixed numbers (an ability's mana cost,
      cooldown length, range and target type at that level; an item's price; a unit type's base stats). A trained
      linear layer from it into the model's width. Only what never changes during a game goes here.
- [ ] The texts: written by an LLM from the game's data (and Liquipedia), one fixed template for every id and both
      games (kind, role, attack, special, abilities; for an ability: what it targets, its effect). No numbers in the
      text (they go in the numbers part). In-game tooltips as they are would embed by their style, not their meaning,
      and Brood War has almost none.
- [ ] The encoder: an open embedding model run once offline (Qwen3-Embedding 4B/8B, or what leads MTEB then); its
      size costs nothing at play time.
- [ ] Measure first: nearest neighbours of the text vectors (stuns near stuns, heals near heals; across games, Grunt
      near Zealot) before training on them. Wrong neighbours: fix the template, not the model.
- [ ] Keep a learned per-id residual, initialised at zero, so common ids can still learn what the text misses.

**Abilities and items as tokens**
- [ ] Each ability a unit has (and each item it carries) its own token: a trained layer of the static table's row
      at (id, the unit's current level), plus a trained layer of that step's own numbers (cooldown remaining, enough
      mana now, autocast on; an item's charges). Replaces the summed bag and the slot-ordered cooldown columns.
- [ ] The argument head as a pointer: query · each available token's key; the chosen token's vector added to the
      query for the later heads, in place of `argument_choice`. One vector per ability for having, choosing and
      having chosen it.
- [ ] Production, build, learn, buy: the same, over their candidates (unit types, buildings, skills, shop items).

**Locations relative to a unit**
- [ ] A move as anchor unit (pointer) x direction x distance, beside the absolute region and cell (also in
      [curriculum.md](curriculum.md)).

**Moving existing weights**
- [ ] The new heads and inputs don't load from current checkpoints: cloning again, or the trunk warm-started with
      the new tables and heads from scratch and a short cloning pass before RL. Compare the two on the cloning loss.

**Evaluate**
- [ ] Against a run of the current model from the same start: win rate by games played, and the share of casts by
      spell (rare spells used sooner?).
- [ ] A held-out hero or item in cloning: its logits before it has been seen once.

## References

- Vinyals et al. 2019, AlphaStar: pointer networks over units; entity tokens.
- Vinyals et al. 2015, Pointer Networks.
- Press & Wolf 2017, tied input and output embeddings.
- Reed et al. 2022, Gato; Raad et al. 2024, SIMA: one model across many games and tasks.
- Baker et al. 2022, VPT: actions recovered from unlabelled gameplay video.
