# 3. Reward

The first proposal, kept for its reasoning. What the code does now, value by value and why: [8-decisions.md](8-decisions.md).

## The problem

A game lasts 10-30 minutes, i.e. 2,400-7,200 decisions at 4/s, and win/loss arrives once at the end.
A random policy with this action space never wins against anything, so from scratch the win/loss
signal is essentially always "lost": nothing to learn from.

AlphaStar solved this with human replays: supervised learning first (the agent already plays at
top-16% level before any RL), then RL with win/loss, a KL penalty keeping it near the human policy,
and "pseudo-rewards" for following a human build order. The paper's ablation shows RL without that
start fails. TStarBot-X confirmed it at smaller scale. Pluto's reward is unpublished; its value head
predicts win/loss/draw.

We have no replays of our own, so we need a different ladder to climb.

## Proposal

**Final reward: +1 win, −1 loss, 0 draw** (game hits the time cap, e.g. 25 minutes). This is the only
thing we ultimately care about, and the value head predicts it.

**Getting off the ground**, in order of cost:

1. **Rally on the gold mine** (see [1-actions.md](1-actions.md)): new workers mine by themselves
   (WC3's own behaviour), so even a random policy has some economy. Worker allocation beyond that is
   the network's job.
2. **Curriculum on the built-in AI.** The env already runs it (`control="computer"`, difficulty easy,
   normal, insane). Easy barely attacks, so early wins are reachable; move up when the win rate is
   above ~70%. mini-AlphaStar's follow-up used exactly this ladder to reach 93% against SC2's level-7
   bot on small compute.
3. **Shaping from the score fields**, annealed to zero. The header carries 25 score counters every
   step, so per-step rewards are differences, e.g.

   ```
   r_t = win/loss at the end
       + 0.002 * Δ(gold_mined + lumber_total) / 100
       + 0.01  * Δ(units_killed + 2*heroes_killed + 3*structures_razed)
       - 0.01  * Δ(own units lost, from death events)
   ```

   Multiply the shaping terms by a factor that goes 1 → 0 over the first N hours of training so the
   final policy optimises winning, not farming kills. Track win rate, not total reward.
4. **Cloning the built-in AI** (the replacement for human replays). The env can let the melee AI
   play *our* slot (`ai_agents`). Recording its games and inferring its actions from own units' order
   changes (a unit's `order_id`/target changes = an action was issued) gives supervised data in our
   action space for free, at 130x realtime. Pre-train on that, then RL with a KL term to the cloned
   policy (AlphaStar's recipe, with the insane AI in place of humans). The cloned policy starts at
   the insane AI's strength instead of zero.

   Human replays are the bigger prize but need a replay parser plus re-simulation through the engine
   with our hook to get observations, and replays must match our game version. Worth checking later,
   not first.

Options 1-3 are enough to see learning on the laptop. Option 4 is probably what makes the expensive
runs worth it; it's a few days of work (order-change → action inference is the fiddly part).

## Discount

γ = 0.999 per step (a half-life of ~3 minutes at 4 steps/s) with GAE λ = 0.95, or γ = 1 with TD(λ)
as AlphaStar did. Start with 0.999; win/loss 20 minutes away still reaches the value head through
bootstrapping.

## What not to do

- AlphaStar's build-order pseudo-rewards need target build orders from humans; skip unless we write
  a few by hand as "strategies" later (that's how TStarBot-X diversified its exploiters).
- Don't reward score totals without annealing: agents learn to farm creeps forever.
