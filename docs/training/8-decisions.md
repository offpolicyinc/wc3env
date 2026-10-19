# 8. Decisions

Every non-obvious choice in cloning and self-play, one per line: the value, why, and what it rests on (a paper, a
bot, or one of our own runs). When a value changes, change its line and say what changed it. Run names: bc1-bc6w
are cloning runs, sp1-sp3 self-play runs; "Tried and dropped" at the end keeps what we undid and why.

## Cloning (training/train/, training/replays/)

| Decision | Value | Why | Rests on |
|---|---|---|---|
| Data | 1,350 pro 1.29 replays, both players | The only human play in our action space; one game gives two players' views | AlphaStar, Pluto: imitation first |
| A walking worker's building | An own worker's build order (its order id is the building's type) fills its `unit_queue`, the slot a structure's training uses | Until placed, nothing else shows a building is coming (gold is spent on placement): bc9 built 2-3 Altars, Graveyards and Tombs and never a Crypt. Enemy workers' build orders stay hidden, as from a player | Pluto's train-type column; AlphaStar's per-unit order ids; bc9 |
| AI games kept as replays | Recording saves each game's .w3g (and its startup options) beside the data; `dataset <replays folder>` writes the data again from them, identical to the live data | Rebuilding the data for a feature change then costs a conversion, not new games: the first 3,000 AI games kept only the data and had to be played again | bc9 |
| Timing | One decision per 250 ms step, "noop" a choice like any other | Simplest; the clone acted 39% of steps against pros' 38%, so no separate timing head was needed | Pluto; bc6 |
| Rally bursts | Clicks within 2 s collapse to the last one | Pros spam rallies to warm up; copying the spam cancelled builds | bc5 replay review |
| Shift-queue | Its own head (queued yes/no) on move, attack, smart, build, cast | Pros queue builds; without it the clone's builds cancelled each other | AlphaStar has the same head |
| Plan input | Heroes, first 20 builds, unit mix of the player's own game; 10% of games get the empty plan | Tells the policy which strategy to play, and self-play can then pick strategies; the empty plan keeps a default | AlphaStar's z (build order and unit counts) |
| The enemy's start location | A twelfth map channel: the start locations with none of the player's buildings within 6 cells (on a 2-player map, the other base); computed when the map is drawn, so no data rebuild | Every player knows where the enemy starts (StarCraft's API gives AlphaStar the enemy's start locations); with two identical start marks bc7 never once walked to an idle player's base in 4 games, and sp5 beat the idle player 1-4% of the time | bc7, sp5 |
| Value head | Sees the trunk detached, weight 0.1 | The value loss had been pulling the shared trunk toward predicting the winner | bc5 |
| Winners fine-tune | bc6w: bc6 trained further on winners' games only | Copy the side that won | AlphaStar's fine-tune on winners |
| Games of Warcraft's AI as data | The insane AI on both sides (`--bots`), its script orders plus its captains' army orders (the engine's moves and attacks to several units at once or to a point; a single unit attacking a unit on the engine's say is auto-acquire and left out) | The clone (bc6w) builds almost no army; the AI builds pro-sized ones and sends them. Without the captain orders an AI game had ~10 army orders, with them 300-500 | warcraftsim clones the AI too; our probe of one AI game |

## The RL algorithm (training/selfplay/learner.py)

| Decision | Value | Why | Rests on |
|---|---|---|---|
| Algorithm | PPO, asynchronous actors, 64-step chunks | Simple and stable; chunks match the attention window | OpenAI Five, AlphaStar (V-trace instead) |
| Learning rate | 4e-5 | Small steps from a cloned start; 2e-5 until cb8 | VPT; cb8-cb9 |
| Clip | 0.2, plus dual clip 3 on negative advantages | sp2 collapsed when one stale sample had a probability ratio of 525 and a negative advantage: plain PPO does not bound that case | Ye et al. 2020; sp2 |
| Epochs, batch | 1 epoch; 128 chunks per update in minibatches of 32 | More chunks per update means less weight lag per chunk; 128 at once did not fit the GPU; one epoch lets the learner keep up with the games | sp1 |
| Discount | γ 0.9999 per step (a win 15 minutes away still counts 0.70), GAE λ 0.95 | 0.999 let little of a 20-minute game's result reach early decisions; 0.9997 (0.34 at 15 minutes) made losing at minute 10 (−0.49) cost more than running out the clock (−0.34): a reason to stall. At 0.9999: −0.79 against −0.70 | sp6 |
| Entropy bonus | AlphaStar's: 1e-4 × the entropy of each head a step uses, over ln(its options) (0 to 1), beside the KL to the clone | AlphaStar, DI-star and TStarBot-X keep both; with the KL alone (VPT's setup) sp7 halved the action type's entropy (0.55 to 0.23) while its returns hardly varied; at 0.001 sp8's still fell 0.75 to 0.29; at 0.01 sp9's rose 1.0 to 1.45 while its wins fell. sp35 at 0.003 on raw nats (~100x AlphaStar's push; cell alone ~4.7 nats): total entropy 0.7 to 2.0 over 1,200 updates with wins falling, then a collapse into a losing habit | sp7, sp8, sp9, sp35 |
| KL anchor | 0.2 to the clone, fixed for the whole run | Keeps what cloning taught in the states the agent reaches; with resets to the current policy (sp3, sp4) RL dropped what it stopped using (attack orders halved), the forgetting Wolczyk et al. 2024 describe | AlphaStar, VPT; sp3-sp4 |
| Value warm-up | 50 updates training only the critic | A fresh critic's noise would otherwise steer the first policy updates | Wolczyk et al.; warcraftsim collapsed without it |
| Critic inputs | The core plus the opponent's scalars (resources, food, research, unit counts), training only | Judges the position better than a player can see it; the policy never sees it | AlphaStar |
| Critic: a deep network per channel (2026-10-11) | `model.Critic`: per reward channel, a linear to 256, 16 residual blocks of 256 and a linear to the value, as AlphaStar's baselines and DI-star's ValueBaseline (13.4M parameters; the model 25.2M). It reads the core detached, as before. Value loss weight 10 against the policy's 1, as theirs. The one-layer critic before it is dropped on load | sp37: the one-layer critic's fit of the advantage's mix rose to 0.3-0.6 over a 400-update warm-up, fell to ~0.08 once the policy learned, and wins fell (11.9% -> 7.3%, p 0.008). With the core detached, what the critic learns must live in its own layers, and a single layer cannot follow a core the policy keeps reshaping | AlphaStar detailed architecture; DI-star actor_critic_default_config.yaml (res_num 16, res_dim 256) |
| Value normalisation | PopArt, statistics rate 3e-3 | Returns change scale as rewards and game lengths change; the critic should not have to chase them | PopArt (van Hasselt 2016) |
| Auxiliary heads | None (dropped 2026-10-12; were opponent scalars and the core 4 and 16 steps ahead, weight 0.1 each) | Started untrained when the policy began learning and froze the core: sp40's cosine to 4 s later went 0.82 -> 0.97 in 150 updates, its build/train head 3.5 nats from the clone on pro games (training/bench/rl_drift.py) | SPR (Schwarzer 2021), AlphaStar-style opponent prediction, as they were |
| Gradient clip | 1.0 | Standard | |

## Reward (training/selfplay/reward.py)

| Decision | Value | Why | Rests on |
|---|---|---|---|
| Win, loss | +1, −1 | What we want | |
| Time limit | −1 (a loss) for both players, against any opponent | sp3 learned to survive for draws worth 0; sp6 scored self-play's clock by material and 98% of those games went to the clock while it boomed instead of fighting | sp3 eval, sp6 |
| No per-step time penalty | | It makes losing quickly better than losing slowly: agents learn to suicide | Common RL pitfall |
| Shaping | The pro's opening by edit distance (AlphaStar's build-order reward: in order only, sp34 paid nothing), the types and research levels the pro made over its game (AlphaStar's built-units and upgrades rewards: the Hamming distance closed, a type it never made costs; x0.5 after 8 min, x0.25 after 16, none after 24), damage dealt to the enemy (only its own units' damage to the enemy's units and buildings, from the game's damage events: sp34's grew from creeps hitting the AI) and taken (buildings only as they die), each enemy building the first time it is seen (the tutorial's "go there": bc10 sent 0.5% of its orders into the enemy's half, pros 32%); each its own critic output. Every term, amount and cap: [rewards.md](rewards.md) | The clone never wins, so win/loss alone teaches nothing (sp31). Earlier hand-made terms were farmed (gathering sp1, workers and food sp4, uncapped plan matching sp6); these pay only for capped targets from a pro game, or for fighting | AlphaStar (z pseudo-rewards, a critic per reward), ROA-Star (pay until the first mistake), OpenAI Five (decay by game time, zero-sum damage), RAISocketAI (a value head per reward, weights shifted toward win/loss) |
| No hero or creeps reward (removed 2026-10-11) | Hero experience (0.5 per 200) and creep value destroyed (1 per 1,000) are no longer paid, and their critic channels are gone. The critic is win, build, units, upgrades, dealt, taken (scout since dropped); bc11's 9-channel critic loads by name (`model.SHAPED_CRITIC`) | bc11 already creeps and levels its heroes, and both paid without ever meeting the enemy: the sp6 kind of farming. Whatever of them helps to win, the win pays for | bc11 eval (2026-10-11); sp6 |
| No scout reward (removed 2026-10-11) | Seeing an enemy building no longer pays (it paid 0.5 per building, once each), and its critic channel is gone; bc12's and sp35's 7-channel critics load by name (`model.SCOUT_CRITIC`) | sp35's learner at update 1,000, 32 games at hp 0.2: units in the enemy's base by minute 2.5 in 31 of 32 games, ~46 units lost a game, no army (0.3 combat units at 15 minutes against bc12's 5.8). Seeing the base pays only if it helps to win, and the win pays for that | sp35 probe eval (2026-10-11) |
| Shaping weight | max(0, 1 − win rate on the ladder's rung / 0.5): 1 while it never wins, 0 at 50% wins; damage taken weighs 1 − that; damage dealt always 1 | Helpers matter while it cannot win and fade once winning carries the signal; a harder AI starts the record over, so they come back. A fight it loses must still pay for the hits it got in | Our choice (sp3) |

## Curriculum and opponents (training/selfplay/league.py, actor.py)

| Decision | Value | Why | Rests on |
|---|---|---|---|
| Mixture per actor | 8 games: 4 against the ladder's rung, 4 self-play (the current policy or a PFSP snapshot) | The ladder keeps a beatable fixed opponent in every actor; self-play scales past it | AlphaStar's league uses fixed opponents for the same reason |
| Self-play opponent | The current policy half the time, else a snapshot weighted (1 − win rate against it)² | Plays the snapshots it still loses to | AlphaStar's PFSP |
| Snapshots | Every 200 updates | | |
| Opponent ladder | Warcraft's easy → normal → insane AI only (ai0, ai1, ai2), each first handicapped (below); up a rung after its last stage, at 70% wins over the last 50 games. --start-rung picks the first rung | Wins must be reachable for win/loss to teach, against opponents that fight back. The idle player, its defence, the spawned soldiers and the frozen-clone rungs are removed (below). Frozen past versions as opponents generalize: the PFSP snapshots continue the idea | sp1-sp4 never beat the easy AI; warcraftsim's income-tax curriculum; mini-AlphaStar |
| AI handicap stages (replace the idle rung as the way in) | On each AI rung, the AI's units take every hit scaled by 1/h, as if they had h = 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85 of their hit points (`league.HANDICAPS`; 0.05 and 0.1 added 2026-10-11: bc12 won 2 of 16 twenty-minute games at 0.05, none at 0.2, and sp35's learner never won at 0.2), then full; the next stage at 70% wins over 50 games; a new rung starts at 0.05 again. Health only: the hook's `handicap` debug op (wc3hook/stage.c), applied in the damage event before the game applies the damage, can also scale the damage they deal. A replay saves its handicap beside it (<name>.w3g.json) and playback applies it again. The eval takes --handicap | A real opponent that builds, defends and attacks, but beatable: the idle player and spawned soldiers taught walking to a base, not fighting. bc11 against the easy AI: 16 of 16 to the time limit, 0 wins, 3 of 5 replays never hit an enemy unit. At hp 0.2: 0 wins in 32, no building razed in any game, its army near an enemy building in only 12 of 32. Combat units (mean of 32 games), ours vs the AI's: 1.4 vs 5.9 at 5 min, 3.9 vs 9.0 at 10, 5.9 vs 6.6 at 15. From midgame it has the army to beat a 0.2-health AI but does not attack. Tested: at hp 0.25 a grunt takes 4x per hit and its hit points drop as much | bc11 eval (2026-10-11) |
| The idle rung's spawn stages (removed 2026-10-11, replaced by the AI handicap stages) | Was: against the idle player, 6 of the race's basic soldiers spawned 0.85, 0.75, ... 0.15 of the way (steps of 0.1: sp8's 0.85 to 0.6 dropped wins from 89% to 5%) from the learner's base to the idle player's; the next stage at 70% wins over 50 games, then none, then up a rung (--spawn-stage set it by hand) | Reverse curriculum (start near the goal, move the start back; JSRL is related): bc10 sent 0.5% of its point orders into the enemy's half against pros' 32%, so sp6/sp7 won 0-6% against a player doing nothing and had no signal. Removed because walking soldiers to a base is not fighting: bc11 wins 0 of 32 even against a 0.2-health easy AI and brings its army near an enemy building in only 12 (AI handicap stages above) | sp6, sp7, bc10 vs idle |
| The idle player's defence (removed 2026-10-11, with the idle player) | Was: every game against it started with 4 of its race's basic soldiers 350 out from its hall toward the learner, and 2 farms, a barracks and an altar in a row 700 out; they fought back where they stood | A player doing nothing fell to the 6 spawned soldiers; with a defence, winning took an army and several buildings destroyed. A handicapped AI is that real opponent, and it also builds and attacks | sp34 (2026-10-10); bc11 eval (2026-10-11) |
| Frozen-clone rungs (removed 2026-10-11) | Was: frozen cloned policies (bc6w, bc7; --fixed) as rungs between the idle player and the AI | A clone builds an economy but never attacks, so beating one means going and destroying it: the same walk-to-a-base lesson as the idle player. Replaced by the AI handicap stages; PFSP snapshots still play past versions | bc11 eval (2026-10-11) |
| Ladder games counted in start order | A ladder game counts once every game started before it has had time to end (4 real seconds per game-second of the limit: an hour at 15 minutes), in the order they started | A win ends a game early and a loss runs to the clock: counted as they finished, the first games after a change were all wins (sp9: 20 of 20 at update 90, 6% of 50 at update 190 on the same stage; sp8 moved on that way too early) | sp8, sp9 |
| Game time limit | 15 minutes, +5 (to 30) each time the last 50 AI games were won 50% of the time, at most once per 50 games; never shorter | Short games: more games per hour, so more wins and losses to learn from, and no time to stall; longer once it can win | sp3 (games averaged 27 of 30 minutes, half at the limit) |
| Plans in self-play | 90% of players get a pro plan of their matchup from the plan bank | As cloning saw (10% empty) | |
| Sampling temperature | 1.0 | Log-probabilities must be the actor's own for PPO | |

## Evaluation (training/play/evaluate.py)

| Decision | Value | Why |
|---|---|---|
| Report | Wins, defeats and time limits separately, plus gold and lumber per minute, peak food, kills, buildings razed, failed decisions, and whether it attacks: orders at enemy units or buildings (right-clicks too: pros fight with them), enemy buildings seen, damage to them, most army units near one | The training log's single "win rate" counted draws half and hid that sp3 never won; sp4's "attack orders" missed right-clicks |
| Standard check | 8-12 games of 15-30 minutes against the easy AI per checkpoint, races cycling, plans from the bank; play/diagnose.py for army and attacks against pros | |

## Infrastructure (training/cloud/)

| Decision | Value | Why |
|---|---|---|
| Self-play on Modal | One container pinned to us-east with two H100s (learner on GPU 0, 2 inference processes on GPU 1, 16 CPU-only actors on its cores); the games in 8 Wine sandboxes, made again when they land on another continent | Container cores cost a third of sandbox cores, so only the games run in sandboxes; inference on a GPU of its own stops batches waiting behind learner updates; the tunnel is shortest with the container in us-east (~14 ms a step from us-east sandboxes). Numbers: [9-benchmarks.md](9-benchmarks.md) |
| The games' Wine | WineHQ stable 11.0; Xvfb taking 2,048 connections | Valve's Proton 11 Wine with fsync took ~20% less sandbox CPU a game-step, but its runs' games died starting their next game far more often (sp33: 69 actor restarts in 36 minutes, sp34: 90 games; sp31 on WineHQ stable about one a run); 64 games fill Xvfb's default 256 connections. Numbers: [9-benchmarks.md](9-benchmarks.md) |
| Game-code cuts | The actors' games with `--game-cuts 1225` (wc3hook/cuts.c, from warcraftsim) | 4.1-4.6% less CPU a game-step and 7% less wall time, every step identical over 230,400 hashed steps; painting now and then (bit 256) cost 3% and stays off. Numbers: [warcraftsim.md](warcraftsim.md) |
| Wine at home | GE-Proton 10-34 with fsync (`$WC3_WINE`, prefix `~/.wine-wc3-ge`, set by the home environment script) | 16-17% less CPU a game-step than WineHQ stable, identical games, 0 of 192 map reloads failed (2026-10-18). Modal's image stays on WineHQ stable: GE there is untested. [warcraftsim.md](warcraftsim.md) |
| Game processes recycled by memory | Combat games get a new process at the next reload once theirs passes 450 MB private (`GameConfig.max_process_memory_mb`), with no episode limit (since E15 an episode count only replaced processes early) | Warcraft's heap grows with each map reload (~6 MB) and staged fight (~0.36 MB since E15), and only a new process frees it; the cap held a game at <= 506 MB (today's 583) for a relaunch every ~12 episodes, 2-3 s each, while the actor's other games step on. [rsi/log.md](rsi/log.md) E2, E12, E17 |
| Local timers and the world frame's unit list in step mode | The hook ticks the local timers by the game time each stepped frame ran (`local_timers`, wc3hook/step.c), and staging rebuilds the world frame's unit list after removing units (wc3hook/stage.c) | Stepping gives each frame no real time, so the local timers that end every spell's visual effects never fired, and the unit list kept every unit made: a staged fight leaked 1.98 MB, now 0.36, with every fight hashing the same. [rsi/log.md](rsi/log.md) E15 |
| Weights | Packed once per update; each inference process pulls them every 1 s on a thread of its own | Games play on weights at most a few updates old; pulls do not stop the batches |
| Old chunks | Dropped when made with weights more than 8 updates old (`--max-lag`) | sp15 trained on chunks ~21 updates old when the games outran the learner |
| Stuck learner | It stops itself when no update finished for 20 minutes (60 before the first) | Its GPU and actors would otherwise run to the machine's maximum duration |
| Cloning on Modal | H100 GPU function | Cheapest per pass among Modal GPUs; functions are not billed at the sandbox rate |
| Every run ends itself | When its work ends, and at a maximum run duration (and, on GCP, on preemption) | sp2 ran 4 hours instead of 1, and sp3's sandboxes outlived their learner by 25 minutes |

## GCP prices (us-central1, Google's billing catalog, 2026-10-01; Modal's rates the same day)

Why GCP VMs were tried for self-play; Modal's own prices and the current cost estimate are in
[9-benchmarks.md](9-benchmarks.md#cost).

| Machine | Spot $/hour | On-demand $/hour | Modal equivalent |
|---|---|---|---|
| a2-ultragpu-1g (A100 80 GB, 12 vCPU, 170 GB) | 3.01 (GPU 2.36) | about 5.0 (GPU 3.93) | A100 80 GB $2.50 plus CPU and memory |
| g2-standard-8 (L4, 8 vCPU, 32 GB) | 0.51 (GPU 0.34) | about 0.85 (GPU 0.56) | L4 $0.80 |
| H100 80 GB GPU alone | 5.86 | 9.80 | H100 $3.95 (no GCP quota) |
| c2d-highcpu-8 (8 vCPU = 4 cores, 16 GB) | 0.17 | about 0.30 | 4 cores in a sandbox: about $0.95 |

## Tried and dropped

| What | Why dropped |
|---|---|
| Self-play on GCP VMs (spot A100 learner, c2d-highcpu-8 actor VMs, weights every ~20 s) | Actors cost ~4x less on GCP than in Modal sandboxes, but sandboxes now run only the games (actors moved to the container's cheaper cores) and spot A100s were taken back twice in 30 minutes. GCP still runs evaluations and recordings (`training/cloud/gcp.py`) |
| The time limit in self-play scored by material (±0.5 to whoever held more than 55%, −0.5 to both when closer) | Paid for booming and for avoiding even trades: in sp6, 98% of self-play games went to the clock |
| Plan pseudo-rewards (0.05 per build-order item matched, 0.02 per planned unit type first made; AlphaStar's) | Farmed in sp6: with wins rare they were the main signal, so gold mined rose from 330 to 418 a minute while wins against the idle player fell from 20% to 6% over 480 games. The plan stays as an input |
| A cloning loss on pro replays during self-play (weight 1, 8 lanes per update; Wolczyk et al. 2024) | Dropped 2026-10-05 for sp6: the self-play game agents we follow (AlphaStar, VPT, OpenAI Five) keep only a KL to the clone, and a constant pull toward pros' average play may hold back attacking more than they do. Being checked against the papers; the code is on the local branch backup/with-pro-loss |
| Rare actions weighted in the loss (0.25-10x by rarity, TStarBot-X) | A weighted loss learns the weighted odds, not the players': bc9 gave cancel 4x and attack-move 0.25x, cancelled its own production 4.8 times a minute (pros 0.11) and barely attacked; undoing the weights at play time gave 6 army units a game instead of 1 and attacks. AlphaStar does not weight |
| A delay head (when to act next) | Pedantic: per-step noop already matched pros' acting rate |
| A harness rule stopping builders from taking new orders | A hack around rally spam; fixed at the source (rally bursts) and with the Shift-queue head |
| KL to the clone decaying by 0.9995 per update | Lets go of the clone entirely over a run |
| KL reference reset to the current policy every 500 updates (ProRL) | sp3-sp4 drifted from the clone and forgot attacking; the reference now stays on the clone |
| Shaping fading over a fixed number of updates | Unrelated to whether the policy can win; replaced by the win-rate weight |
| Gathering-only shaping, 0.002 per 100 | Too small (a game's gathering worth 0.006 against a win's 1), and it paid for mining without spending (sp1) |
| Time limit as a draw worth 0 | Made surviving the goal (sp3) |
| Shaping by "standing": gathering 0.01 per 100, kills 0.02 (hero 2x, building 3x), food in use 0.02, buildings started 0.02, zero-sum | Its total over a game is "have more at the end than the opponent", including an army kept alive: attacks cost food, kills paid little; sp4 turtled (attack orders halved, never razed a building) |

## Open questions

- Whether the opponent ladder produces the first wins and climbs to the easy AI (sp5).

- The AI/self-play mixture: if self-play games stay mostly time limits, more AI games until the first wins.
- Whether self-play ±0.5 at the limit lets two policies farm the tiebreak instead of attacking.
