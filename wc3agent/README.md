# wc3agent

A Warcraft III agent with two models: **macro (System 2)** plans the economy, worker duties and
army objectives; **micro (System 1, Jev)** controls army movement, scouting, loot and combat.
Macro requests start at least five game seconds apart, with only one request in flight. After a
slow reply, the next request starts from fresh state as soon as its orders have been submitted;
there is no additional five-second delay after the reply. `--turn-interval-seconds` can increase
this spacing, but values below five are rejected.
Ready macro orders reach the game before micro's next decision. Micro follows explicit group objectives,
including retreat, waiting and recovery; economic and worker-job decisions stay with macro.
In realtime, each group asks Jev at most once per second using the shared observation loop.
Answers are sent immediately; new macro orders invalidate outdated answers.

`group ... at X Y` delegates those units and an objective to micro; they walk there, or
attack-move with `group ... attack at X Y`. Direct move, stop, attack, take and worker orders take
control back until macro explicitly delegates again; pending replies from the previous owner are
discarded. A direct cast or item use keeps the unit in its group: micro leaves it alone until the
spell or item has gone off. A summoned unit joins its summoner's group. The dispatcher sends the
chosen orders as written: it neither inserts Stop before teleporting nor prevents micro from
interrupting its own spells. It does not invent initial
army objectives or turn direct attack orders into implicit delegations.

Ordinary build commands use native autoplace. Macro chooses the worker, building type and an anchor
(`near goldmine1`, `near X Y`, or its hall by default); the build action calls Warcraft's native placement
search and submits a normal build order at the returned site. It does not start an AI controller. The hook
treats sites already promised to pending builds as occupied, so several builds in one reply get separate
sites. The runner records the selected coordinates from `step()` feedback and uses later
observations/events to confirm construction. Python performs no footprint search. See [build semantics and limits](../docs/specs/actions.md#native-building-placement).
Owned building abilities are observed from the game as well.

Parsing checks command syntax and entity references. The engine decides gameplay legality;
there is no extra dispatcher budget model, skill-rank validator, mana/cooldown veto or cast lock.
Micro still uses its existing observation-derived candidate menu, and binds item selections to
the item shown when asked. API acceptance is not proof an order executed: subsequent observations
and events are the evidence of its outcome.

## Run

Install from the repository root and configure model keys in [.env.example](../.env.example):
`MACRO_PROVIDER`, `MACRO_MODEL`, the provider's key, and `TYPESAFE_API_KEY` for Jev.

```powershell
python -m pip install -e . -e wc3agent
python -m wc3agent scenarios --list
python -m wc3agent scenario opening --out sessions/opening --realtime
python -m wc3agent scenarios creep_easy fight_even
python -m wc3agent melee --difficulty hard
```

Stepped play pauses while models think. `--realtime` keeps the game running; `--visible` shows
scenario windows. The runner applies the requested race during match setup, including nonhuman
scenario starts. Environment setup is in the [main README](../README.md).

## Structure

Under `src/wc3agent/`:

| Path | Purpose |
|---|---|
| `agent.py`, `play.py` | Coordinate models and run the game |
| `control.py` | Who commands each unit: macro directly or micro through a group |
| `macro/`, `micro/` | Strategic planning and unit control |
| `prompts.py`, `macro/prompts.py`, `micro/prompts.py` | Shared, macro and micro prompts |
| `game/` | Warcraft knowledge and order translation |
| `game/policies.py` | Every fixed rule code applies without asking a model (menu trimming, when micro is skipped) |
| `models/` | Model connections |
| `scenarios/` | Test setups and scoring |
| `recording.py`, `report.py` | Logs, costs and reports |

## Evaluate

[Scenario definitions](src/wc3agent/scenarios/definitions/) specify a starting situation, goal,
time limit and checks. Each run saves an HTML report, transcript, scored summary and replay.
The report shows order outcomes and the actual model request/response messages.
Micro stays active throughout scenarios and full games; their game-time limits bound each run.
Programmatic callers can set `MeleeConfig.micro_call_limit` for an explicit per-group/type call budget.

```powershell
python -m wc3agent report sessions/opening
```
