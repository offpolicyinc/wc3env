"""python -m training.play.replay_behavior <game.w3g>: Play a replay back headless (handicap re-staged by reset); player 0's scouting, attacks in the enemy's half, fights, and each minute its army (combat units: not workers or
buildings), how much of it is in the enemy's half, and its workers.
Echo Isles has two starts: ours is the one nearest our first units, theirs the other. Their half: nearer their start."""

import json
import sys
from pathlib import Path

import numpy as np

from training.policy.features import STEP_MS, env_race, start_points, worker_types
from training.selfplay.reward import CHANNELS, Shaping
from wc3env.binary import DEAD, EVENT_KINDS, OWN, STRUCTURE
from wc3env.session import GameConfig, GameSession, PlayerConfig

DAMAGED = next(k for k, n in EVENT_KINDS.items() if n == "damaged")
DEATH = next(k for k, n in EVENT_KINDS.items() if n == "death")
WORKERS = list(worker_types())
GAP = 20.0  # seconds without damage between us and them that end a fight
path = Path(sys.argv[1]).absolute()
config = GameConfig(
    map=str(path),
    players=(PlayerConfig(0, env_race("orc")), PlayerConfig(1, env_race("orc"), control="computer")),
    render=False,
    sound=False,
    step_ms=STEP_MS,
    observation="binary",
)
starts = np.array(start_points("Maps/FrozenThrone/(2)EchoIsles.w3x"))
out = {
    "game": path.stem,
    "scouted": None,
    "saw_building": None,
    "first_hit_their_half": None,
    "dealt_their_half": 0.0,
    "dealt_our_half": 0.0,
    "taken": 0.0,
    "fights": [],
    "kills": 0,
    "losses": 0,
    "per_minute": [],
    "reward_dealt": 0.0,
    "reward_taken": 0.0,
}  # [minute, army, army in their half, workers]; the reward's channels, unweighted
owner, pos, mine_s, theirs_s = {}, {}, None, None
fight = None
shaping = Shaping(None)  # dealt and taken as self-play pays them (no plan: build, units, upgrades pay nothing)
with GameSession(config) as s:
    s.reset()
    while True:
        obs, done, info = s.step({0: []})
        o = obs.get(0) if not done else (info.get("final_observations") or {}).get(0)
        if o is None:
            break
        t = o.game_time_seconds
        paid = shaping.step(o)
        out["reward_dealt"] += float(paid[CHANNELS.index("dealt")])
        out["reward_taken"] += float(paid[CHANNELS.index("taken")])
        u = o.units
        alive = (u["flags"] & DEAD) == 0
        for i, ow, x, y in zip(u["unit_id"].tolist(), u["owner"].tolist(), u["x"].tolist(), u["y"].tolist()):
            owner[i], pos[i] = ow, (x, y)
        mine = ((u["flags"] & OWN) != 0) & alive
        if mine_s is None and mine.any():
            c = np.array([u["x"][mine].mean(), u["y"][mine].mean()])
            k = int(((starts - c) ** 2).sum(1).argmin())
            mine_s, theirs_s = starts[k], starts[1 - k]
        if theirs_s is not None:
            d = np.hypot(u["x"][mine] - theirs_s[0], u["y"][mine] - theirs_s[1])
            if out["scouted"] is None and (d < 1500).any():
                out["scouted"] = round(t / 60, 1)
        if theirs_s is not None and int(t // 60) >= len(out["per_minute"]):
            body = mine & ((u["flags"] & STRUCTURE) == 0)
            army = body & ~np.isin(u["type_id"], WORKERS)
            far = np.hypot(u["x"] - theirs_s[0], u["y"] - theirs_s[1]) < np.hypot(
                u["x"] - mine_s[0], u["y"] - mine_s[1]
            )
            out["per_minute"].append(
                [
                    int(t // 60),
                    int(army.sum()),
                    int((army & far).sum()),
                    int((body & np.isin(u["type_id"], WORKERS)).sum()),
                ]
            )
        if out["saw_building"] is None and ((u["owner"] == 1) & ((u["flags"] & STRUCTURE) != 0) & alive).any():
            out["saw_building"] = round(t / 60, 1)
        for e in o.events:
            k, uid, other = int(e["kind"]), int(e["unit_id"]), int(e["other_id"])
            if k == DAMAGED:
                v = int(e["value"]) / 100
                ours_hit = owner.get(other) == 0 and owner.get(uid) == 1
                them_hit = owner.get(uid) == 0 and owner.get(other) == 1
                if not (ours_hit or them_hit):
                    continue
                x, y = pos.get(other if ours_hit else uid, (0.0, 0.0))  # where our unit stood
                their_half = np.hypot(x - theirs_s[0], y - theirs_s[1]) < np.hypot(x - mine_s[0], y - mine_s[1])
                if ours_hit:
                    out["dealt_their_half" if their_half else "dealt_our_half"] += v
                    if their_half and out["first_hit_their_half"] is None:
                        out["first_hit_their_half"] = round(t / 60, 1)
                else:
                    out["taken"] += v
                if fight is None or t - fight["end"] > GAP:
                    fight = {"start": t, "end": t, "dealt": 0.0, "taken": 0.0, "their_half": False}
                    out["fights"].append(fight)
                fight["end"] = t
                fight["dealt" if ours_hit else "taken"] += v
                fight["their_half"] |= bool(their_half)
            elif k == DEATH:
                if owner.get(uid) == 1:
                    out["kills"] += 1
                elif owner.get(uid) == 0:
                    out["losses"] += 1
        if done:
            break
out["result"] = o.result if o is not None else None
for f in out["fights"]:
    f.update(
        start=round(f["start"] / 60, 1), end=round(f["end"] / 60, 1), dealt=round(f["dealt"]), taken=round(f["taken"])
    )
for k in ("dealt_their_half", "dealt_our_half", "taken"):
    out[k] = round(out[k])
out["reward_dealt"], out["reward_taken"] = round(out["reward_dealt"], 2), round(out["reward_taken"], 2)
print(json.dumps(out))
