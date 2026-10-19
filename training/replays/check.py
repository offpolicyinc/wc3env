"""Check labels and features against replays: what the converter and features.py make of real games.

    python -m training.replays.check runs/replays/1.29 --limit 40 --workers 6

Each replay is played with both players observed every 250 ms (the action step). Per replay, one JSON line:

- labels: each step's commands converted (training/replays/convert.py): actions per type, commands dropped as no-ops
  or left without an action and why, and fidelity: whether translating the label back (training/policy/actions.py)
  commands the same units with the same order, target and point the player gave. Production labels name no
  building (the translator picks the least busy one), so they are checked for a building that can make it.
- masks: each label's choices must be allowed by the masks of the observation it was made on (wc3hook/masks.c,
  through policy/native.py): a head that forbids a player's choice is counted.
- features: each player's features (the same C) against the truth the other player's observation holds: a visible enemy row must match its owner's own row (type, position, hp); a remembered structure is
  counted as still standing or gone; production progress stays within [0, 1] and is near 1 when a unit
  finishes; no number is NaN or infinite.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..policy.features import PAIRS, RACES, STEP_MS, env_race


def play(path: Path):
    """Yield ({player: observation}, static) for each step of a replay, both players observed."""
    from wc3env import w3g
    from wc3env.binary import SharedObservations
    from wc3env.game import launch, replay_startup_path

    from ..policy.features import Static

    path = path.absolute()
    replay = w3g.read(path)
    slots = tuple(p.slot for p in replay.opponents)
    race = {p.slot: p.race for p in replay.opponents}
    startup = replay_startup_path(path)
    if startup.is_file():  # a staged game (write_bots): the .w3g says "random", the staging picked the races
        for player in json.loads(startup.read_text())["setup"]["players"]:
            race[player["slot"]] = player["race"].replace("night_elf", "nightelf")
    with contextlib.redirect_stdout(io.StringIO()):
        game = launch(map=path, agents=slots, render=False, sound=False)
    try:
        rpc = game.rpc
        rpc.create_game(str(path), [{"slot": s, "control": "agent"} for s in slots])
        statics = None  # made at the first step: a player who picked random has a race only once the game starts
        rpc.debug("speed", factor=1024)
        rpc.debug("waitfloor", ms=1)
        shared = None
        while True:
            try:
                step = rpc.step(STEP_MS, observe=list(slots))
            except Exception:
                return  # the recording ended with the simulation
            shared = shared or SharedObservations(game.pid)
            observations = {o["player"]: shared.read(o["offset"], o["size"]) for o in step["observations"]}
            if statics is None:
                for s in slots:
                    if race[s] not in RACES:  # "random": the race of its own units
                        race[s] = _race(observations[s])
                statics = {s: Static.load(replay.map, (race[s], race[next(o for o in slots if o != s)])) for s in slots}
            yield observations, statics
            if step["reason"] != "target":
                return
    finally:
        game.close()


def _race(obs) -> str:
    """A player's race from its own units (a replay names a random pick only "random")."""
    from wc3env.binary import OWN, fourcc

    from ..policy.actions import reference

    units = reference()["units"]
    own = obs.units[(obs.units["flags"] & OWN) != 0]
    found = [units.get(fourcc(int(t)), {}).get("race") for t in own["type_id"]]
    return next(r for r in found if r in RACES)


# wc3env command -> the order it gives (cast orders by name)
_COMMAND_ORDERS = {
    "move": 851986,
    "attack": 851983,
    "smart": 851971,
    "stop": 851972,
    "drop_item": 852001,
    "revive": 852039,
}
# a label may give an order the engine treats alike: a move onto a unit follows it as a right-click does; a
# structure's right-click is its rally; a right-click on the ground is a move
_ALIKE = {(851986, 851971), (851971, 851980), (851971, 851986), (851988, 851971), (851988, 851986)}


def _order(command: dict) -> int:
    c, a = command["command"], command["arguments"]
    if c in _COMMAND_ORDERS:
        return _COMMAND_ORDERS[c]
    if c == "cast":
        return a["order"]
    if c in ("build", "train", "research"):
        return int.from_bytes(a["type_id"].encode("latin-1"), "big")
    if c == "learn":
        return int.from_bytes(a["ability_id"].encode("latin-1"), "big")
    if c == "use_item":
        return 852008 + a["slot"]
    return 0


def fidelity(converted, before) -> str:
    """How exactly translating the label reproduces the player's command."""
    from wc3env.binary import POINT, TARGET

    from ..policy.actions import PRODUCTION, reference, translate

    action, command = converted.action, converted.command
    commands = translate(action, before)
    if not commands:
        return "nothing to command"
    if action.type in PRODUCTION:
        building = next(u for u in before.units if int(u["unit_id"]) == commands[0]["unit_id"])
        kind = reference()["units"].get(int(building["type_id"]).to_bytes(4, "big").decode("latin-1"), {})
        if commands[0]["unit_id"] in command.units:
            return "exact"
        makes = kind.get("trains", []) + kind.get("researches", []) + kind.get("upgrades_to", [])
        return (
            "another building that can"
            if action.argument.to_bytes(4, "big").decode("latin-1") in makes
            else "wrong building"
        )
    if action.type == "buy":
        return "exact"
    order = _order(commands[0])
    same_item = action.type == "use_item" and 852008 <= order <= 852013  # the same item type in another slot
    if order != command.order_id and (command.order_id, order) not in _ALIKE and not same_item:
        return f"order differs: {command.order_id} -> {order}"
    a = commands[0]["arguments"]
    if command.kind == TARGET and a.get("target_id") not in (None, command.target):
        return "target differs"
    if command.kind == POINT and "x" in a and abs(a["x"] - command.x) + abs(a["y"] - command.y) > 1:
        return "point differs"
    if {c["unit_id"] for c in commands} != set(command.units):
        return "units differ"
    return "exact"


def check_features(f, obs, other, last_progress: dict, finished) -> Counter:
    """The features block f of `obs` against the other player's observation `other`; last_progress (unit id ->
    production progress a step ago) is updated."""
    from ..policy.features import COLUMN, OWNERS

    out = Counter()
    for k, v in f.items():
        if np.asarray(v).dtype.kind == "f" and not np.isfinite(v).all():
            out["non-finite " + k] += 1
    truth = {int(u["unit_id"]): u for u in other.units if u["flags"] & 1}  # the other player's own units
    n = int(f["unit_count"])
    rows, owners, ids = f["units"][:n], f["unit_owner"][:n], f["unit_id"][:n]
    for row, owner, uid in zip(rows, owners, ids):
        if owner != OWNERS.index("enemy"):
            continue
        u = truth.get(int(uid))
        if row[COLUMN["remembered"]]:
            out["remembered: standing" if u is not None else "remembered: gone"] += 1
            continue
        out["visible enemy"] += 1
        if u is None:
            out["visible enemy: not among its owner's units"] += 1
            continue
        mine = obs.units[obs.units["unit_id"] == uid][0]
        if int(mine["type_id"]) != int(u["type_id"]):
            out["visible enemy: type differs"] += 1
        if abs(float(mine["x"]) - float(u["x"])) + abs(float(mine["y"]) - float(u["y"])) > 1:
            out["visible enemy: position differs"] += 1
        if abs(float(mine["hp"]) - float(u["hp"])) > 1:
            out["visible enemy: hp differs"] += 1
    progress = rows[:, COLUMN["progress"]]
    out["progress out of [0, 1]"] += int(((progress < 0) | (progress > 1)).sum())
    for uid in finished:  # structures whose unit, upgrade or research finished this step: progress a step ago
        if uid in last_progress:
            out["finished"] += 1
            out["finished: progress was >= 0.9"] += last_progress[uid] >= 0.9
    last_progress.clear()
    last_progress.update({int(i): float(p) for i, p in zip(ids, progress) if p > 0})
    return out


def check_masks(action, f) -> list[str]:
    """Which heads' masks (f: the observation's features block) would have forbidden a player's action: none, for a
    sound mask."""
    from ..policy.actions import TYPES
    from ..policy.masks import argument_ids
    from ..policy.native import chosen_masks

    out = []
    kind = TYPES.index(action.type)
    if not f["type_mask"][kind]:
        out.append("type masked")
    k = argument_ids().get(action.argument, 0) if action.argument else 0
    masks = chosen_masks(f, kind, k)
    if action.argument:
        if not k:
            out.append("argument not in the vocabulary")
        elif not masks["argument"][k]:
            arg = action.argument
            out.append(f"argument masked: {arg.to_bytes(4, 'big').decode('latin-1') if arg >> 24 else arg}")
    rows = [int(u) for u in f["unit_id"]]
    if action.actor:
        if action.actor not in rows:
            out.append("actor not a row")
        elif not masks["actor"][rows.index(action.actor)]:
            out.append("actor masked")
    if action.target:
        if action.target not in rows:
            out.append("target not a row")
        elif not masks["target"][rows.index(action.target)]:
            out.append("target masked")
    return out


def play_bots(seed: int, races: tuple[str, str], map_name: str, minutes: float, difficulty: int = 2,
              replay: Path | None = None):  # fmt: skip
    """Yield ({player: observation}, static) for each step of a game Warcraft's AI plays on both sides (slots 0
    and 1, observed like agents), for `minutes` of game time or until it ends; then save it as `replay` (a .w3g,
    its startup options beside it) when given."""
    from wc3env.session import GameConfig, GameSession, MatchSetup, PlayerConfig

    from ..policy.features import Static

    config = GameConfig(
        map=map_name,
        players=tuple(PlayerConfig(slot, env_race(race)) for slot, race in enumerate(races)),
        ai_agents=(0, 1),
        ai_difficulty=difficulty,
        setup=MatchSetup(seed=seed, randomize_starts=True),
        render=False,
        sound=False,
        step_ms=STEP_MS,
        observation="binary",
    )
    statics = {0: Static.load(map_name, races), 1: Static.load(map_name, races[::-1])}
    with contextlib.redirect_stdout(io.StringIO()), GameSession(config) as session:
        session.reset()
        # as fast as the machine allows (the default is real time: 4x at most, against 56x observed every step)
        session.debug("speed", factor=2048)
        session.debug("waitfloor", ms=1)
        for _ in range(int(minutes * 60 * 1000 / STEP_MS)):
            observations, done, _ = session.step({0: [], 1: []})
            yield observations, statics
            if done:
                break
        if replay is not None:
            session.save_replay(replay)


def check_bots(seed: int) -> dict:
    """check() on a game of Warcraft's AI against itself, labels from its `script` orders."""
    races = RACE_PAIRS[seed % len(RACE_PAIRS)]
    return check(f"bots-{seed}", stream=play_bots(seed, races, BOT_MAP, BOT_MINUTES), origin="script")


BOT_MAP, BOT_MINUTES = "Maps/FrozenThrone/(2)EchoIsles.w3x", 15
RACE_PAIRS = PAIRS


def check(path: str, stream=None, origin: str = "player") -> dict:
    from wc3env.binary import EVENT_KINDS, fourcc

    from ..policy.native import Features, chosen_masks
    from .convert import convert

    types, dropped, unmapped, exact, feats, masked = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    type_orders = Counter()  # "<unit type> <order id>" -> casts of orders no known ability gives
    from ..policy.masks import ability_orders

    known_orders = ability_orders()
    per_step = Counter()
    steps, previous, states, progress = 0, {}, {}, {}
    try:
        for observations, statics in stream if stream is not None else play(Path(path)):
            steps += 1
            for player, obs in observations.items():
                if player not in states:
                    states[player], progress[player] = Features(statics[player]), {}
                f = states[player].step(obs, None)  # labels are not folded in here: checks only
                other = next(o for p, o in observations.items() if p != player)
                finished = {
                    int(e["unit_id"])
                    for e in obs.events
                    if EVENT_KINDS.get(int(e["kind"]))
                    in ("train_finish", "research_finish", "upgrade_finish", "construct_finish")
                }
                feats.update(check_features(f, obs, other, progress[player], finished))
                before, f_before = previous.get(player, (None, None))
                previous[player] = obs, f
                if before is None:
                    continue
                kept = 0
                for c in convert(before, obs, origin):
                    if c.action is None:
                        unmapped[c.reason] += 1
                    elif c.reason:
                        dropped[f"{c.action.type}: {c.reason}"] += 1
                    else:
                        kept += 1
                        types[c.action.type] += 1
                        exact[f"{c.action.type}: {fidelity(c, before) if c.command else 'exact'}"] += 1
                        for problem in check_masks(c.action, f_before):
                            masked[f"{c.action.type}: {problem}"] += 1
                        if c.action.type.startswith("cast") and c.command and c.action.argument not in known_orders:
                            # which unit types give orders the reference does not tie to an ability
                            for uid in c.command.units:
                                u = before.units[before.units["unit_id"] == uid]
                                if len(u):
                                    type_orders[f"{fourcc(u['type_id'][0])} {c.action.argument}"] += 1
                if steps % 20 == 0:  # how much the masks rule out: types allowed, and arguments for each
                    from ..policy.actions import TYPES

                    allowed = f_before["type_mask"]
                    masked["share: steps"] += 1
                    masked["share: types allowed"] += int(allowed.sum())
                    for kind in ("train", "build", "research", "cast", "learn", "use_item"):
                        k = TYPES.index(kind)
                        if allowed[k]:
                            masked[f"share: {kind} steps"] += 1
                            masked[f"share: {kind} arguments allowed"] += int(
                                chosen_masks(f_before, k, 0)["argument"].sum()
                            )
                per_step[min(kept, 3)] += 1
    except Exception:
        return {"id": Path(path).stem, "origin": origin, "error": traceback.format_exc()[-1200:]}
    return {
        "id": Path(path).stem,
        "origin": origin,
        "steps": steps,
        "actions": dict(types),
        "fidelity": dict(exact),
        "dropped": dict(dropped),
        "unmapped": dict(unmapped),
        "features": dict(feats),
        "masked": dict(masked),
        "type_orders": dict(type_orders),
        "actions_per_step": {str(k): v for k, v in sorted(per_step.items())},
    }


def write_type_orders(results: Path, least: int = 2) -> dict:
    """training/type_orders.json from a run's results: each unit type's orders that no known ability gives,
    seen at least `least` times."""
    from ..policy.masks import TYPE_ORDERS

    counts = Counter()
    for line in results.read_text().splitlines():
        counts.update(json.loads(line).get("type_orders", {}))
    table = {}
    for key, n in counts.items():
        kind, order = key.split()
        if n >= least:
            table.setdefault(kind, []).append(int(order))
    table = {k: sorted(v) for k, v in sorted(table.items())}
    TYPE_ORDERS.write_text(json.dumps(table, indent=1) + "\n")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", type=Path, help="a folder with good.txt (tools/check_replays.py)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("runs/check_training.jsonl"))
    parser.add_argument("--type-orders", action="store_true", help="only write training/type_orders.json from --out")
    parser.add_argument("--ids", type=Path, help="a file of replay ids to check instead of the folder's good.txt")
    parser.add_argument("--bots", type=int, default=0, help="instead: this many games of Warcraft's AI against itself")
    parser.add_argument("--first-seed", type=int, default=0, help="the bot games' seeds start here")
    args = parser.parse_args()
    if args.type_orders:
        table = write_type_orders(args.out)
        print(f"{sum(len(v) for v in table.values())} orders for {len(table)} unit types")
        return 0
    listing = args.ids or args.replays / "good.txt"
    ids = [line.split()[0] for line in listing.read_text().splitlines() if line.strip()]
    random.Random(7).shuffle(ids)
    ids = ids[: args.limit or None]
    keys = (
        "actions",
        "fidelity",
        "dropped",
        "unmapped",
        "features",
        "masked",
        "type_orders",
        "actions_per_step",
    )
    total: dict[str, Counter] = {k: Counter() for k in keys}
    errors = 0
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("w") as out:
        if args.bots:
            futures = [pool.submit(check_bots, seed) for seed in range(args.first_seed, args.first_seed + args.bots)]
        else:
            futures = [pool.submit(check, str((args.replays / f"{i}.w3g").absolute())) for i in ids]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            errors += "error" in row
            for k in total:
                total[k].update(row.get(k, {}))
            if "error" in row or n % 25 == 0:
                print(f"{n}/{len(ids)} {row['id']} {row.get('error', '')[-300:]}", flush=True)
    print(f"errors: {errors}")
    for k, counts in total.items():
        print(f"{k}: {dict(counts.most_common(40))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
