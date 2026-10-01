"""A stand-in for wc3hook.dll that speaks the RPC in docs/specs/protocol.md, for tests and for
developing agents without a game. It must pass tests/rpc_contract.py exactly as the DLL does.

The world is `SyntheticWorld`, a tiny model: units with positions, `move` orders walk them at
a fixed speed, gold rises by a fixed income, a `debug end` op ends the game. Enough to test the
env loop and the contract.

Time is game milliseconds; a step is a multiple of one 25 ms turn, as in the game.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .protocol import PROTOCOL_VERSION, QUEUABLE_COMMANDS, SCORE_FIELDS
from .session import PlayerConfig

TURN_MS = 25
FAKE_EXE_HASH = "fake-0000"


class RpcFault(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail)
        self.code, self.detail = code, detail


@dataclass
class SyntheticWorld:
    speed: float = 270.0  # map units per game second (a Peasant walks 190; keep tests short)
    income: int = 10  # gold per game second
    units: dict[int, dict] = field(default_factory=dict)
    targets: dict[int, list[tuple[float, float] | None]] = field(default_factory=dict)
    gold: int = 500
    game_time_ms: int = 0
    result: str = ""
    races: dict[int, str] = field(default_factory=lambda: {0: "human", 1: "orc"})

    @classmethod
    def echo_isles_start(cls) -> SyntheticWorld:
        w = cls()
        w.units[1000] = {
            "unit_id": 1000,
            "type_id": "htow",
            "owner": 0,
            "x": -5184.0,
            "y": 2944.0,
            "hp": 1500,
            "max_hp": 1500,
            "mana": 0,
            "max_mana": 0,
            "structure": True,
            "hero": False,
            "level": 0,
            "buffs": [],
        }
        for i in range(5):
            w.units[1001 + i] = {
                "unit_id": 1001 + i,
                "type_id": "hpea",
                "owner": 0,
                "x": -5000.0 + 60 * i,
                "y": 2800.0,
                "hp": 220,
                "max_hp": 220,
                "mana": 0,
                "max_mana": 0,
                "structure": False,
                "hero": False,
                "level": 0,
                "buffs": [],
            }
        w.units[2000] = {
            "unit_id": 2000,
            "type_id": "ogre",
            "owner": 1,
            "x": 5184.0,
            "y": -2944.0,
            "hp": 1500,
            "max_hp": 1500,
            "mana": 0,
            "max_mana": 0,
            "structure": True,
            "hero": False,
            "level": 0,
            "buffs": [],
        }
        return w

    def advance(self, ms: int) -> None:
        for _ in range(ms // TURN_MS):
            self.game_time_ms += TURN_MS
            dt = TURN_MS / 1000
            for uid, orders in list(self.targets.items()):
                if not orders or uid not in self.units:
                    del self.targets[uid]
                    continue
                target = orders[0]
                if target is None:
                    orders.pop(0)
                    continue
                tx, ty = target
                u = self.units[uid]
                dx, dy = tx - u["x"], ty - u["y"]
                d = math.hypot(dx, dy)
                s = self.speed * dt
                if d <= s:
                    u["x"], u["y"] = tx, ty
                    orders.pop(0)
                else:
                    u["x"] += dx / d * s
                    u["y"] += dy / d * s
            if self.game_time_ms % 1000 == 0:
                self.gold += self.income

    def observe(self, player: int) -> dict:
        own = [dict(u) for u in self.units.values() if u["owner"] == player]
        for u in own:  # own-only fields: the synthetic world has no orders, production or abilities
            u["order"] = None
            u["abilities"] = []
            if u["structure"]:
                u.update(state=None, state_seconds=0.0, queue=[], queue_seconds=0.0)
        return {
            "observer": player,
            "events_lost": 0,
            "orders": [],
            "orders_lost": 0,
            "chat": [],
            "destructables": [],
            "map": {"bounds": {"min_x": -10000, "min_y": -10000, "max_x": 10000, "max_y": 10000}},
            "players": [
                {
                    "id": p,
                    "kind": "neutral" if p in (12, 15) else "player",
                    "relation": "self" if p == player else "neutral" if p == 15 else "enemy",
                    "team": p,
                    "active": True,
                    "controller": "human" if p == 0 else "computer",
                    "shares_vision": p == player,
                    "shares_control": p == player,
                }
                for p in sorted(set(self.races) | {12, 15})
            ],
            "score": {key: 0 for key in SCORE_FIELDS},
            "player": {"gold": self.gold, "lumber": 0, "food_used": 5, "food_cap": 12},
            "units": own,
            "inside": [],
            "visible_enemies": [dict(u) for u in self.units.values() if u["owner"] != player],
            "items": [],
            "inventory": [],
            "events": [],
            "result": self.result,
        }

    def act(self, player: int, actions: list[dict]) -> list[dict]:
        rejected = []
        for i, a in enumerate(actions):
            uid = a["unit_id"]
            u = self.units.get(uid)
            if u is None:
                rejected.append({"index": i, "reason": "unknown_unit"})
            elif u["owner"] != player:
                rejected.append({"index": i, "reason": "not_your_unit"})
            elif a["command"] == "move":
                args = a.get("arguments", {})
                if any(
                    type(args.get(k)) not in (int, float) or not math.isfinite(args[k]) or abs(args[k]) > 1000000
                    for k in ("x", "y")
                ):
                    rejected.append({"index": i, "reason": "bad_arguments"})
                else:
                    if not args.get("queued", False):
                        self.targets[uid] = []
                    self.targets.setdefault(uid, []).append((float(args["x"]), float(args["y"])))
            elif a["command"] == "stop":
                if a.get("arguments", {}).get("queued", False):
                    self.targets.setdefault(uid, []).append(None)
                else:
                    self.targets.pop(uid, None)
            else:
                rejected.append({"index": i, "reason": "unknown_command"})
        return rejected


COMMANDS = (
    "move",
    "stop",
    "attack",
    "smart",
    "harvest",
    "build",
    "train",
    "research",
    "learn",
    "cast",
    "use_item",
    "drop_item",
    "select",
    "buy",
    "revive",
)
VALID_IN = {
    "info": None,
    "create_game": ("launched", "ended"),
    "step": ("in_game",),
    "observe": ("in_game", "ended"),
    "act": ("in_game",),
    "debug": ("in_game",),
    "reset": ("in_game", "ended"),
    "save_replay": ("in_game", "ended"),
    "quit": None,
}


class FakeServer:
    """`handle_line(line) -> line`, the server side of rpc.py."""

    def __init__(self, world_factory=None):
        self.world_factory = world_factory or SyntheticWorld.echo_isles_start
        self.world = None
        self.mode = None
        self.status = "launched"
        self.game_config = None
        self.quit_called = False
        self.sequence = [0] * 16
        self.log: list[dict] = []

    # ---- envelope ---------------------------------------------------------------------------
    def handle_line(self, line: str) -> str:
        def finite_number(text):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("non-finite JSON number")
            return value

        try:
            request = json.loads(line, parse_float=finite_number, parse_constant=finite_number)
        except (ValueError, TypeError):
            request = None
        return json.dumps(self.handle(request))

    def handle(self, request) -> dict:
        rid = request.get("id") if isinstance(request, dict) else None
        if not isinstance(rid, int) or isinstance(rid, bool):
            rid = 0
        try:
            if not isinstance(request, dict):
                raise RpcFault("bad_request", "not a JSON object")
            if type(request.get("protocol_version")) is not int or request["protocol_version"] != PROTOCOL_VERSION:
                raise RpcFault(
                    "bad_version", f"protocol_version {request.get('protocol_version')!r}, expected {PROTOCOL_VERSION}"
                )
            if not isinstance(request.get("id"), int) or isinstance(request.get("id"), bool):
                raise RpcFault("bad_request", "id must be an integer")
            method = request.get("method")
            if not isinstance(method, str):
                raise RpcFault("bad_request", "method must be a string")
            params = request.get("params", {})
            if not isinstance(params, dict):
                raise RpcFault("bad_params", "params must be an object")
            if method not in VALID_IN:
                raise RpcFault("unknown_method", method)
            allowed = VALID_IN[method]
            if allowed and self.status not in allowed:
                raise RpcFault("bad_status", f"{method} is not valid in {self.status}")
            self.log.append(request)
            result = getattr(self, f"_{method}")(params)
            return self._reply(rid, ok=True, result=result)
        except RpcFault as f:
            return self._reply(rid, ok=False, error=f.code, detail=f.detail)

    def _reply(self, rid, ok, result=None, error=None, detail=None) -> dict:
        r = {"protocol_version": PROTOCOL_VERSION, "id": rid, "ok": ok, "status": self.status}
        if ok:
            r["result"] = result if result is not None else {}
        else:
            r["error"], r["detail"] = error, detail or ""
        return r

    @staticmethod
    def _need(params, name, kind, check=None, detail=None):
        v = params.get(name)
        if kind is int and isinstance(v, bool) or not isinstance(v, kind) or (check and not check(v)):
            raise RpcFault("bad_params", detail or f"{name} must be {kind.__name__}")
        return v

    # ---- methods ----------------------------------------------------------------------------
    def _info(self, p) -> dict:
        return {
            "exe_hash": FAKE_EXE_HASH,
            "dll_version": "fake",
            "mode": self.mode,
            "game_time_ms": self.world.game_time_ms if self.world else 0,
            "instance": 0,
            "fake": True,
            "map": self.game_config["map"] if self.game_config else None,
        }

    def _create_game(self, p) -> dict:
        m = self._need(p, "map", str, bool)
        players = self._need(p, "players", list)
        try:
            if any(not isinstance(pl, dict) or ("race" in pl and pl["race"] is None) for pl in players):
                raise ValueError("race must be omitted or a name")
            parsed = [PlayerConfig(pl["slot"], pl.get("race"), pl["control"]) for pl in players]
            if not parsed or len({pl.slot for pl in parsed}) != len(parsed):
                raise ValueError("players must be nonempty with unique slots")
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcFault("bad_params", "players must have unique slots in 0..15 and known race/control") from exc
        mode = self._need(p, "mode", str, lambda v: v in ("stepping", "realtime"), "mode must be stepping or realtime")
        world = self.world_factory()
        for pl in parsed:
            if pl.slot not in world.races or (pl.race is not None and pl.race != world.races[pl.slot]):
                raise RpcFault("unsupported_config", f"player {pl.slot} race differs from the synthetic setup")
        players = [
            PlayerConfig(pl.slot, world.races[pl.slot], pl.control).to_dict()
            for pl in sorted(parsed, key=lambda pl: pl.slot)
        ]
        self.mode = mode
        self.game_config = {"map": m, "players": players}
        self.world = world
        self.sequence = [0] * 16
        self.status = "in_game"
        return {"game_time_ms": self.world.game_time_ms, "map": m, "players": players}

    def _reset(self, p) -> dict:
        self.world = None
        self.mode = None
        self.sequence = [0] * 16
        self.status = "launched"
        return {}

    def _save_replay(self, p) -> dict:
        raise RpcFault("bad_status", "the fake server records no replay")

    def _step(self, p) -> dict:
        if self.mode != "stepping":
            raise RpcFault("not_stepping", "step is stepping mode only")
        ms = self._need(
            p, "ms", int, lambda v: 25 <= v <= 60000 and v % TURN_MS == 0, "ms must be a multiple of 25 in 25..60000"
        )
        actions = p.get("actions") or {}
        if not isinstance(actions, dict):
            raise RpcFault("bad_params", "actions must map players to lists")
        players = {}
        for name in actions:  # every batch checked before any is applied, as the DLL does
            if not name.isdigit():
                raise RpcFault("bad_params", f"actions: {name} is not an agent player")
            players[name] = int(name)
        acts = {name: self._act({"player": players[name], "actions": batch}) for name, batch in actions.items()}
        self.world.advance(ms)
        if self.world.result:
            self.status = "ended"
        result = {
            "game_time_ms": self.world.game_time_ms,
            "frames": ms // TURN_MS,
            "elapsed_ms": ms,
            "reason": "game_over" if self.world.result else "target",
        }
        if acts:
            result["acts"] = acts
        return result

    def _observe(self, p) -> dict:
        player = (
            self._need(p, "player", int, lambda v: 0 <= v < 16, "player must be a slot 0..15") if "player" in p else 0
        )
        o = self.world.observe(player)
        for meta in o["players"]:
            if any(p["slot"] == meta["id"] and p["control"] == "agent" for p in self.game_config["players"]):
                meta["controller"] = "agent"
        o["protocol_version"] = PROTOCOL_VERSION
        o["sequence"] = self.sequence[player]
        o["game_time_seconds"] = self.world.game_time_ms / 1000
        self.sequence[player] += 1
        return o

    def _act(self, p) -> dict:
        player = self._need(p, "player", int, lambda v: 0 <= v < 16, "player must be a slot 0..15")
        if not any(pl["slot"] == player and pl["control"] == "agent" for pl in self.game_config["players"]):
            raise RpcFault("bad_params", "player is not configured as an agent")
        actions = self._need(p, "actions", list)
        rejected = []
        clean = []
        for i, a in enumerate(actions):
            if (
                not isinstance(a, dict)
                or type(a.get("unit_id")) is not int
                or not isinstance(a.get("command"), str)
                or not isinstance(a.get("arguments", {}), dict)
            ):
                raise RpcFault("bad_params", f"actions[{i}] must be {{unit_id, command, arguments}}")
            if a["command"] not in COMMANDS:
                rejected.append({"index": i, "reason": "unknown_command"})
                continue
            queued = a.get("arguments", {}).get("queued", False)
            if type(queued) is not bool or (queued and a["command"] not in QUEUABLE_COMMANDS):
                rejected.append({"index": i, "reason": "bad_arguments"})
                continue
            args = a.get("arguments", {})
            if "auto_place" in args and (type(args["auto_place"]) is not bool or a["command"] != "build"):
                rejected.append({"index": i, "reason": "bad_arguments"})
                continue
            clean.append((i, a))
        for j in self.world.act(player, [a for _, a in clean]):
            rejected.append({"index": clean[j["index"]][0], "reason": j["reason"]})
        rejected.sort(key=lambda r: r["index"])
        return {"rejected": rejected, "placements": []}

    def _debug(self, p) -> dict:
        op = self._need(p, "op", str)
        args = p.get("args", {})
        if not isinstance(args, dict):
            raise RpcFault("bad_params", "args must be an object")
        if op == "end":  # the harness's way to end a game: result becomes victory
            self.world.result = "victory"
            self.status = "ended"
            return {}
        raise RpcFault("bad_params", f"unknown debug op {op!r}")

    def _quit(self, p) -> dict:
        self.quit_called = True
        self.world = None
        self.status = "launched"
        return {}
