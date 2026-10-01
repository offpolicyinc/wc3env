"""JSON-lines RPC between the host and the game server (wc3hook.dll), docs/specs/protocol.md.

Request:   {"protocol_version": 1, "id": 7, "method": "step", "params": {"ms": 250}}
Response:  {"protocol_version": 1, "id": 7, "ok": true, "status": "in_game", "result": {...}}
           {"protocol_version": 1, "id": 7, "ok": false, "status": "in_game", "error": "bad_params", "detail": "..."}

The transport is anything with `send(line)` and `readline()`: `HookPipe` for the real DLL,
`LocalTransport` for an in-process fake server (tests).
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Any, Protocol

from .protocol import PROTOCOL_VERSION

STATUSES = ("launched", "in_game", "ended")
ERROR_CODES = (
    "bad_version",
    "bad_request",
    "unknown_method",
    "bad_params",
    "bad_status",
    "not_stepping",
    "unsupported_config",
)
REJECT_REASONS = ("not_your_unit", "unknown_unit", "unknown_command", "bad_arguments", "queue_full", "no_build_site")


class RpcError(RuntimeError):
    """The server answered ok=false. `code` is one of ERROR_CODES, `detail` the human text."""

    def __init__(self, method: str, code: str, detail: str = "", status: str | None = None):
        super().__init__(f"{method}: {code}{': ' + detail if detail else ''}")
        self.method, self.code, self.detail, self.status = method, code, detail, status


class Transport(Protocol):
    def send(self, line: str, timeout: float = 120.0) -> None: ...
    def readline(self, timeout: float = 120.0) -> str: ...
    def close(self) -> None: ...


class RpcClient:
    def __init__(self, transport: Transport, timeout: float = 120.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout must be finite and positive")
        self.timeout = timeout
        self._t = transport
        self._next_id = 1
        self.status: str | None = None  # the lifecycle status from the last reply
        self.last_reply: dict | None = None
        self.last_timing_ms: dict | None = None
        self.tuning: dict[str, dict] = {}
        self._lock = threading.Lock()  # one complete request/reply owns the shared pipe

    def call(self, method: str, **params: Any) -> dict:
        with self._lock:
            started = time.perf_counter()
            self.last_reply = None
            self.last_timing_ms = None
            rid = self._next_id
            self._next_id += 1
            deadline = time.monotonic() + self.timeout
            try:
                self._t.send(
                    json.dumps(
                        {"protocol_version": PROTOCOL_VERSION, "id": rid, "method": method, "params": params},
                        allow_nan=False,
                    ),
                    timeout=self._remaining(deadline),
                )
                result = self._reply(method, rid, deadline)
                if method == "debug" and params.get("op") in ("speed", "waitfloor", "render"):
                    self.tuning[params["op"]] = dict(params.get("args", {}))
                return result
            finally:
                total = 1000 * (time.perf_counter() - started)
                timing = (self.last_reply or {}).get("timing_ms", {})
                # Older DLLs/fakes may omit timing. Unknown server time is not zero.
                server = timing.get("server")
                valid = isinstance(server, (int, float)) and math.isfinite(server) and server >= 0
                self.last_timing_ms = {
                    "total": total,
                    "server": server if valid else None,
                    "game_thread": timing.get("game_thread"),
                    "overhead": max(0.0, total - server) if valid else None,
                }

    def call_raw(self, request: dict | str) -> dict:
        """Send an arbitrary request object (for conformance tests) and return the raw reply."""
        with self._lock:
            self.last_timing_ms = None
            deadline = time.monotonic() + self.timeout
            self._t.send(
                json.dumps(request) if not isinstance(request, str) else request, timeout=self._remaining(deadline)
            )
            while True:
                line = self._t.readline(timeout=self._remaining(deadline))
                try:
                    self.last_reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(self.last_reply, dict) and "ok" in self.last_reply:
                    self.status = self.last_reply.get("status")
                    return self.last_reply

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RPC request deadline expired")
        return remaining

    def _reply(self, method: str, rid: int, deadline: float) -> dict:
        while True:
            line = self._t.readline(timeout=self._remaining(deadline))
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # a diagnostic line from the DLL (`native ...`), not a reply
            if not isinstance(msg, dict) or msg.get("id") != rid:
                continue  # a stale reply from a timed-out call
            self.last_reply = msg
            self.status = msg.get("status")
            if msg.get("protocol_version") != PROTOCOL_VERSION:
                raise RpcError(method, "bad_version", f"reply carried {msg.get('protocol_version')!r}")
            if not msg.get("ok"):
                raise RpcError(method, msg.get("error", "unknown"), msg.get("detail", ""), self.status)
            return msg.get("result", {})

    # The calls in docs/specs/protocol.md.
    def info(self) -> dict:
        return self.call("info")

    def create_game(self, map: str, players: list[dict], mode: str = "stepping") -> dict:
        return self.call("create_game", map=map, players=players, mode=mode)

    def step(self, ms: int, observe: list[int] | None = None, actions: dict[int, list[dict]] | None = None) -> dict:
        """Advance ms; with `observe`, also write those players' binary observations (binary.py) and
        return their offsets in `observations`; with `actions` ({player: [action, ...]}), first send them as act
        would, their acknowledgements in `acts` ({"player": {rejected, placements}}): one round trip."""
        params: dict = {"ms": ms}
        if observe:
            params["observe"] = observe
        if actions:
            params["actions"] = {str(p): batch for p, batch in actions.items()}
        return self.call("step", **params)

    def reset(self) -> dict:
        return self.call("reset")

    def save_replay(self, path: str) -> dict:
        """Write this episode's native .w3g to an absolute path. Recording ends; one save per episode."""
        return self.call("save_replay", path=path)

    def observe(self, player: int = 0) -> dict:
        return self.call("observe", player=player)

    def observe_binary(self, player: int = 0) -> dict:
        """Write the player's binary observation (binary.py); returns its offset in `observations`."""
        return self.call("observe", player=player, format="binary")

    def act(self, player: int, actions: list[dict]) -> dict:
        return self.call("act", player=player, actions=actions)

    def ai_difficulty(self, player: int) -> dict:
        """Read the native player difficulty (0 easy, 1 normal, 2 insane)."""
        return self.debug("ai_difficulty", player=player)

    def overlay(self, panel: str, *, x: float = -0.30, y: float = 0.60, seconds: float = 2) -> dict:
        """Render local text; layout and content belong to the caller."""
        return self.debug("overlay", panel=panel, x=x, y=y, seconds=seconds)

    def debug(self, op: str, **args: Any) -> dict:
        return self.call("debug", op=op, args=args)

    def quit(self) -> dict:
        return self.call("quit")

    def close(self) -> None:
        self._t.close()


class LocalTransport:
    """In-process transport: each sent line is handled by `server.handle_line(line) -> line`."""

    def __init__(self, server):
        self.server = server
        self._out: deque[str] = deque()

    def send(self, line: str, timeout: float = 120.0) -> None:
        self._out.append(self.server.handle_line(line))

    def readline(self, timeout: float = 120.0) -> str:
        return self._out.popleft()

    def close(self) -> None:
        pass
