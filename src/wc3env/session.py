"""One owner for a game's process, episode lifecycle and combined player step."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .groups import ControlGroups
from .protocol import (
    Action,
    Observation,
    ProtocolError,
    normalize_actions,
    observation_result,
    observation_seconds,
    validate_actions,
)
from .rpc import RpcError

RACES = ("human", "orc", "undead", "night_elf", "random")
CONTROLS = ("agent", "computer")
OBSERVATIONS = ("json", "binary")


@dataclass(frozen=True)
class PlayerConfig:
    """Named races are requirements unless GameConfig.setup enables startup overrides."""

    slot: int
    race: str | None = None
    control: str = "agent"

    def __post_init__(self):
        if type(self.slot) is not int or not 0 <= self.slot < 16:
            raise ValueError("player slot must be an integer in 0..15")
        if (self.race is not None and self.race not in RACES) or self.control not in CONTROLS:
            raise ValueError("unknown player race or control")

    def to_dict(self) -> dict:
        return {"slot": self.slot, "control": self.control, **({"race": self.race} if self.race is not None else {})}


@dataclass(frozen=True)
class MatchSetup:
    """Opt in to startup race/controller overrides and a reproducible match seed."""

    seed: int | None = None
    randomize_starts: bool = False

    def __post_init__(self):
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed <= 0x7FFFFFFF):
            raise ValueError("seed must be an integer in 0..2147483647 or None")
        if type(self.randomize_starts) is not bool:
            raise ValueError("randomize_starts must be a bool")


@dataclass(frozen=True)
class GameConfig:
    map: str | None = None
    players: tuple[PlayerConfig, ...] = (
        PlayerConfig(0),
        PlayerConfig(1, control="computer"),
    )
    mode: str = "stepping"
    step_ms: int = 1000
    window_mode: str = "background"
    render: bool = True
    sound: bool | None = None  # None uses WC3_SOUND (on unless configured otherwise)
    background_visible: bool = False  # show the input-isolated background window
    ai_difficulty: int | None = None  # 0 easy, 1 normal, 2 insane for computer players; None keeps the map's
    ai_agents: tuple[int, ...] = ()  # agent slots whose melee AI keeps playing; step() orders act alongside it
    # Agent slots loaded as computer slots without an AI: Warcraft then has their units use spells on their
    # own, as it does for every computer player. ai_agents are always computer slots.
    computer_agents: tuple[int, ...] = ()
    setup: MatchSetup | None = None
    max_episodes_per_process: int | None = 32
    output_dir: str | Path | None = None
    # "binary": agent slots' observations are binary.BinaryObservation numpy records read from shared memory,
    # taken in the step's own round trip; computer players are not observed. Needs numpy. "json": dictionaries
    # for every configured player (the default).
    observation: str = "json"

    def __post_init__(self):
        if self.map is not None and (not isinstance(self.map, str) or not self.map.strip()):
            raise ValueError("map must be a nonempty path or map name")
        object.__setattr__(self, "players", tuple(self.players))
        if not self.players or any(not isinstance(p, PlayerConfig) for p in self.players):
            raise ValueError("players must contain PlayerConfig objects")
        if len({p.slot for p in self.players}) != len(self.players):
            raise ValueError("player slots must be unique")
        if not self.agent_slots:
            raise ValueError("at least one player must be an agent")
        if self.mode not in ("stepping", "realtime"):
            raise ValueError("mode must be stepping or realtime")
        if self.observation not in OBSERVATIONS:
            raise ValueError("observation must be json or binary")
        if self.window_mode not in ("interactive", "background"):
            raise ValueError("window_mode must be interactive or background")
        if type(self.render) is not bool:
            raise ValueError("render must be a bool")
        if self.sound is not None and type(self.sound) is not bool:
            raise ValueError("sound must be a bool or None")
        if type(self.background_visible) is not bool or (self.background_visible and self.window_mode != "background"):
            raise ValueError("background_visible requires background window mode and a bool")
        if self.ai_difficulty is not None and (
            type(self.ai_difficulty) is not int or self.ai_difficulty not in (0, 1, 2)
        ):
            raise ValueError("ai_difficulty must be None, 0, 1 or 2")
        object.__setattr__(self, "ai_agents", tuple(self.ai_agents))
        object.__setattr__(self, "computer_agents", tuple(self.computer_agents))
        if not set(self.ai_agents) | set(self.computer_agents) <= set(self.agent_slots):
            raise ValueError("ai_agents and computer_agents must be agent slots")
        if (self.ai_agents or self.computer_agents) and self.setup is None:
            raise ValueError("ai_agents and computer_agents need setup=MatchSetup(...)")
        if self.setup is not None and not isinstance(self.setup, MatchSetup):
            raise ValueError("setup must be MatchSetup or None")
        if self.max_episodes_per_process is not None and (
            type(self.max_episodes_per_process) is not int or self.max_episodes_per_process < 1
        ):
            raise ValueError("max_episodes_per_process must be a positive integer or None")
        if type(self.step_ms) is not int or not 25 <= self.step_ms <= 60000 or self.step_ms % 25:
            raise ValueError("step_ms must be a multiple of 25 in 25..60000")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().absolute())

    @property
    def agent_slots(self) -> tuple[int, ...]:
        return tuple(p.slot for p in self.players if p.control == "agent")

    def launch_setup(self) -> dict:
        """Fresh per-process startup options; returned seed is recorded by the DLL."""
        if self.setup is None:
            return {}
        seed = self.setup.seed if self.setup.seed is not None else secrets.randbelow(0x80000000)
        return {
            "_setup": {
                "seed": seed,
                "randomize_starts": self.setup.randomize_starts,
                "players": [
                    {**p.to_dict(), "control": "agent_ai"}
                    if p.slot in self.ai_agents or p.slot in self.computer_agents
                    else p.to_dict()
                    for p in self.players
                ],
            }
        }


def _launch(config: GameConfig):
    from .game import launch

    return launch(
        map=config.map,
        agents=config.agent_slots,
        window_mode=config.window_mode,
        render=config.render,
        sound=config.sound,
        background_visible=config.background_visible,
        ai_difficulty=config.ai_difficulty,
        ai_agents=config.ai_agents,
        output_dir=config.output_dir,
        **config.launch_setup(),
    )


@dataclass(frozen=True)
class PlayerView:
    """A player's latest observation; reading it never steps or consumes events."""

    _session: GameSession
    player: int

    @property
    def observation(self) -> dict | None:
        return self._session.observations.get(self.player)

    @property
    def done(self) -> bool:
        return bool(self.observation is not None and observation_result(self.observation))

    @property
    def groups(self) -> ControlGroups:
        return self._session.groups(self.player)


class GameSession:
    def __init__(self, config: GameConfig = GameConfig(), *, game_factory=None):
        self.config = config
        # A generated seed belongs to the session, including recycled/recovered processes.
        self._launch_config = (
            replace(config, setup=replace(config.setup, seed=secrets.randbelow(0x80000000)))
            if config.setup is not None and config.setup.seed is None
            else config
        )
        self._factory = game_factory or _launch
        self._game = None
        # Round trips may block. Ownership uses a separate, short-held lock so close
        # can detach and cancel the process without waiting for a reply.
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._faulted = False
        self._episodes_in_process = 0
        self._tuning: dict[str, dict] = {}
        self._shared = None  # binary.SharedObservations of the current process
        self.observations: dict[int, dict] = {}
        self.setup: dict | None = None
        self.steps = 0
        self.done = False
        self._groups: dict[int, dict[str, tuple[int, ...]]] = {p.slot: {} for p in config.players}

    @property
    def game(self):
        """Owned process, available after reset; exposed for diagnostics and test staging."""
        with self._state_lock:
            self._check_open()
            if self._game is None:
                raise RuntimeError("call reset() first")
            return self._game

    def view(self, player: int) -> PlayerView:
        if player not in {p.slot for p in self.config.players}:
            raise ValueError("player is not configured in this session")
        return PlayerView(self, player)

    def groups(self, player: int) -> ControlGroups:
        if type(player) is not int or player not in self.config.agent_slots:
            raise ValueError("control groups require a configured agent slot")
        return ControlGroups(self, player)

    def _release_game(self) -> None:
        with self._state_lock:
            game, self._game = self._game, None
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        if game is not None:
            # A replacement may fail before tuning is restored. Keep the previous
            # successful settings for the next recovery attempt in that case.
            self._tuning.update({op: dict(args) for op, args in getattr(game.rpc, "tuning", {}).items()})
            game.close()

    def _check_open(self) -> None:
        """Caller holds _state_lock."""
        if self._closed:
            raise RuntimeError("session is closed")

    def _ensure_game(self):
        """Caller owns the round lock; a new process stays local until published."""
        with self._state_lock:
            self._check_open()
            if self._game is not None:
                return self._game, False
        game = self._factory(self._launch_config)
        with self._state_lock:
            if not self._closed:
                self._game = game
                return game, True
        # close won the race. The launcher still owns this unpublished process.
        game.close()
        raise RuntimeError("session was closed during launch")

    def reset(self) -> dict[int, dict]:
        with self._lock:
            with self._state_lock:
                self._check_open()
                replay = self._game is not None and str(self._game.map).lower().endswith(".w3g")
            limit = self.config.max_episodes_per_process
            if replay or self._faulted or (limit is not None and self._episodes_in_process >= limit):
                self._release_game()
            self.observations = {}
            for groups in self._groups.values():
                groups.clear()
            self.setup = None
            self.steps = 0
            self.done = False
            self._faulted = False
            try:
                game, created = self._ensure_game()
                if created:
                    self._episodes_in_process = 0
                if not created:
                    game.rpc.reset()
                self.setup = game.rpc.create_game(
                    str(game.map), [p.to_dict() for p in self.config.players], self.config.mode
                )
                if created:
                    for op, args in self._tuning.items():
                        game.rpc.debug(op, **args)
                observations = self._observe_all()
                self._episodes_in_process += 1
                return observations
            except Exception:
                self._release_game()
                self.observations = {}
                self.setup = None
                raise

    def step(
        self, actions: dict[int, list[Action] | list[dict]], ms: int | None = None
    ) -> tuple[dict[int, dict], bool, dict]:
        """Send every player's batch, then advance the game by `ms` (default `config.step_ms`) in stepping mode."""
        step_ms = self.config.step_ms if ms is None else ms
        if type(step_ms) is not int or not 25 <= step_ms <= 60000 or step_ms % 25:
            raise ValueError("ms must be a multiple of 25 in 25..60000")
        with self._lock:
            started = time.perf_counter()
            timings = {"act": 0.0, "step": 0.0}
            rpc_timings = []
            with self._state_lock:
                self._check_open()
            if not self.observations or self._faulted:
                raise RuntimeError("call reset() before stepping")
            if self.done:
                raise RuntimeError("game is over; call reset()")
            if (
                not isinstance(actions, dict)
                or any(type(p) is not int for p in actions)
                or set(actions) != set(self.config.agent_slots)
            ):
                raise ProtocolError("provide one action batch for every agent slot, including empty batches")
            batches = {}
            for player in self.config.agent_slots:
                batch = normalize_actions(actions[player])
                if batch:
                    if observation_result(self.observations[player]):
                        raise ProtocolError("finished players must submit an empty batch")
                    validate_actions(Observation.of(self.observations[player]), batch)
                batches[player] = batch
            # Validate every player's batch before sending any order. One shared clock advance.
            before = observation_seconds(next(iter(self.observations.values())))
            timings["validation"] = 1000 * (time.perf_counter() - started)
            try:
                rejected = {p: [] for p in batches}
                placements = {p: [] for p in batches}
                step_result = None
                try:
                    phase = time.perf_counter()
                    stepping = self.config.mode == "stepping"  # stepping sends the batches with the step
                    try:
                        for p, batch in batches.items():
                            if batch and not stepping:
                                rpc = self.game.rpc
                                try:
                                    acknowledgement = rpc.act(p, [a.to_dict() for a in batch])
                                    rejected[p] = acknowledgement["rejected"]
                                    placements[p] = acknowledgement.get("placements", [])
                                finally:
                                    rpc_timings.append({"method": "act", "player": p, **(rpc.last_timing_ms or {})})
                    finally:
                        timings["act"] = 1000 * (time.perf_counter() - phase)
                    if stepping:
                        phase = time.perf_counter()
                        rpc = self.game.rpc
                        sent = {p: [a.to_dict() for a in batch] for p, batch in batches.items() if batch}
                        try:
                            observe = list(self.config.agent_slots) if self.config.observation == "binary" else None
                            step_result = rpc.step(step_ms, observe=observe, actions=sent)
                            for p, acknowledgement in step_result.get("acts", {}).items():
                                rejected[int(p)] = acknowledgement["rejected"]
                                placements[int(p)] = acknowledgement.get("placements", [])
                        finally:
                            timings["step"] = 1000 * (time.perf_counter() - phase)
                            rpc_timings.append({"method": "step", **(rpc.last_timing_ms or {})})
                        if step_result.get("reason") == "stalled":
                            raise RuntimeError("game clock stalled; call reset() to launch a fresh process")
                except RpcError as exc:
                    if exc.code != "bad_status" or exc.status != "ended":
                        raise
                    # Realtime and test staging can finish the game between two RPC calls.
                    step_result = {"reason": "game_over"}
                phase = time.perf_counter()
                if step_result is not None and "observations" in step_result:
                    observations = self._publish(self._copy_binary(step_result["observations"]))
                else:
                    observations = self._observe_all(rpc_timings)
                timings["observe"] = 1000 * (time.perf_counter() - phase)
                elapsed_ms = round((observation_seconds(next(iter(observations.values()))) - before) * 1000)
                if step_result is not None and elapsed_ms != step_ms and not self.done:
                    raise RuntimeError(f"game advanced {elapsed_ms} ms, expected {step_ms}; call reset()")
            except Exception:
                self._faulted = True  # a partially submitted round must not be retried
                raise
            self.steps += 1
            timings["total"] = 1000 * (time.perf_counter() - started)
            return (
                observations,
                self.done,
                {
                    "rejected": rejected,
                    "placements": placements,
                    "steps": self.steps,
                    "elapsed_ms": elapsed_ms,
                    "timings_ms": timings,
                    "rpc_timings_ms": rpc_timings,
                    "step_reason": step_result.get("reason", "target") if step_result else "realtime",
                },
            )

    def _copy_binary(self, located: list[dict]) -> dict:
        """Copy the binary observations a reply located out of shared memory, before the next request
        overwrites them."""
        if self._shared is None:
            from .binary import SharedObservations

            self._shared = SharedObservations(self.game.pid)
        return {o["player"]: self._shared.read(o["offset"], o["size"]) for o in located}

    def _publish(self, observations: dict) -> dict:
        self.observations = observations
        self.done = self.game.rpc.status == "ended"
        return observations

    def _observe_all(self, rpc_timings: list | None = None) -> dict:
        binary = self.config.observation == "binary"
        observations = {}
        for slot in self.config.agent_slots if binary else [p.slot for p in self.config.players]:
            rpc = self.game.rpc
            if binary:  # each request overwrites the mapping: copy before the next
                observations.update(self._copy_binary(rpc.observe_binary(slot)["observations"]))
            else:
                obs = observations[slot] = rpc.observe(slot)
                previous = self.observations.get(slot)
                delta = obs["game_time_seconds"] - previous["game_time_seconds"] if previous else 0
                obs["ticks_skipped"] = max(0, int(delta) - 1) if self.config.mode == "realtime" else 0
            if rpc_timings is not None:
                rpc_timings.append({"method": "observe", "player": slot, **(rpc.last_timing_ms or {})})
        return self._publish(observations)

    def debug(self, op: str, **args) -> dict:
        """A staging or tuning op (`spawn`, `resources`, `speed`, ...); see the protocol's Debug section."""
        with self._lock:
            with self._state_lock:
                self._check_open()
            return self.game.rpc.debug(op, **args)

    def save_replay(self, path: str | Path) -> Path:
        """Save the current episode's native Warcraft replay (.w3g). Call it once, when the episode is over:
        recording stops, and debug staging is not part of a replay. A staged game's startup options (match
        setup, AI difficulty, AI slots) go next to it (`<name>.w3g.json`), where launching the replay finds them."""
        from .game import replay_startup_path

        target = Path(path).expanduser().absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.game.rpc.save_replay(str(target))
        if self.game.startup:
            replay_startup_path(target).write_text(json.dumps(self.game.startup), encoding="utf-8")
        return target

    def close(self) -> None:
        """Cancel the owned process. An in-flight launch disposes its result on return."""
        with self._state_lock:
            self._closed = True
            game, self._game = self._game, None
        if game is not None:
            game.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
