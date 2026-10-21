"""A self-play actor: plays games with the learner's latest policy and sends what happened, in 64-step chunks.

    python -m training.selfplay.actor --learner 127.0.0.1:7600 --inference 127.0.0.1:7500 --games 4 --ai-games 1

Each game is the learner against itself or a frozen snapshot (league.choose), or, in the first --ai-games games, the
league's ladder rung (observed so the critic can see its side), handicapped at first. The learner plays slot 0, the
local player (Game.slot), the opponent slot 1, a computer slot whose units cast on their own; in self-play both are
the latest policy, but only slot 0 is learned from. It is recorded per step: the data set's arrays (features, the choices as
indices, the masks they were sampled under), the log-probability and the critic's value, the reward that
followed (training/selfplay/reward.py), the opponent's scalars and the memory the model had at the chunk's
start. Chunks follow the attention windows (every 64 steps from the game's start); a chunk's last step's
value is the next step's, or 0 when the game ended there.

The game hook computes each observation's features and masks (GameConfig.features; each step sends back the
decisions they fold in), the inference process (training/selfplay/inference.py) runs the model for every actor at
once, and the actor turns the choices into commands and records them (policy/agent.py); it uses no GPU. The games whose observations are back decide together, one request per policy, and
step on while the others keep going; a round's decisions are ended a round later, so this process works while
the inference process runs the model. Games run in wc3env.vector worker processes (Windows Python here, or on another
machine: --dial, or --remote).

What an episode is comes from the mode (--mode; training/selfplay/modes): whole games (modes/game.py) or fights
(modes/combat.py), with their map, matchups, plans, who the computer plays, each step's rewards and when it is over;
the rest is the same for every mode.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from wc3env.session import GameConfig, MatchSetup, PlayerConfig
from wc3env.vector import VectorSession, dial_workers, local_workers, remote_workers, wine_workers

from ..policy.agent import Agents, Decision, Seat
from ..policy.features import STEP_MS, Static, env_race
from ..policy.native import hook_decision, hook_spec
from ..policy.plan import PlanBank
from ..policy.plan import empty as empty_plan
from ..replays.dataset import _lists, append_decision, arrays
from . import modes
from .batches import chunks_address
from .league import SELF, choose
from .reward import CHANNELS
from .runfiles import Recording
from .scenarios import SCENARIOS
from .transport import Client, address

MAX_RESTARTS = 20
POLL = 0.001  # seconds between looks for the inference process's answers while games step
MAX_ROUNDS = 3  # requests in flight to the inference process at most
# chunks sent together as one segment, the learner's advantages computed over all of it (OpenAI Five's 256-step
# segments, 16-step training samples): a decision sees real rewards to its segment's end, up to 64 s, not its chunk's
# 16; each chunk still trains on its own, the model's attention window still the chunk
SEGMENT = 4


class Stages:
    """Mean milliseconds per stage since the last report. Per game step: game (commands sent -> observation back),
    of it (wc3env.vector's info["hops"]) h_round the round trip from this process's socket, w_step the game's step
    in its worker and h_queue the observation waiting here to be read; queued (observation back -> its decision
    begun: --gather, or a full MAX_ROUNDS); decide (begun -> commands sent). Per request: inference (sent -> answer
    read), blocked (of that, the actor waiting on it), begin and end (the actor's CPU), games (decided in it),
    in_flight (requests already out when it began). Per report: busy (% of the time the actor was not waiting)."""

    def __init__(self):
        self.total, self.count = defaultdict(float), defaultdict(int)

    def add(self, stage: str, seconds: float, scale: float = 1000.0) -> None:
        self.total[stage] += seconds * scale
        self.count[stage] += 1

    def line(self) -> str:
        out = ", ".join(f"{k} {self.total[k] / self.count[k]:.1f}" for k in self.total)
        self.total.clear(), self.count.clear()
        return out


@dataclass
class Recorder:
    """One learning player's steps since its chunk started."""

    static: Static
    plan: dict | None
    start_state: list | None  # the model's memory before the chunk's first step (numpy; None: the game's start)
    version: int
    lists: dict = field(default_factory=_lists)
    extra: dict = field(
        default_factory=lambda: {k: [] for k in ("log_prob", "value", "reward", "done", "opponent", "recent")}
    )

    def add(self, obs, d: Decision, opponent: np.ndarray) -> None:
        append_decision(self.lists, obs, d.features, d.choices, d.masks)
        self.extra["recent"].append(d.features["recent"])  # the cells of the decisions before this step
        self.extra["log_prob"].append(d.log_prob)
        self.extra["value"].append(d.value)
        self.extra["opponent"].append(opponent.astype(np.float16))
        self.extra["reward"].append(np.zeros(len(CHANNELS), np.float32))
        self.extra["done"].append(False)

    def __len__(self) -> int:
        return len(self.extra["log_prob"])

    def chunk(self, bootstrap) -> dict:
        """The chunk to send: `bootstrap` the critic's values [CHANNELS] after its last step (0 when the game ended)."""
        out = arrays(self.lists, self.static, -1)
        out.update(
            log_prob=np.array(self.extra["log_prob"], np.float32),
            value=np.stack(self.extra["value"]).astype(np.float32),  # [T, CHANNELS]
            reward=np.stack(self.extra["reward"]).astype(np.float32),  # [T, CHANNELS]
            done=np.array(self.extra["done"], bool),
            opponent=np.stack(self.extra["opponent"]),
            recent=np.stack(self.extra["recent"]),
            bootstrap=np.broadcast_to(np.asarray(bootstrap, np.float32), (len(CHANNELS),)).copy(),
            version=np.int64(self.version),
            start_state=self.start_state,
        )
        # always the plan it played with: a chunk without one would get a plan read from its own actions in the
        # learner (ReplayData), so a player given the empty plan was trained as if it had followed its own
        plan = self.plan if self.plan is not None else empty_plan()
        for k in ("heroes", "order", "mix"):
            out[f"plan_{k}"] = np.asarray(plan[k])
        return out


class _Inference:
    """Agents' backend when the model runs in the inference process: a policy's requests on the actor's link to it
    (in flight until received, in order); the version of "latest" each answer was made with."""

    def __init__(self, actor, policy: str):
        self.actor, self.policy = actor, policy

    def send(self, inputs: dict) -> None:
        self.actor.inference.send("act", (self.policy, inputs))
        self.actor.requests.append(time.perf_counter())

    def receive(self) -> dict:
        began = time.perf_counter()
        reply = self.actor.inference.receive()
        now = time.perf_counter()
        self.actor.stages.add("blocked", now - began)
        self.actor.stages.add("inference", now - self.actor.requests.popleft())
        version = reply.pop("version")
        if self.policy == "latest":
            self.actor.version = version
        return reply


@dataclass
class Game:
    races: tuple[str, str]  # (the learning player's race, the opponent's)
    ladder: bool  # a game against the league's ladder rung
    # the learning player's slot (the opponent plays the other): 0, the local player, the only human one (another
    # agent's slot is a computer's, whose units cast on their own); starts are random either way. The AI is weaker in
    # slot 0 (against itself in orc mirrors, slot 1 won 16 of 16), so it stays in 1
    slot: int = 0
    opponent: str = SELF  # SELF, a snapshot's name, or a ladder rung ("ai<difficulty>")
    began: float | None = None  # when a ladder game began (unix seconds; None: not running): the league's order
    stage: tuple | None = None  # the (rung, stage) a ladder game began under
    seats: dict = field(default_factory=dict)  # slot -> Seat
    policy: dict = field(default_factory=dict)  # slot -> "latest" or a snapshot's name
    recorders: dict = field(default_factory=dict)  # slot -> Recorder (learning players)
    episode: modes.Episode | None = None  # the running episode's rules (the mode's)
    waiting: dict = field(default_factory=dict)  # slot -> a full Recorder waiting for its bootstrap value
    held: dict = field(default_factory=dict)  # slot -> its finished chunks waiting for their segment to fill (SEGMENT)
    steps: int = 0
    staging: bool = False  # the step staging its next episode is out (modes.Episode.start)
    # only evaluates (--eval-games): against the ladder's rung at the learner's fixed evaluation stage, the learning
    # seat played by each evaluated policy in turn (League plan's "evaluate"), nothing recorded or learned
    eval: bool = False
    rng: np.random.Generator | None = None  # an evaluation game's draws (the fight's armies): the same for each policy
    recording: Recording | None = None  # this episode, recorded for the run's replays/ (--replay-minutes)
    # (a scenario, its level) in place of the ladder's fight (--scenarios): its opponent, outside the ladder's count
    scenario: tuple | None = None

    @property
    def other(self) -> int:
        """The opponent's slot."""
        return 1 - self.slot

    def matchup(self, slot: int) -> tuple[str, str]:
        """(own race, opponent's race) as `slot` plays it."""
        return self.races if slot == self.slot else self.races[::-1]

    @property
    def seatless(self) -> bool:
        """The opponent's slot is not ours to decide: the ladder's opponent plays it (Warcraft's AI, or the mode's
        script)."""
        return self.ladder


class Actor:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        # a new learner may take a while to listen (it loads its checkpoint first)
        self.client = Client(address(args.learner), args.authkey.encode(), wait=1800)
        self.version, self.league = self.client.call("league")
        self.chunks = self._chunks_client(1800)  # chunks go to the learner's batch process (batches.py)
        self.inference = Client(address(args.inference), args.authkey.encode(), wait=1800, compress=False)
        self.agents: dict[str, Agents] = {}  # per policy ("latest", or a snapshot's name)
        self.bank = PlanBank(args.plans) if args.plans and os.path.exists(args.plans) else None
        self.outbox: list[dict] = []
        self.segments = 0  # segments sent so far (their ids)
        self.ended: list[tuple] = []  # games ended since the last report: (opponent, outcome, score, minutes, gold/min,
        # when it began, its (rung, stage); None for self-play)
        self.mode = modes.get(args.mode)
        self.map = args.map or self.mode.map
        pairs = self.mode.pairs(args.races)
        evaluating = range(args.games - args.eval_games, args.games)  # the last --eval-games only evaluate
        self.games = [Game(pairs[(args.seed * args.games + g) % len(pairs)], g < args.ai_games or g in evaluating,
                           eval=g in evaluating) for g in range(args.games)]  # fmt: skip
        self.eval_turns = 0  # evaluation games begun: each policy in turn, a round of them on the same draws
        self.rung = self.league["rung"]
        self.steps, self.started, self.last_refresh, self.failures = 0, time.time(), time.time(), 0
        self.statics: dict[tuple, Static] = {}  # the map's data per (own race, opponent's race): read once
        self.idle, self.reported = 0.0, time.perf_counter()  # seconds this thread waited (for games or answers)
        self.stages, self.requests = Stages(), deque()  # timings for the report; the inference requests' send times
        self.sent_at: dict[int, float] = {}  # game -> when its commands went, when its observation came, when begun
        self.back_at: dict[int, float] = {}
        self.begun_at: dict[int, float] = {}
        self.next_replay = time.time() + 60  # the first a minute in, then every --replay-minutes
        self.scenarios = [(n, float(lv) if n == "fight" else int(lv))  # fight's level: a stage (combat.HANDICAPS)
                          for n, lv in (x.split(":") for x in self.args.scenarios.split(",") if x)]  # fmt: skip

    # ---- episodes
    def _start(self, game: Game) -> None:
        game.seats, game.policy, game.recorders, game.waiting, game.held = {}, {}, {}, {}, {}
        game.steps = 0
        game.opponent = self.rung if game.ladder else choose(self.league, self.rng)
        game.scenario, evaluated = None, None
        if game.ladder and not game.eval and self.scenarios and self.rng.random() < self.args.scenario_share:
            name, level = self.scenarios[self.rng.integers(len(self.scenarios))]
            game.scenario, game.opponent = (SCENARIOS[name](), level), f"{name}/{level}"
        if game.eval:  # the next policy in turn; a round (one game each) shares its draws
            policies = self.league.get("evaluate") or ["latest"]
            evaluated = policies[self.eval_turns % len(policies)]
            game.rng = np.random.default_rng([self.args.seed, self.eval_turns // len(policies)])
            self.eval_turns += 1
        # the rung and stage a ladder game begins under: it counts only if the league is still there
        elif game.ladder and game.scenario is None:
            game.began, game.stage = time.time(), (game.opponent, self.league.get("stage", 0))
        for slot in (game.slot,) if game.seatless else (0, 1):  # the ladder's opponent has no seat
            race, other = game.matchup(slot)
            game.seats[slot] = Seat(self._static((race, other)), self.mode.plan(self, race, other))
            game.policy[slot] = "latest" if slot == game.slot or game.opponent == SELF else game.opponent
            if evaluated is not None:
                game.policy[slot] = evaluated
        game.episode = self.mode.episode(self, game)

    def handicap(self, game: Game) -> float:
        """The ladder opponent's handicap for this game: the league's stage, or the fixed evaluation stage."""
        return self.league.get("eval_handicap", 1.0) if game.eval else self.league.get("handicap", 1.0)

    def _begin(self, i: int, obs: dict, session) -> dict | None:
        """Game i's next episode begins at `obs`: its observations to decide on, or None when the mode stages it
        first (the step that stages it comes back through recv)."""
        game = self.games[i]
        self._start(game)
        obs = game.episode.start(i, obs, session)
        game.staging = obs is None
        return obs

    def _agents(self, policy: str) -> Agents:
        if policy not in self.agents:
            self.agents[policy] = Agents(remote=_Inference(self, policy))
        return self.agents[policy]

    def _configs(self):
        return [
            GameConfig(
                map=self.map,
                players=tuple(PlayerConfig(slot, env_race(g.matchup(slot)[0])) for slot in (0, 1)),
                **self.mode.config(g, self.rung),
                setup=MatchSetup(seed=int(self.rng.integers(2**31 - 1)), randomize_starts=True),
                render=False,
                game_cuts=self.args.game_cuts,
                sound=False,
                step_ms=STEP_MS,
                observation="binary",
                features={slot: hook_spec(self._static(g.matchup(slot))) for slot in (0, 1)},
            )
            for g in self.games
        ]

    # ---- the loop
    def run(self, launch) -> None:
        """Plays until --steps (or for good); `launch` starts the games' workers (wc3env.vector's Launch)."""
        while True:  # a new set of games whenever the ladder's rung changes (the AI, its difficulty, or neither)
            with VectorSession(self._configs(), group_size=self.args.group, launch=launch) as session:
                observations = dict(session.reset())
                waiting, first = {}, time.perf_counter()  # observations back, their decisions not begun
                rounds: deque = deque()  # decisions begun, their answers due in this order
                in_flight = 0  # games stepping
                for i in range(len(self.games)):
                    obs = self._begin(i, observations[i], session)
                    if obs is None:
                        in_flight += 1
                    else:
                        waiting[i] = obs
                while self.rung == self.league["rung"]:
                    if self.args.steps and self.steps >= self.args.steps:
                        return
                    # each round's answers become commands as soon as they are in, and each observation is taken as it
                    # comes: nothing waits for the rest of a round
                    while rounds and (self.inference.connection.poll(0) or not (in_flight or waiting)):
                        in_flight += self._send_round(rounds.popleft(), session)
                    if (
                        waiting
                        and len(rounds) < MAX_ROUNDS
                        and (not in_flight or time.perf_counter() - first >= self.args.gather)
                    ):
                        self.stages.add("in_flight", len(rounds), scale=1.0)  # requests already out
                        rounds.append(self._begin_round(waiting))
                        waiting = {}
                    if not in_flight:
                        if rounds:
                            waited = time.perf_counter()
                            self.inference.connection.poll(0.05)
                            self.idle += time.perf_counter() - waited
                        continue
                    if rounds:
                        timeout = POLL
                    elif waiting:
                        timeout = max(0.0, first + self.args.gather - time.perf_counter())
                    else:
                        timeout = None
                    try:
                        waited = time.perf_counter()
                        got = session.recv(timeout=timeout)
                        self.idle += time.perf_counter() - waited
                    except RuntimeError as error:  # a game that failed (a hung game thread, a crash)
                        i = int(re.match(r"game (\d+):", str(error)).group(1))
                        in_flight -= 1
                        first = first if waiting else time.perf_counter()
                        if (obs := self._lost(i, session, error)) is None:
                            in_flight += 1
                        else:
                            waiting[i] = obs
                        continue
                    if got is not None:
                        i, obs, done, info = got
                        in_flight -= 1
                        self._arrived(i, info)
                        first = first if waiting else time.perf_counter()
                        if (obs := self._after(i, obs, done, info, session)) is None:
                            in_flight += 1
                        else:
                            waiting[i] = obs
                    self._sync()
                for deciding in rounds:  # the answers still on their way are read (they come in order), unused
                    self._decide_end(deciding)
            self.rung = self.league["rung"]
            self.drop_games()

    def _begin_round(self, observations: dict) -> tuple:
        now = time.perf_counter()
        for i in observations:
            if i in self.back_at:
                self.stages.add("queued", now - self.back_at.pop(i))
            self.begun_at[i] = now
        begun = self._decide_begin(observations)
        self.stages.add("begin", time.perf_counter() - now)
        self.stages.add("games", len(observations), scale=1.0)
        return begun

    def _send_round(self, deciding: tuple, session) -> int:
        """A round's answers as commands, sent; returns how many games step on."""
        began = time.perf_counter()
        actions, decisions = self._decide_end(deciding)
        self.stages.add("end", time.perf_counter() - began)
        for i, a in actions.items():
            session.send(i, a, decisions[i])
            self.sent_at[i] = time.perf_counter()
            if i in self.begun_at:
                self.stages.add("decide", self.sent_at[i] - self.begun_at.pop(i))
        return len(actions)

    def _arrived(self, i: int, info: dict) -> None:
        """Game i's observation is back: its time away, each part of the way (wc3env.vector's info["hops"])."""
        if i not in self.sent_at:
            return
        self.back_at[i] = time.perf_counter()
        self.stages.add("game", self.back_at[i] - self.sent_at.pop(i))
        for part, ms in info.get("hops", {}).items():
            self.stages.add(part, ms, scale=1.0)

    def _decide_begin(self, observations: dict) -> tuple:
        """The CPU's part of these games' decisions, sent to the model: one request per policy."""
        groups: dict[str, list] = {}
        for i, obs in observations.items():
            game = self.games[i]
            for slot, seat in game.seats.items():
                groups.setdefault(game.policy[slot], []).append((i, slot, seat, obs[slot], obs.get(1 - slot)))
        begun = []
        for policy, members in groups.items():
            agents = self._agents(policy)
            learning = policy == "latest"
            # the critic also reads the opponent's side: its scalars, from the features the hook computed for it
            opponents = [m[4].features["scalars"] for m in members] if learning else None
            blocks = [m[3].features for m in members]
            for block in blocks if getattr(self.mode, "allow", None) else ():  # the mode's legal types (combat)
                self.mode.allow(block)
            begun.append((agents, members, opponents, agents.begin([m[2] for m in members], [m[3] for m in members],
                                                                   blocks, opponents)))  # fmt: skip
        return observations, begun

    def _decide_end(self, deciding: tuple) -> tuple[dict, dict]:
        """The decisions _decide_begin sent for, as each game's commands and the decisions the hook folds into the
        next features ({slot: hook_decision}); the learning players' steps recorded."""
        observations, begun = deciding
        actions = {i: {slot: [] for slot in observations[i]} for i in observations}
        taken: dict[int, dict] = {i: {} for i in observations}
        for agents, members, opponents, handle in begun:
            learning = opponents is not None
            decisions = agents.end(handle)
            for k, ((i, slot, seat, obs, _), d) in enumerate(zip(members, decisions)):
                actions[i][slot] = d.commands
                if d.action is not None:
                    taken[i][slot] = hook_decision(d.action)
                if learning and slot == self.games[i].slot and not self.games[i].eval:  # self-play's other "latest"
                    # seat is a computer slot, whose units cast on their own: it plays, it is not learned from
                    game = self.games[i]
                    if d.chunk_start:
                        self._begin_chunk(i, slot, seat, d.start_state)
                    if slot in game.waiting:  # the chunk before this one ends where this step's value begins
                        self._ship(game, slot, game.waiting.pop(slot).chunk(bootstrap=d.value))
                    game.recorders[slot].add(obs, d, opponents[k])
        for i, obs in observations.items():  # the mode's scripted side, if any
            for slot, commands in self.games[i].episode.commands(obs).items():
                actions[i][slot] = actions[i][slot] + commands
        self.steps += len(observations)
        return actions, taken

    def _static(self, races: tuple[str, str]) -> Static:
        if races not in self.statics:
            self.statics[races] = Static.load(self.map, races)
        return self.statics[races]

    def _begin_chunk(self, i: int, slot: int, seat: Seat, state: list | None) -> None:
        game = self.games[i]
        old = game.recorders.get(slot)
        if old is not None and len(old):
            game.waiting[slot] = old
        game.recorders[slot] = Recorder(seat.static, seat.plan, state, self.version)

    def _after(self, i: int, obs: dict, done: bool, info: dict, session) -> dict | None:
        """Game i's step came back: its rewards recorded; at an episode's end, the result reported and the next
        begun. Returns the observations to decide on next (None: the next episode is being staged)."""
        game = self.games[i]
        if info.get("reset"):  # the map loaded again (send_reset below): the next episode begins
            return self._begin(i, obs, session)
        if game.staging:  # the step that staged the episode: its first observations
            game.staging = False
            if done:  # (the game ended in it: stage again in the next one)
                return self._begin(i, obs, session)
            game.episode.staged(obs)
            if i == 0 and not game.eval and self.args.replay_minutes and time.time() >= self.next_replay:
                stage = None if game.scenario else self.handicap(game)  # a scenario's: its level, in its name
                game.recording = Recording(self.args.seed, game.slot, game.opponent, stage)
                self.next_replay = time.time() + 60 * self.args.replay_minutes
            if game.recording is not None:
                game.recording.add(obs)
            return obs
        game.steps += 1
        final = info.get("final_observations") if done else obs
        step = game.episode.step(final, done, game.steps)
        if game.recording is not None:
            game.recording.add(final)
        for slot, rec in game.recorders.items():
            rec.extra["reward"][-1] = step.rewards[slot]
        if step.over:
            # what the league counts: a win 1, a loss or the time limit 0, a draw 0.5 (0 against the ladder rung)
            outcome, minutes = step.outcome, step.minutes
            score = {"victory": 1.0, "defeat": 0.0, "time limit": 0.0}.get(outcome, 0.0 if game.ladder else 0.5)
            gold = final[game.slot].score.get("gold_mined", 0)
            opponent = f"eval:{game.policy[game.slot]}" if game.eval else game.opponent  # the league leaves these out
            self.ended.append((opponent, outcome, score, minutes, gold / max(minutes, 1e-6), game.began, game.stage))
            game.began = game.stage = None
            if game.recording is not None:
                self.client.call("replay", game.recording.finish(outcome))
                game.recording = None
            for slot, rec in game.recorders.items():
                rec.extra["done"][-1] = True
                if len(rec):
                    self._ship(game, slot, rec.chunk(bootstrap=0.0))
                self._flush(game, slot)
            if step.reload:  # the map loaded again (a game at its time limit; a fights' game whose clock is long),
                # without waiting: a new process takes 2-3 s and the other games step on (a failure comes back
                # through recv, and _lost starts the game again)
                session.send_reset(i)
                return None
            return self._begin(i, obs, session)
        return obs

    def _lost(self, i: int, session, error) -> dict:
        """A failed game starts over; its unfinished chunks go with it (no result: it never ended)."""
        self.report(f"actor {self.args.seed}: game {i} failed ({error}); restarting it")
        self.failures += 1
        self._abandon(self.games[i])
        for slot in list(self.games[i].held):  # its finished chunks still count, scored over what there is
            self._flush(self.games[i], slot)
        return self._begin(i, session.reset_one(i), session)

    def drop_games(self) -> None:
        """Every game is gone (its set of games was replaced, or failed), and its unfinished chunks with it."""
        for g in self.games:
            for slot in list(g.held):
                self._flush(g, slot)
            g.recorders, g.waiting = {}, {}
            self._abandon(g)

    def _ship(self, game: Game, slot: int, chunk: dict) -> None:
        """A finished chunk joins its segment; a full segment (SEGMENT chunks) goes out together."""
        game.held.setdefault(slot, []).append(chunk)
        if len(game.held[slot]) >= SEGMENT:
            self._flush(game, slot)

    def _flush(self, game: Game, slot: int) -> None:
        """The slot's held chunks out as one segment (in order, one id): the learner's batches score them together."""
        chunks = game.held.pop(slot, [])
        if chunks:
            self.segments += 1
            for c in chunks:
                c["segment"] = np.int64(self.args.seed * 2**32 + self.segments)
            self.outbox.extend(chunks)

    def _abandon(self, game: Game) -> None:
        """A game that will not end (it failed, or its set of games is replaced): no longer running for the league."""
        game.began = game.stage = None
        game.recording = None

    def _sync(self) -> None:
        """Ship finished chunks and game results; the league's state now and then (the inference process follows
        the weights; the decisions carry the version they were made with)."""
        if self.outbox:
            self.chunks.call("chunks", self.outbox)
            self.outbox = []
        refresh = time.time() - self.last_refresh > self.args.refresh
        if self.ended or refresh:  # at least every --refresh seconds: the league waits only for actors it hears from
            running = [g.began for g in self.games if g.began is not None]
            self.client.call(
                "games", {"actor": self.args.seed, "ended": self.ended, "oldest": min(running, default=None)}
            )
            self.ended = []
        if refresh:
            self.last_refresh = time.time()
            _, self.league = self.client.call("league")
            rate = self.steps / (time.time() - self.started)
            now = time.perf_counter()
            self.stages.add("busy", 1 - self.idle / max(now - self.reported, 1e-6), scale=100.0)  # % of the time
            self.idle, self.reported = 0.0, now
            self.report(
                f"actor {self.args.seed}: {self.steps} game steps, {rate:.1f}/s, version {self.version}, "
                f"{self.failures} failed games; ms: {self.stages.line()}"
            )

    def reconnect_inference(self) -> None:
        """A new link to the inference process (answers to the failed round's requests may be on the old one), and
        new Agents: the process then has none of this actor's map data."""
        self.inference.close()
        self.inference = Client(address(self.args.inference), self.args.authkey.encode(), wait=300, compress=False)
        self.agents = {}

    def _chunks_client(self, wait: float) -> Client:
        return Client(address(chunks_address(self.args.learner)), self.args.authkey.encode(), wait=wait)

    def learner_alive(self) -> bool:
        """Whether the learner still answers (on a fresh connection, waiting up to a minute for a restart)."""
        try:
            self.client = Client(address(self.args.learner), self.args.authkey.encode(), wait=60)
            self.chunks = self._chunks_client(60)
            self.client.call("log", f"actor {self.args.seed}: reconnected")
            return True
        except Exception:  # noqa: BLE001
            return False

    def report(self, line: str) -> None:
        """Into the learner's log too: an actor's own output may be nowhere anyone can read (a sandbox)."""
        print(line, flush=True)
        try:
            self.client.call("log", line)
        except Exception:  # noqa: BLE001  the learner is gone; the print above remains
            pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--learner", required=True, help="host:port")
    p.add_argument("--authkey", default=os.environ.get("WC3_SELFPLAY_KEY", "wc3-selfplay"))
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--ai-games", type=int, default=1, help="of them, against Warcraft's AI")
    p.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="of them (the last), games that only evaluate: the "
        "ladder's rung at the learner's --eval-stage, each of its --evaluate policies in turn, not learned from",
    )
    p.add_argument("--group", type=int, default=2, help="games per worker process")
    p.add_argument(
        "--replay-minutes",
        type=float,
        default=10,
        help="one fight of the first game recorded for the run's replays/ every this many minutes (0: none)",
    )
    p.add_argument(
        "--scenarios",
        default="",
        help="fights mode: scenarios a share of the ladder's fights become, name:level comma-separated (e.g. "
        "focus_fire:0,focus_fire:1; training/selfplay/scenarios.py), each as likely; outside the ladder's count",
    )
    p.add_argument(
        "--scenario-share", type=float, default=0.25, help="the share of the ladder's fights --scenarios takes"
    )
    p.add_argument(
        "--gather",
        type=float,
        default=0.003,
        help="seconds the actor waits for more of its games' results before deciding those it has",
    )
    p.add_argument(
        "--mode", choices=modes.NAMES, default="game", help="whole games, or fights (training/selfplay/modes)"
    )
    p.add_argument("--map", default="", help="the games' map (default: the mode's)")
    p.add_argument(
        "--races", nargs=2, metavar=("OWN", "OPPONENT"), help="one matchup in every game (default: each in turn)"
    )
    p.add_argument("--plans", default="runs/plans.npz")
    p.add_argument("--refresh", type=float, default=20, help="seconds between pulls of the league's state")
    p.add_argument(
        "--remote",
        type=int,
        default=0,
        help="the games' workers connect in on this port from another "
        "machine (python -m wc3env.vector --connect host:port --workers games/group)",
    )
    p.add_argument(
        "--dial",
        default="",
        help="host:port where another machine's games' workers listen (python -m "
        "wc3env.vector --listen port --workers games/group): one connection out to it a worker",
    )
    p.add_argument(
        "--wine",
        action="store_true",
        help="the games' workers as Windows Python under Wine, here "
        "(a Linux host; $WINEPREFIX, a copy of it per worker)",
    )
    p.add_argument(
        "--game-cuts",
        type=int,
        default=1225,
        help="wc3hook's equivalent faster game code (wc3hook/cuts.c "
        "bits; 1225: 4.6%% less CPU a game-step, every step identical: docs/training/warcraftsim.md); 0: none",
    )
    p.add_argument(
        "--nice", type=int, default=0, help="with --wine: the games' niceness (the learner and inference first)"
    )
    p.add_argument(
        "--inference",
        required=True,
        help="host:port of the inference process deciding for this actor (training/selfplay/inference.py)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="stop after this many game steps (0: never; for tests)")
    return p


def _watchdog(learner: str, every: float = 60, misses: int = 3) -> None:
    """Ends the process once the learner's port has not answered `misses` times in a row: closing the games after
    the learner went away could hang (sp3's sandboxes outlived their learner by 25 minutes)."""
    missed = 0
    while missed < misses:
        time.sleep(every)
        try:
            socket.create_connection(address(learner), timeout=10).close()
            missed = 0
        except OSError:
            missed += 1
    print(f"the learner at {learner} is gone; exiting", flush=True)
    os._exit(0)


def _wine(args: argparse.Namespace):
    """wine_workers, numbered past the other actors' workers: each worker its own prefix and wineserver, none shared
    with another actor's games (actor 0's first worker in $WINEPREFIX itself)."""
    wine, first = wine_workers(nice=args.nice), args.seed * -(-args.games // args.group)
    return lambda worker: wine(first + worker)


def main() -> None:
    """An actor that keeps going: after a failure outside a game (a launch that timed out on a busy machine,
    say) it reports it and starts a new set of games, up to MAX_RESTARTS times."""
    args = parser().parse_args()
    actor = Actor(args)  # waits for the learner (up to 30 minutes) before the watchdog starts counting
    # the games run here, or elsewhere (python -m wc3env.vector --listen / --connect): one listener, kept across restarts
    launch = (
        dial_workers(args.dial)
        if args.dial
        else remote_workers(args.remote)
        if args.remote
        else _wine(args)
        if args.wine
        else local_workers
    )
    threading.Thread(target=_watchdog, args=(args.learner,), daemon=True).start()
    for attempt in range(MAX_RESTARTS + 1):
        try:
            actor.run(launch)
            return
        except Exception:
            if not actor.learner_alive():  # the learner stopped (its time limit, or stop): so does the actor
                print(f"actor {actor.args.seed}: the learner is gone; stopping", flush=True)
                return
            actor.report(f"actor {actor.args.seed} failed (attempt {attempt + 1}): {traceback.format_exc()[-1500:]}")
            actor.drop_games()
            actor.reconnect_inference()
            time.sleep(10)
    raise SystemExit(f"actor {actor.args.seed}: too many failures")


if __name__ == "__main__":
    main()
