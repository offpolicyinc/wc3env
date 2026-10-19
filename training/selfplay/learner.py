"""The self-play learner: PPO from the cloned policy, kept near it (docs/training/5-trainer.md).

    python -m training.selfplay.learner --init runs/bc10/last.pt --out runs/selfplay/sp1

Actors (training/selfplay/actor.py) send 64-step chunks; every `batch` chunks make one update:

- the critic (model.value: the core and the opponent's scalars) gives values; GAE (gamma, lambda) gives
  advantages and returns, the chunk's bootstrap closing each;
- PPO's clipped objective on each step's log-probability (the sum over the heads its action used), recomputed
  from the chunk's start memory with the masks it was sampled under;
- the KL from a frozen reference (the clone, unless --reset-every moves it) to the learner on the actors' states,
  per head, weighted by the reference's probabilities (AlphaStar, DI-star, VPT; --kl 0.005) - it keeps what cloning
  taught in the situations the agent reaches without charging much for what the clone rarely did - and a small
  entropy bonus (--entropy);
- the first `warmup` updates train only the critic, the policy frozen: the cloned value is not fit for the
  agent's own games, and noisy early advantages would push the policy anywhere.

The values in GAE are the ones the actors recorded with each step (the critic as the inference process had it),
so the update needs no value pass of its own (--no-actor-values: the learner's). Chunks whose weights are more than
--max-lag updates old are dropped instead of trained on; the batches are put together in a process of their own
(batches.py) while the GPU trains.

Every `snapshot_every` updates a frozen copy joins the league. Losses on negative advantages are dual-clipped (Ye
et al. 2020). Logs are plain lines in <out>/log.txt; <out>/last.pt (saved every 10 updates) resumes a run, its time
included; with no update for --stall-minutes the process exits with code 3.
"""

from __future__ import annotations

import argparse
import collections
import copy
import gc
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..policy import model as policy_model
from functools import cache

from ..policy.actions import TYPES, reference
from ..policy.masks import ORDER_IDS, argument_ids
from ..policy.agent import CHUNK
from ..policy.features import MAX_UNITS
from ..policy.model import ROW_SIZES, float32, load, masked_mean, pack, row_size
from ..replays.dataset import ID_FIELDS
from ..train.loader import ReplayData
from ..train.track import Track
from . import modes
from .batches import Batches, chunks_address
from .league import START_MINUTES, League
from .reward import CHANNELS, weights
from .transport import Packed, Server, address


def _rows(batch: dict, state: list | None, rows: torch.Tensor) -> tuple[dict, list | None]:
    """Some chunks of a batch (every entry is per chunk first), with their start memory."""
    return {k: v[rows] for k, v in batch.items()}, None if state is None else [h[rows] for h in state]


def _var(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x[mask].var() (unbiased) without waiting for the GPU to count the mask, as masked_mean."""
    m = mask.to(x.dtype)
    return (((x - masked_mean(x, mask)) ** 2) * m).sum() / (m.sum() - 1).clamp(min=1)


class _Clock:
    """Seconds per part of an update, the GPU's work finished at each lap (stats: time_<part>)."""

    def __init__(self, cuda: bool):
        self.cuda, self.laps, self.last = cuda, {}, time.perf_counter()

    def lap(self, name: str) -> None:
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.laps[name] = now - self.last
        self.last = now


EVAL_WINDOW = 400  # evaluation games per policy the log's share counts (the last)


def eval_label(path: str) -> str:
    """An evaluated checkpoint's name: its run's (runs/cb8/last.pt: cb8), or the file's (checkpoints/cb9.pt: cb9)."""
    p = Path(path)
    return p.parent.name if p.stem == "last" else p.stem


def gammas(args) -> np.ndarray:
    """Each reward channel's discount [CHANNELS]: --gamma, unless --channel-gamma names the channel."""
    out = np.full(len(CHANNELS), args.gamma, np.float32)
    for item in filter(None, args.channel_gamma.split(",")):
        name, value = item.split("=")
        out[CHANNELS.index(name.strip())] = float(value)
    return out


def gae(reward, value, done, bootstrap, valid, gamma, lam: float):
    """Advantages and returns [B, T, ...] (GAE), each reward channel on its own, at its own discount (`gamma` a float
    or [CHANNELS]): `bootstrap` [B, ...] is the value after a chunk's last real step; done and valid [B, T]."""
    t = reward.shape[1]
    extra = (None,) * (reward.dim() - 2)  # done and valid broadcast over the channels
    advantage = torch.zeros_like(reward)
    last = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for i in reversed(range(t)):
        live = valid[(slice(None), i, *extra)]
        not_done = (~done[(slice(None), i, *extra)]).float()
        delta = reward[:, i] + gamma * next_value * not_done - value[:, i]
        last = delta + gamma * lam * not_done * last
        last = torch.where(live, last, torch.zeros_like(last))
        advantage[:, i] = last
        next_value = torch.where(live, value[:, i], next_value)
    return advantage, advantage + value


def upgo(reward, value, done, bootstrap, valid, gamma):
    """UPGO's advantages [B, T, ...] (batches.upgo, for chunks scored on their own): each step's return follows what
    came after while every next step did at least as well as the critic expected, else the critic's value."""
    t = reward.shape[1]
    extra = (None,) * (reward.dim() - 2)
    advantage = torch.zeros_like(reward)
    after = follow = bootstrap
    for i in reversed(range(t)):
        live = valid[(slice(None), i, *extra)]
        on = (~done[(slice(None), i, *extra)]).float()
        g = reward[:, i] + gamma * on * follow
        advantage[:, i] = torch.where(live, g - value[:, i], torch.zeros_like(g))
        good = reward[:, i] + gamma * on * after >= value[:, i]
        follow = torch.where(live, torch.where(good, g, value[:, i]), follow)
        after = torch.where(live, value[:, i], after)
    return advantage


# the order types whose argument the learner logs, each argument's share of the steps as <prefix><name> (spell_hex,
# item_potion_of_healing, learn_wind_walk)
ARGUMENT_LOGS = (("spell_", ("cast", "cast_point", "cast_unit")), ("item_", ("use_item",)), ("learn_", ("learn",)))


@cache
def argument_names() -> list[str]:
    """Each argument index's name (masks.argument_ids, 0: none): a cast's order ('hex'), an item's or an ability's."""
    ref, orders = reference(), {}
    for ability in ref["abilities"].values():
        for level in ability["levels"].values():
            for o in level.get("orders", []):
                if o.get("order_id") is not None:
                    orders.setdefault(o["order_id"], o["name"])
    ids = argument_ids()
    names = ["none"] * (len(ids) + 1)
    for value, i in ids.items():
        code = "" if value in ORDER_IDS else value.to_bytes(4, "big").decode("latin-1")
        name = orders.get(value) or ref["items"].get(code, {}).get("name") or ref["abilities"].get(code, {}).get("levels", {}).get("1", {}).get("name") or code or str(value)
        names[i] = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return names


def prepare(chunks: list[dict], widths: list[int], allocate) -> dict:
    """A batch's CPU part, no GPU: each array written once, straight into memory from allocate(shape, dtype) ->
    (what is kept, a numpy view to fill): pinned tensors in Learner._host, a file's pages in batches.py (stacking first
    and copying after moved each of a batch's ~360 MB twice). The unit rows go unpadded (placed on the GPU: padded to
    MAX_UNITS here they were most of the batch's bytes and of its building time)."""
    t, n = CHUNK, len(chunks)

    def buffer(shape, dtype, fill=None):
        out, view = allocate(shape, dtype)
        if fill is not None:
            view.fill(fill)
        return out, view

    def stacked(arrays: list):
        out, view = buffer((len(arrays), *arrays[0].shape), arrays[0].dtype)
        for i, a in enumerate(arrays):
            view[i] = a
        return out

    def joined(arrays: list):
        out, view = buffer((sum(len(a) for a in arrays), *arrays[0].shape[1:]), arrays[0].dtype)
        at = 0
        for a in arrays:
            view[at : at + len(a)] = a
            at += len(a)
        return out

    def per_step(arrays: list, dtype, fill=0):
        out, view = buffer((n, t, *np.shape(arrays[0])[1:]), dtype, fill)
        for i, a in enumerate(arrays):
            view[i, : len(a)] = a
        return out

    data = [ReplayData(c) for c in chunks]
    rows = [r.chunk(0, t, rows=False) for r in data]
    h = {"chunks": n, "versions": np.array([int(c["version"]) for c in chunks])}
    h["dense"] = {k: stacked([r[k] for r in rows]) for k in rows[0]}
    units = [r.unit_rows(0, t) for r in data]
    h["where"] = [joined([np.full(len(u["step"]), i) for i, u in enumerate(units)])]
    h["where"] += [joined([u[k] for u in units]) for k in ("step", "row")]
    h["units"] = {k: joined([u[k] for u in units]) for k in ("units", *ID_FIELDS)}
    h["steps"] = {"old_log_prob": per_step([c["log_prob"] for c in chunks], np.float32),
                  "reward": per_step([c["reward"] for c in chunks], np.float32),
                  "done": per_step([c["done"] for c in chunks], bool, True),
                  "actor_value": per_step([c["value"] for c in chunks], np.float32, np.nan),
                  "bootstrap": stacked([np.float32(c["bootstrap"]) for c in chunks]),
                  **({"advantage": per_step([c["advantage"] for c in chunks], np.float32),
                      "returns": per_step([c["returns"] for c in chunks], np.float32),
                      "upgo": per_step([c["upgo"] for c in chunks], np.float32)}
                     if all("upgo" in c for c in chunks) else {}),
                  "opponent": per_step([c["opponent"] for c in chunks], np.float32, 0)}  # fmt: skip
    # read here, on the CPU: the unit rows each chunk needs, and whether every step carries the actor's value
    h["fullest"] = np.array([int(np.max(c["unit_count"], initial=1)) for c in chunks])
    h["actor_values"] = all(np.isfinite(c["value"]).all() for c in chunks)
    h["state"] = None
    if any(c["start_state"] is not None for c in chunks):
        h["state"] = [stacked([c["start_state"][k] if c["start_state"] is not None else np.zeros(w, np.float32) for c in chunks])
                      for k, w in enumerate(widths)]  # fmt: skip
    return h


class Learner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        (self.out / "pool").mkdir(parents=True, exist_ok=True)
        self.log = (self.out / "log.txt").open("a", buffering=1)
        # every ladder game's start, end and result, a JSON line each: results by when games began
        # (training/cloud/selfplay_dashboard.py), which the running totals, counted as games finish, cannot give
        self.games_log = (self.out / "games.jsonl").open("a", buffering=1)
        self.track = Track(self.out.name, {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}, "selfplay")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gammas = gammas(args)  # [CHANNELS]: each reward channel's discount
        self.gamma = torch.from_numpy(self.gammas).to(self.device)
        start = args.start or args.init  # the weights trained on; --init is the KL's anchor either way
        self.model = load(start, self.device, eval=False)
        self.reference = load(args.init, self.device)  # the KL's anchor, frozen
        for p in self.reference.parameters():
            p.requires_grad_(False)
        # the step transformer on real tokens only, FlashAttention-2 per step (a mid-game minibatch is ~73% of its
        # padded tokens; a block 12.7 -> 9.5 ms forward + backward on an H100, training/bench/attention_varlen.py)
        self.varlen = self.device.type == "cuda" and policy_model.flash_attn_varlen_func is not None
        if args.compile and self.device.type == "cuda":  # the blocks' many small kernels fused
            dynamic = None if self.varlen else False  # packed: the token count changes with every piece
            self.model.compile_blocks(dynamic)
            self.reference.compile_blocks(dynamic)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        # a learner's checkpoint carries Adam's moments on into the new run (warcraftsim: each fork with a fresh Adam
        # lost ~10 points in its first 20 minutes, none with them); this run's --lr, not the old one's
        saved = torch.load(start, map_location=self.device, weights_only=False) if args.carry_adam else {}
        if "optimizer" in saved and "league" in saved:  # a self-play learner's (a cloning run's Adam is not this one's)
            try:
                self.opt.load_state_dict(saved["optimizer"])
                for group in self.opt.param_groups:
                    group["lr"] = args.lr
                self.say(f"Adam's state carried from {start}")
            except ValueError as e:  # the model changed since (a new part): its moments do not fit
                self.say(f"Adam's state not carried from {start}: {e}")
        del saved
        self.warmed = False  # every unit row size's compiled programs made (_warm, before the first update)
        self.update, self.version = 0, 0
        self.seconds_done = 0.0  # training time of earlier starts of this run: --hours counts the whole run
        self.mode = modes.get(args.mode)  # the ladder and its stages, the shaping's weight
        ladder = self.mode.ladder
        if args.start_rung and args.start_rung not in ladder:
            raise SystemExit(f"--start-rung {args.start_rung}: not on this mode's ladder {ladder}")
        if args.start_stage and args.start_stage not in self.mode.handicaps:
            raise SystemExit(f"--start-stage {args.start_stage}: not one of this mode's {self.mode.handicaps}")
        self.league = League(ladder=ladder, handicaps=self.mode.handicaps, minutes=args.minutes, hold=args.hold_stage,
                             level=ladder.index(args.start_rung) if args.start_rung else 0,
                             stage=self.mode.handicaps.index(args.start_stage) if args.start_stage else 0)
        self.tries = np.zeros(len(TYPES), np.int64)  # each order type, how often chosen in all the steps trained on
        self.tried_at = None  # the (rung, stage) self.tries counts in: a promotion starts the count again
        self.floor = None  # [len(TYPES)] each type's least chance this update: --explore-floor, --explore-min, or 0
        resumed_eval = None  # the evaluation stage a resumed run had
        self.games = 0  # games ended in the whole run (play/games, the dashboard's x): a resumed run counts on
        last = self.out / "last.pt"
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(saved["model"])
            self.opt.load_state_dict(saved["optimizer"])
            self.update, self.league = saved["update"], saved["league"]
            self.seconds_done = saved.get("seconds_done", 0.0)
            self.tries = saved.get("tries", self.tries)
            self.tried_at = saved.get("tried_at")
            if "games" in saved:
                self.games = saved["games"]
            elif (self.out / "games.jsonl").exists():  # saved before the count was (cb8): one line a game
                with (self.out / "games.jsonl").open("rb") as f:
                    self.games = sum(1 for _ in f)
            if "reference" in saved:  # the KL's anchor as it was (a reset may have moved it off the clone)
                self.reference.load_state_dict(saved["reference"])
            self.version = self.update
            resumed_eval = saved.get("eval_handicap")
            self.say(f"resumed at update {self.update}")
        self.lock = threading.Lock()
        self.weights = self._weights()
        # (version, the weights' reply packed once for every pull): this one here, each later one by _pack
        self.reply = (self.version, Packed((self.version, self.weights, None), compress=False))
        self.outcomes: dict[str, int] = {}  # every game's outcome counted (any span of games: differences of these)
        self.saving: threading.Thread | None = None  # the thread writing last.pt
        self.save_seconds = self.write_seconds = 0.0  # the last save: copying it here, writing it there
        self.recent_results: collections.deque = collections.deque(maxlen=200)
        # evaluation games (actors' --eval-games): the current policy ("latest") and each --evaluate checkpoint, in
        # turn on the same draws, at one fixed stage: policies compared under the same conditions at the same time
        self.evaluate = ["latest"]
        for path in filter(None, args.evaluate.split(",")):
            name = "eval-" + eval_label(path)
            torch.save(load(path).state_dict(), self.out / "pool" / f"{name}.pt")  # served as snapshots are
            self.evaluate.append(name)
        # default: the stage the run started on (kept over resumes: one run's evaluation games all at one stage)
        self.eval_handicap = args.eval_stage or resumed_eval or self.league.handicap
        self.eval_results = {p: collections.deque(maxlen=EVAL_WINDOW) for p in self.evaluate}

    def say(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {line}", flush=True)
        self.log.write(f"{stamp} {line}\n")
        self.track.line(line)

    def _weights(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ---- the actors' requests
    def handle(self, kind: str, payload):
        if kind == "games":  # an actor's report: the games that ended, and when its oldest running ladder game began
            with self.lock:
                ended = payload["ended"]
                self.league.report(payload["actor"], [(e[0], e[2], e[5], e[6]) for e in ended], payload["oldest"])
                for opponent, outcome, _, minutes, gold_per_minute, began, stage in ended:
                    if opponent.startswith("eval:"):  # an evaluation game: its policy's record, nothing else
                        policy = opponent[5:]
                        self.eval_results.setdefault(policy, collections.deque(maxlen=EVAL_WINDOW)).append(outcome)
                        self.games_log.write(json.dumps({"ended": round(time.time(), 1), "eval": policy, "outcome": outcome,
                                                         "minutes": round(minutes, 2), "update": self.update,
                                                         "stage": self.eval_handicap}) + "\n")
                        continue
                    self.games += 1
                    if began is not None:  # a ladder game, with the stage it began under: [rung, handicap]
                        handicap = None if stage is None else self.league.handicaps[min(stage[1], len(self.league.handicaps) - 1)]
                        self.games_log.write(json.dumps({"began": round(began, 1), "ended": round(time.time(), 1), "opponent": opponent,
                                                         "outcome": outcome, "minutes": round(minutes, 2), "update": self.update,
                                                         "stage": None if stage is None else [stage[0], handicap]}) + "\n")
                    else:  # a self-play game (from cb6): player 0's outcome, for the dashboard's blocks of games
                        self.games_log.write(json.dumps({"ended": round(time.time(), 1), "opponent": opponent, "outcome": outcome,
                                                         "minutes": round(minutes, 2), "update": self.update}) + "\n")
                    self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
                    if opponent in self.league.ladder:  # the ladder's alone, for the stop test (build/stop_check.py)
                        self.outcomes["ladder " + outcome] = self.outcomes.get("ladder " + outcome, 0) + 1
                    self.recent_results.append((opponent, outcome, minutes, gold_per_minute))
            return "ok"
        if kind == "weights":  # (version, the weights or None when the caller has them, None)
            with self.lock:
                version, reply = self.reply
            return (version, None, None) if payload == version else reply
        if kind == "league":  # the league without the weights (actors whose model runs in the inference process)
            with self.lock:
                return self.version, {**self.league.plan(), "evaluate": self.evaluate, "eval_handicap": self.eval_handicap}
        if kind == "log":
            self.say(payload)
            return "ok"
        if kind == "snapshot":
            return torch.load(self.out / "pool" / f"{payload}.pt", map_location="cpu", weights_only=True)
        raise ValueError(kind)

    # ---- one update
    def _batch(self, chunks: list[dict]) -> tuple[dict, list | None]:
        return self._to_device(self._host(chunks))

    def _host(self, chunks: list[dict]) -> dict:
        """A batch's arrays on the CPU (prepare()), in page-locked memory on a GPU machine: the copies to it then run
        at full speed without waiting. Benchmarks and tests; a run's batches come from the batch process (batches.py)."""
        pin = self.device.type == "cuda"

        def allocate(shape, dtype) -> tuple[torch.Tensor, np.ndarray]:
            out = torch.empty(shape, dtype=torch.from_numpy(np.empty(0, dtype)).dtype, pin_memory=pin)
            return out, out.numpy()

        return prepare(chunks, [m.width for m in self.model.memory], allocate)

    def _to_device(self, h: dict) -> tuple[dict, list | None]:
        """The batch on the device from _host's arrays."""
        dev, t = self.device, CHUNK
        batch = {k: v.to(dev, non_blocking=True) for k, v in h["dense"].items()}
        where = [w.to(dev, non_blocking=True) for w in h["where"]]
        for k, v in h["units"].items():
            v = v.to(dev, non_blocking=True)
            dtype = torch.float16 if k == "units" else torch.int32
            grid = torch.zeros(h["chunks"], t, MAX_UNITS, *v.shape[1:], dtype=dtype, device=dev)
            grid[where[0], where[1], where[2]] = v.to(dtype)
            batch[k] = grid
        for k, v in h["steps"].items():
            batch[k] = v.to(dev, non_blocking=True)
        state = None if h["state"] is None else [x.to(dev, non_blocking=True) for x in h["state"]]
        return batch, state

    def step(self, chunks: list[dict] | None, host: dict | None = None) -> dict:
        """One update: the cloned policy's distributions on the whole batch (no gradients; and the learner's values,
        unless the actors' are used), GAE, then `epochs` passes over it in shuffled minibatches of `minibatch`
        chunks, one gradient step each, each run through the GPU `micro` chunks at a time with their gradients added
        up (a smaller GPU, same training). `host`: _host(chunks), made ahead."""
        a = self.args
        cuda = self.device.type == "cuda"
        clock = _Clock(cuda)
        if host is None:
            host = self._host(chunks)
            clock.lap("time_host")
        batch, state = self._to_device(host)
        valid = batch["step_valid"].bool()
        if not self.warmed:
            self._warm(batch, state)
        clock.lap("time_device")
        # the exploration floor: every order type but noop at least --explore-floor until tried --explore-tries times at
        # this stage (a promotion re-arms it: what did not pay against a weaker opponent may against a stronger), then
        # at least --explore-min, so none dies out
        if (self.league.rung, self.league.stage) != self.tried_at:
            if self.tried_at is not None:
                self.say(f"stage {self.league.handicap:g}: exploration floor re-armed")
            self.tries[:], self.tried_at = 0, (self.league.rung, self.league.stage)
        eligible = np.arange(len(TYPES)) > 0
        if getattr(self.mode, "types", None) is not None:  # only the mode's (combat: FIGHT_TYPES)
            eligible &= self.mode.types
        fresh = eligible & (self.tries < a.explore_tries)
        floor = np.where(fresh, a.explore_floor, np.where(eligible, a.explore_min, 0.0))
        self.floor = torch.from_numpy(floor.astype(np.float32)).to(self.device) if floor.any() else None
        n = host["chunks"]
        micro = a.micro or a.minibatch
        fullest = host["fullest"]
        counts = host["dense"]["unit_count"].numpy()

        def piece(chunks: np.ndarray) -> tuple[int, tuple | None]:  # a piece's row size, and its packing
            size = row_size(fullest[chunks].max())
            return size, (pack(counts[chunks].reshape(-1), size, self.device) if self.varlen else None)

        parts = [(torch.arange(k, min(k + micro, n), device=self.device), *piece(np.arange(k, min(k + micro, n))))
                 for k in range(0, n, micro)]  # fmt: skip
        actor_values = a.actor_values and host["actor_values"]
        values, references = [], []
        with torch.no_grad():
            for rows, size, packing in parts:
                b, st = _rows(batch, state, rows)
                with torch.autocast(self.device.type, torch.bfloat16, enabled=cuda):
                    if not actor_values:
                        values.append(self.model.value(self.model(b, st, rows=size, pack=packing), b["opponent"]).float())
                    reference = float32(self.reference(b, st, rows=size, pack=packing))
                references.append({k: v[0] for k, v in self.reference.log_probs(b, reference)[1].items()})
        value = torch.where(valid[..., None], batch["actor_value"], 0.0) if actor_values else torch.cat(values)
        reference_lp = {k: torch.cat([r[k] for r in references]) for k in references[0]}
        clock.lap("time_score")
        # per reward channel [B, T, CHANNELS], then weighed by each channel's weight now (reward.weights: the shaping
        # fades with the win rate) into one advantage
        # two lambdas, as AlphaStar's (V-trace ~1 for the policy, TD(0.8) for its baseline): the policy is scored on the
        # real rewards up to the chunk's end, the critic trained on smoother targets that lean on its next guesses
        scored = actor_values and "upgo" in batch  # over each chunk's whole segment, in the batch process
        if scored:
            channels, returns, upgo_channels = batch["advantage"], batch["returns"], batch["upgo"]
        else:  # each chunk on its own
            channels, _ = gae(batch["reward"], value, batch["done"], batch["bootstrap"], valid, self.gamma, a.lam)
            _, returns = gae(batch["reward"], value, batch["done"], batch["bootstrap"], valid, self.gamma, a.critic_lam)
            upgo_channels = upgo(batch["reward"], value, batch["done"], batch["bootstrap"], valid, self.gamma) if a.upgo else None
        # the shaping's weight: the league's fading one, or the mode's own (combat: always 1, modes/combat.py)
        weight = torch.from_numpy(weights(self.mode.shaping(self.league.shaping()), a.shaping_scale)).to(self.device)
        advantage = (channels * weight).sum(-1)
        spread = _var(advantage, valid).sqrt()  # in reward units (a win is 1): logged, as it now sets the step's size
        # UPGO's (--upgo): scaled by the same spread but not centred, its sign is its point (better than expected)
        upgo_advantage = (upgo_channels * weight).sum(-1) if a.upgo else None
        if a.upgo:
            upgo_mean = masked_mean(upgo_advantage, valid)  # in reward units: how much better than expected
            if a.normalize_advantage:
                upgo_advantage = upgo_advantage / (spread + 1e-8)
        if a.normalize_advantage:  # the PPO libraries' default (baselines, CleanRL, SB3); AlphaStar and DI-star do not
            advantage = (advantage - masked_mean(advantage, valid)) / (spread + 1e-8)
        # the critic learns normalised returns; the first updates set the statistics (a mean over all so far), then they
        # follow at popart_rate
        self.model.update_value_statistics(returns[valid], max(a.popart_rate, 1 / (self.update + 1)))
        target = (returns - self.model.value_mean) / self.model.value_std
        # statistics stay on the GPU until the update's end (a float() of each waited for the GPU: ~270 waits an update)
        stats = collections.defaultdict(lambda: torch.zeros((), device=self.device))
        stats["advantage_std"] = spread
        if a.upgo:
            stats["upgo_mean"] = upgo_mean
        steps = a.epochs * -(-n // a.minibatch)  # optimizer steps in this update

        def probe():  # the policy's log-probabilities of the chosen actions on the first piece, and its live steps
            rows, size, packing = parts[0]
            b, st = _rows(batch, state, rows)
            with torch.no_grad(), torch.autocast(self.device.type, torch.bfloat16, enabled=cuda):
                return self.model.log_probs(b, float32(self.model(b, st, rows=size, pack=packing)))[0], b["step_valid"].bool()

        before = probe() if self.update >= a.warmup else None
        for epoch in range(a.epochs):
            order_cpu = torch.randperm(n)  # on the CPU too: each piece's unit row size is read from it
            order = order_cpu.to(self.device)
            for k in range(0, n, a.minibatch):
                minibatch = order[k : k + a.minibatch]
                self.opt.zero_grad(set_to_none=True)
                for j in range(0, len(minibatch), micro):
                    rows = minibatch[j : j + micro]
                    size, packing = piece(order_cpu[k + j : k + j + micro].numpy())
                    share = len(rows) / len(minibatch)  # this piece's part of the minibatch's mean loss
                    b, st = _rows(batch, state, rows)
                    live = valid[rows]
                    with torch.autocast(self.device.type, torch.bfloat16, enabled=cuda):
                        out = self.model(b, st, rows=size, pack=packing)
                        predicted = self.model.normalized_value(out, b["opponent"]).float()
                        value_loss = masked_mean(0.5 * ((predicted - target[rows]) ** 2).sum(-1), live)
                    loss = a.value_coef * value_loss
                    if self.update >= a.warmup:  # after the warm-up the policy learns too
                        out32 = float32(out)
                        reference_rows = {name: lp[rows] for name, lp in reference_lp.items()}
                        loss = loss + self._policy_loss(b, out32, advantage[rows], reference_rows, epoch, stats, steps / share,
                                                        None if upgo_advantage is None else upgo_advantage[rows])
                    (loss * share).backward()
                    stats["value_loss"] += value_loss.detach() * share / steps
                stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.clip_grad) / steps
                self.opt.step()
                if a.update_pause_ms and cuda:  # the GPU to the inference process alone for a moment (one GPU for both)
                    torch.cuda.synchronize(self.device)
                    time.sleep(a.update_pause_ms / 1000)
        if before is not None:  # update_kl: how far this update moved the policy, KL(before || after) on the first
            # piece's steps (Schulman's k3: (r - 1) - log r, r = after / before, on actions the actors drew from about
            # `before`). The adaptive schedules aim at ~0.01 a step (rsl_rl, rl_games); kl is the drift from --init
            log_r = probe()[0] - before[0]
            stats["update_kl"] = masked_mean(torch.expm1(log_r) - log_r, before[1])
        count = batch["unit_count"].float()
        stats["units_mean"], stats["units_max"] = masked_mean(count, valid), count.masked_fill(~valid, 0).max()
        win = CHANNELS.index("win")
        stats["value_explained"] = 1 - _var(returns[..., win] - value[..., win], valid) / (_var(returns[..., win], valid) + 1e-8)
        for k, name in enumerate(CHANNELS):  # each channel's critic fit; "weighted": of the channels as the advantage adds them
            stats[f"explained_{name}"] = 1 - _var(returns[..., k] - value[..., k], valid) / (_var(returns[..., k], valid) + 1e-8)
        mixed = lambda x: (x * weight).sum(-1)  # noqa: E731
        stats["explained_weighted"] = 1 - _var(mixed(returns) - mixed(value), valid) / (_var(mixed(returns), valid) + 1e-8)
        stats["return_std"] = _var(returns[..., win], valid).sqrt()
        stats["shaping"] = torch.tensor(self.league.shaping(), device=self.device)
        for k, name in enumerate(CHANNELS):  # each channel's reward per game-minute (240 steps)
            stats[f"reward_{name}"] = masked_mean(batch["reward"][..., k], valid) * 240
        chosen = torch.bincount(batch["type"][valid].long(), minlength=len(TYPES))  # this batch's order types
        for k, name in enumerate(TYPES):  # each type's share of the steps, and how often it has been chosen in all
            stats[f"share_{name}"] = chosen[k] / valid.sum().clamp(min=1)
        stats["floored"] = torch.tensor(float((fresh & (a.explore_floor > 0)).sum()), device=self.device)
        names = list(stats)
        out = dict(zip(names, torch.stack([stats[k].float().reshape(()) for k in names]).tolist()))  # one wait
        self.tries += np.round([out[f"share_{name}"] * float(valid.sum()) for name in TYPES]).astype(np.int64)
        out.update({f"tries_{name}": int(n) for name, n in zip(TYPES, self.tries) if name != "noop"})
        # which spells, items and skills: each argument's share of the steps (only those chosen this update)
        names, kind, argument = argument_names(), batch["type"][valid].long(), batch["argument"][valid].long()
        for prefix, kinds in ARGUMENT_LOGS:
            chosen = argument[torch.isin(kind, torch.tensor([TYPES.index(k) for k in kinds], device=kind.device))]
            for i, n in enumerate(torch.bincount(chosen, minlength=len(names)).tolist()):
                if n and i:
                    out[prefix + names[i]] = out.get(prefix + names[i], 0.0) + n / max(int(valid.sum()), 1)
        clock.lap("time_train")
        out.update(clock.laps)
        out["actor_values"] = float(actor_values)
        out["segments"] = float(scored)
        out["policy_lag"] = float(self.version - host["versions"].mean())
        return out

    def _warm(self, batch, state) -> None:
        """Before the first update: the compiled blocks' programs for every unit row size, with and without
        gradients, on this batch's first piece (no weight changes). Made on first need instead, a size first seen
        mid-run stopped the updates ~3.5 s (sp24, at 96 rows and again at 128; training/bench/learner_sizes.py)."""
        self.warmed = True
        if not (self.args.compile and self.device.type == "cuda"):  # nothing compiled
            return
        first = min(self.args.micro or self.args.minibatch, len(batch["reward"]))
        b, st = _rows(batch, state, torch.arange(first, device=self.device))
        counts = b["unit_count"].reshape(-1).cpu().numpy()
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
            for size in ROW_SIZES:
                packing = pack(counts, size, self.device) if self.varlen else None
                with torch.no_grad():
                    self.reference(b, st, rows=size, pack=packing)
                    self.model(b, st, rows=size, pack=packing)
                self.model(b, st, rows=size, pack=packing)["core"].float().mean().backward()
        self.opt.zero_grad(set_to_none=True)

    def _policy_loss(self, batch, out, advantage, reference_lp, epoch, stats, steps, upgo_advantage=None):
        """PPO's clipped loss and the KL to the cloned policy. The heads run in float32: their log-probabilities are
        compared with the actors'."""
        a = self.args
        valid = batch["step_valid"].bool()
        log_prob, per = self.model.log_probs(batch, out)
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        if epoch == 0:  # before any change the ratios are 1, up to the actors' older weights and rounding
            # actor_kl: KL(actors' || learner's) on the actors' actions (k3), the actors' lag and this update's steps so far
            stats["actor_kl"] += masked_mean((ratio - 1 - (log_prob - batch["old_log_prob"])).detach(), valid) / steps
            drift = (ratio - 1).abs().detach()
            stats["ratio_drift"] = torch.maximum(stats["ratio_drift"], drift.masked_fill(~valid, 0).max())
        clipped = torch.clamp(ratio, 1 - a.clip, 1 + a.clip)
        surrogate = torch.min(ratio * advantage, clipped * advantage)
        # dual clip (Ye et al. 2020): a bad action that stale data says became far likelier (ratio 525 in the second
        # pilot) still counts at most dual_clip times its advantage, so one sample cannot yank the whole policy
        surrogate = torch.where(advantage < 0, torch.max(surrogate, a.dual_clip * advantage), surrogate)
        policy_loss = -masked_mean(surrogate, valid)
        if upgo_advantage is not None:  # UPGO's term beside it (--upgo), clipped the same way
            u = torch.min(ratio * upgo_advantage, clipped * upgo_advantage)
            u = torch.where(upgo_advantage < 0, torch.max(u, a.dual_clip * upgo_advantage), u)
            upgo_loss = -masked_mean(u, valid)
            policy_loss = policy_loss + a.upgo * upgo_loss
            stats["upgo_loss"] += upgo_loss.detach() / steps
        kl, entropy, bonus = torch.zeros_like(log_prob), torch.zeros_like(log_prob), torch.zeros_like(log_prob)
        type_bonus = torch.zeros((), device=log_prob.device)
        for name, (lp, _, use) in per.items():
            p, ref = lp.exp(), reference_lp[name]
            # KL(clone || learner), weighted by the clone's probabilities (AlphaStar, DI-star, VPT): it holds on to what
            # the clone does and barely charges for what it rarely does. The other way round (learner-weighted, as it
            # was) charged most for moving probability onto the clone's rare choices, such as orders into the enemy's
            # half (the clone 0.5%), which attacking needs
            head_kl, head_entropy = (ref.exp() * (ref - lp)).sum(-1), -(p * lp).sum(-1)
            kl = kl + torch.where(use, head_kl, torch.zeros_like(head_kl))
            entropy = entropy + torch.where(use, head_entropy, torch.zeros_like(head_entropy))
            # the bonus as AlphaStar's (its rl.py entropy()): each head's entropy over ln(its options), 0 to 1, so a
            # head of hundreds of options (cell, ~4.7 nats) weighs no more than one of two; the options are the ones
            # the mask leaves (masked logits are -1e9), at least 2
            options = (lp > -1e8).sum(-1).clamp(min=2).float()
            bonus = bonus + torch.where(use, head_entropy / options.log(), torch.zeros_like(head_entropy))
            if name == "type":  # its own bonus too (--type-entropy): whether to give an order at all
                type_bonus = masked_mean(head_entropy / options.log(), use & valid)
            stats[f"entropy_{name}"] += masked_mean(head_entropy, use).detach() / steps
            stats[f"kl_{name}"] += masked_mean(head_kl, use).detach() / steps
        kl, entropy, bonus = masked_mean(kl, valid), masked_mean(entropy, valid), masked_mean(bonus, valid)
        loss = policy_loss + a.kl * kl - a.entropy * bonus - a.type_entropy * type_bonus
        if self.floor is not None:  # each floored type at least its floor where legal: log(floor) - log p, when above 0
            lp, _, use = per["type"]
            under = (self.floor.clamp(min=1e-30).log() - lp).clamp(min=0) * ((self.floor > 0) & (lp > -1e8))
            explore = masked_mean(under.sum(-1), use & valid)
            loss = loss + explore
            stats["explore_loss"] += explore.detach() / steps
        stats["policy_loss"] += policy_loss.detach() / steps
        stats["kl_to_clone"] += kl.detach() / steps
        stats["entropy"] += entropy.detach() / steps
        stats["entropy_normalized"] += bonus.detach() / steps
        stats["clip_fraction"] += masked_mean(((ratio - 1).abs() > a.clip).float(), valid) / steps
        return loss

    # ---- the loop
    def _pack(self, fresh: threading.Event) -> None:
        """On a thread of its own: each new version's weights packed once, uncompressed, as soon as the update
        that made them ends, so a pull only sends bytes (packing on the pull, with zlib, made it ~2 s in sp15-17:
        the games played on weights 2-3 updates older than they had to)."""
        while True:
            fresh.wait()
            fresh.clear()
            with self.lock:
                version, weights = self.version, self.weights
            reply = Packed((version, weights, None), compress=False)  # outside the lock: chunks keep coming
            with self.lock:
                self.reply = (version, reply)

    def _stall_watch(self) -> None:
        """Ends the process when no update has finished for stall_minutes (three times that before this start's first,
        while the actors start): a stuck learner would otherwise keep its GPU, and its actors, until the machine's
        limit."""
        a = self.args
        while True:
            time.sleep(60)
            limit = a.stall_minutes * (3 if self.update == self.first_update else 1) * 60
            if time.time() - self.last_progress > limit:
                self.say(f"no update for {limit / 60:.0f} minutes; stopping")
                self.log.flush()
                os._exit(3)

    def run(self, stop=lambda: False) -> None:
        """Updates until --updates, or until stop() says so (run.py: when every actor has exited)."""
        a = self.args
        server = self.server = Server(address(a.listen), a.authkey.encode(), self.handle)
        self.say(f"listening on {server.address}; update {self.update}, {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        started, steps = time.time(), 0
        started_run = started - self.seconds_done
        self.last_progress, self.first_update = started, self.update
        if a.stall_minutes:
            threading.Thread(target=self._stall_watch, daemon=True).start()
        # the batches come from a process of their own (batches.py): on a thread here they slowed every update ~1.75x
        batches = Batches(chunks_address(a.listen), a.authkey.encode(), a.batch, a.buffer, a.max_lag,
                          [m.width for m in self.model.memory],
                          (self.gammas, a.lam, a.critic_lam) if a.actor_values else None)  # fmt: skip
        batches.version.value = self.version
        fresh = threading.Event()  # new weights to pack
        threading.Thread(target=self._pack, args=(fresh,), daemon=True).start()
        waited, frozen = 0.0, False
        times: dict = collections.defaultdict(float)  # the time_* parts summed since the last log line
        between = time.perf_counter()  # the last update's end: time_update is from one update's end to the next's
        while a.updates <= 0 or self.update < a.updates:
            waiting = time.perf_counter()
            try:
                host = batches.get(timeout=0.5)
                if host is None:
                    raise queue.Empty
            except queue.Empty:
                waited += time.perf_counter() - waiting  # all of it: a wait longer than 0.5 s comes here first
                if stop():
                    self.say("stopping: no actors")
                    break
                continue
            waited += time.perf_counter() - waiting
            stats = self.step(None, host)
            stats["time_wait"], stats["time_host"], waited = waited, batches.seconds, 0.0
            stats["dropped"] = batches.dropped
            # the save's copy on the update that made one (0 on the others), and its last write on the thread
            stats["time_save"], stats["save_write"], self.save_seconds = self.save_seconds, self.write_seconds, 0.0
            if not frozen:
                # everything alive after the first update (the compiled model's many objects) out of the garbage
                # collector's sight: its full passes over them held the GIL ~0.2-0.3 s at a time while chunks
                # arrived, the GPU waiting for kernels meanwhile (training/bench/learner_gap.py)
                gc.freeze()
                frozen = True
            self.update += 1
            self.last_progress = time.time()
            steps += int(host["dense"]["step_valid"].sum())
            with self.lock:
                self.weights, self.version = self._weights(), self.update
            batches.version.value = self.update
            fresh.set()
            now = time.perf_counter()
            times["time_update"] += now - between
            between = now
            for k, v in stats.items():
                if k.startswith("time_"):
                    times[k] += v
            if self.update % a.log_every == 0:
                # the parts as means over the last log_every updates (one update's were a poor sample: waits vary)
                stats.update({k: v / a.log_every for k, v in times.items()})
                stats["time_other"] = stats["time_update"] - sum(stats.get(k, 0.0) for k in ("time_wait", "time_device", "time_score", "time_train"))
                times.clear()
                rate = steps / (time.time() - started)
                results = list(self.recent_results)
                outcomes = {}  # per kind of opponent: the share of wins, losses and time limits
                for opponent, outcome, _, _ in results:
                    key = opponent if opponent in self.league.ladder else "self/pool"
                    outcomes.setdefault(key, []).append(outcome)
                record = " ".join(f"{k} won {v.count('victory') / len(v):.2f} lost {v.count('defeat') / len(v):.2f} "
                                  f"time {v.count('time limit') / len(v):.2f} ({len(v)})" for k, v in sorted(outcomes.items()))  # fmt: skip
                record += f" | limit {self.league.minutes:g} min"
                evals = {p: list(r) for p, r in self.eval_results.items() if r}
                if evals:  # each evaluated policy's last EVAL_WINDOW evaluation games at the fixed stage
                    record += f" | eval {self.eval_handicap:g}: " + " ".join(
                        f"{p} {v.count('victory') / len(v):.2f} ({len(v)})" for p, v in evals.items())
                    stats.update({f"eval_won_{p}": v.count("victory") / len(v) for p, v in evals.items()})
                if results:  # the learner's games: how long they last, and its economy (gold mined per minute)
                    record += f" | minutes {np.mean([r[2] for r in results]):.1f} gold/min {np.mean([r[3] for r in results]):.0f}"
                stats.update({"won_total": self.outcomes.get("victory", 0), "lost_total": self.outcomes.get("defeat", 0),
                              **{f"ladder_{k}_total": self.outcomes.get(f"ladder {o}", 0) for k, o in
                                 (("won", "victory"), ("lost", "defeat"), ("timeout", "time limit"))},
                              "handicap": self.league.handicap})  # fmt: skip
                fields = " ".join(f"{k} {v}" if k.endswith("_total") else f"{k} {v:.3g}" for k, v in sorted(stats.items()))
                self.say(f"update {self.update} {rate:.0f} steps/s games {self.games} ladder {self.league.rung} "
                         f"({self.league.level + 1}/{len(self.league.ladder)}) | {record} | {fields}" + (" (warm-up)" if self.update <= a.warmup else ""))
            if self.update % a.snapshot_every == 0:
                name = f"u{self.update:06d}"
                torch.save(self._weights(), self.out / "pool" / f"{name}.pt")
                with self.lock:
                    self.league.add(name, self.update)
            if a.reset_every and self.update % a.reset_every == 0:
                # the KL's anchor becomes the current policy (ProRL): the policy may go far over a run, never far
                # from where it recently was
                self.reference.load_state_dict(self.model.state_dict())
                self.say(f"update {self.update}: the KL's reference is now this policy")
            if a.hours and time.time() - started_run > a.hours * 3600:
                self.say(f"stopping: {a.hours} hours")
                self._save(started_run, wait=True)
                break
            if self.update % a.save_every == 0:
                self._save(started_run)
        batches.close()
        if self.saving is not None:
            self.saving.join()

    def _save(self, started_run: float, wait: bool = False) -> None:
        """last.pt: a copy on the CPU here, written on a thread (~190 MB to the run's volume took the updates ~0.4 s
        each in sp16, every 10th waiting for it); a save still being written when the next is due is skipped."""
        if self.saving is not None and self.saving.is_alive():
            if not wait:
                return
            self.saving.join()
        began = time.perf_counter()

        def cpu(tree):
            if torch.is_tensor(tree):
                return tree.detach().to("cpu", copy=True)
            if isinstance(tree, dict):
                return {k: cpu(v) for k, v in tree.items()}
            if isinstance(tree, (list, tuple)):
                return type(tree)(cpu(v) for v in tree)
            return tree

        with self.lock:
            league = copy.deepcopy(self.league)
        state = {"model": cpu(self.model.state_dict()), "optimizer": cpu(self.opt.state_dict()),
                 "reference": cpu(self.reference.state_dict()), "update": self.update, "league": league, "critic": CHANNELS,
                 "seconds_done": time.time() - started_run, "tries": self.tries.copy(),
                 "tried_at": self.tried_at, "games": self.games, "eval_handicap": self.eval_handicap}  # fmt: skip
        self.save_seconds = time.perf_counter() - began

        def write():
            began = time.perf_counter()
            torch.save(state, self.out / "last.tmp")
            os.replace(self.out / "last.tmp", self.out / "last.pt")  # never a half-written last.pt
            if self.args.on_save:
                self.args.on_save()
            self.write_seconds = time.perf_counter() - began

        self.saving = threading.Thread(target=write, daemon=not wait)
        self.saving.start()
        if wait:
            self.saving.join()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="the KL's anchor (the clone, or a checkpoint chosen as better), and "
                   "the start without --start")
    p.add_argument("--start", default="", help="the weights to start from, apart from the anchor (another run's "
                   "last.pt or a checkpoint; default --init). Until 2026-10-13 a run from another's last.pt took it as "
                   "its anchor too: cb7's anchor was cb6's end, cb8's cb7's")
    p.add_argument("--carry-adam", action=argparse.BooleanOptionalAction, default=True, help="the start's Adam state "
                   "too, when it is a learner's checkpoint (--no-: a fresh optimizer)")
    p.add_argument("--out", required=True)
    p.add_argument("--listen", default="0.0.0.0:7600")
    p.add_argument("--authkey", default=os.environ.get("WC3_SELFPLAY_KEY", "wc3-selfplay"))
    p.add_argument("--batch", type=int, default=128, help="chunks per update (more: less weight lag per chunk)")
    p.add_argument("--buffer", type=int, default=2048, help="chunks kept waiting at most (older ones drop)")
    p.add_argument("--max-lag", type=int, default=24, help="chunks made with weights more updates old than this are "
                   "dropped, not trained on (sp15 trained on chunks ~21 updates old when the games outran the learner)")
    p.add_argument("--actor-values", action=argparse.BooleanOptionalAction, default=True,
                   help="GAE from the values the actors recorded (no value pass of the learner's own)")
    p.add_argument("--epochs", type=int, default=1, help="passes over each batch: 1 lets the one L4 take in twice the new games (it was the limit at 2)")
    p.add_argument("--minibatch", type=int, default=32, help="chunks per gradient step")
    p.add_argument("--update-pause-ms", type=float, default=0, help="after each gradient step, wait for the GPU and then "
                   "this long, so the inference process's batches are not stuck behind the update (learner and inference "
                   "on one GPU; warcraftsim: 8 ms took its rounds during an update 39 -> 6 ms)")
    p.add_argument("--micro", type=int, default=0, help="chunks through the GPU at once, gradients added up to a "
                   "minibatch (0: the whole minibatch; an L4's 24 GB needs 4: 8 ran out once games grew)")
    p.add_argument("--lr", type=float, default=4e-5, help="cb8-cb9's (2e-5 until then, VPT's)")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.9999, help="per 250 ms step: a win 15 min away still counts 0.70 (0.9997 gave 0.34, and losing at minute 10 cost more than running out a 15-minute clock)")
    p.add_argument("--channel-gamma", default="dealt=0.995,taken=0.995", help="a discount of their own for named "
                   "reward channels (name=gamma,...; the rest --gamma): a fight's trades pay off within seconds, and at "
                   "0.9999 the dealt and taken critics answered for the next ~40 minutes of fights; 0.995 a ~50 s horizon")
    p.add_argument("--lam", type=float, default=1.0, help="the policy's advantages: every real reward to the chunk's end "
                   "counts in full (0.95 until sp40: a reward 5 s away counted 36%%, 15 s away 5%%)")
    p.add_argument("--critic-lam", type=float, default=0.9, help="the critic's targets (AlphaStar's TD lambda 0.8): "
                   "lower leans more on its own next guesses, less on one game's luck")
    p.add_argument("--value-coef", type=float, default=10.0, help="the value loss's weight against the policy's 1 "
                   "(AlphaStar's and DI-star's win/loss baseline: 10; 0.5 until sp37, whose critic fit fell from ~0.4 to "
                   "~0.08 once the policy learned)")
    p.add_argument("--type-entropy", type=float, default=0.0, help="a bonus on the type head's entropy alone (over ln(its "
                   "options), 0 to 1), beside --entropy's on every head: keeps the policy trying orders. cb4 gave none on "
                   "98%% of its steps (its type head 0.11 nats, cb1's start 0.59) and stopped improving")
    p.add_argument("--upgo", type=float, default=0.0, help="UPGO's policy term beside PPO's, at this weight (AlphaStar; "
                   "DI-star 1.0, as its main term): a step's return follows what came after only while each next step "
                   "did at least as well as the critic expected, so a good action is not charged for a later mistake")
    p.add_argument("--explore-floor", type=float, default=0.0, help="each order type but noop at least this chance "
                   "wherever legal, until it has been chosen --explore-tries times at the stage (each promotion re-arms it), then "
                   "--explore-min (0: off). Rare "
                   "but valuable orders get tried: cb5 cast a targeted spell on 0.04%% of steps, about once in 7 "
                   "fights, too few for PPO to learn whether it pays. 0.005: ~1.8 tries per 1.5-minute fight")
    p.add_argument("--explore-tries", type=int, default=50_000, help="--explore-floor's tries of a type before its "
                   "floor comes off at a stage (~45 minutes at 0.5%% of ~3,600 steps/s)")
    p.add_argument("--explore-min", type=float, default=0.0, help="each order type but noop at least this chance "
                   "wherever legal once its --explore-tries are done, for good (0: off): none dies out. cb6's hold and "
                   "stop fell to 0.3%% of steps once its floor came off")
    p.add_argument("--popart-rate", type=float, default=3e-3, help="how fast the value statistics follow the returns")
    p.add_argument("--entropy", type=float, default=1e-4, help="the entropy bonus, as AlphaStar's: each head's "
                   "entropy over ln(its options) (0 to 1), summed over the heads a step uses, at AlphaStar's weight. Was "
                   "0.003 on raw nats (~100x AlphaStar's push): sp35's policy grew random, then fell into a losing habit")
    p.add_argument("--kl", type=float, default=0.005, help="the KL from the reference policy (0.02 until 2026-10-12: cb1's "
                   "policy stayed ~0.065 from the clone and stopped improving; DI-star 0.002; VPT starts at "
                   "0.2 and decays it 0.5%% an update)")
    p.add_argument("--reset-every", type=int, default=0, help="updates between moving the KL's reference to the "
                   "current policy (0: always the clone, as AlphaStar and VPT; sp3-sp4's resets let it drop attacking)")
    p.add_argument("--dual-clip", type=float, default=3.0, help="a negative advantage counts at most this many times")
    p.add_argument("--hours", type=float, default=0, help="stop after this long (0: never)")
    p.add_argument("--start-stage", type=float, default=0, help="the rung's stage to start on, one of the mode's "
                   "handicaps (default: its first)")
    p.add_argument("--hold-stage", action="store_true", help="never move up the ladder: a fixed opponent (with "
                   "--warmup past the run's end, the policy measured as it is)")
    p.add_argument("--start-rung", default="", help="the ladder's rung to start on, one of the mode's (default: its "
                   "first, at its first stage; whole games: ai0, ai1, ai2)")
    p.add_argument("--stall-minutes", type=float, default=20, help="stop when no update finished for this long, "
                   "three times as long before the first (0: never)")
    p.add_argument("--minutes", type=float, default=START_MINUTES, help="the games' first time limit (the league extends it)")
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True,
                   help="rescale each batch's advantages to mean 0, spread 1 (the PPO libraries' default; --no-: raw, in "
                   "reward units, as AlphaStar and DI-star: sp40's spread was 0.01-0.09, its policy loss then small "
                   "against everything else)")
    p.add_argument("--mode", choices=modes.NAMES, default="game", help="what the actors play (training/selfplay/modes): "
                   "its shaping's weight")
    p.add_argument("--shaping-scale", type=float, default=1.0, help="every reward channel but the win's, times this "
                   "(0.2: the win counts five times as much against the shaping as by default)")
    p.add_argument("--warmup", type=int, default=50, help="updates training only the critic")
    p.add_argument("--snapshot-every", type=int, default=200)
    p.add_argument("--evaluate", default="", help="checkpoints the evaluation games (actors' --eval-games) play beside "
                   "the current policy, comma-separated: each named eval-<run> (runs/cb8/last.pt: eval-cb8)")
    p.add_argument("--eval-stage", type=float, default=0, help="the ladder stage evaluation games are played at, "
                   "fixed for the run (0: the stage it starts on)")
    p.add_argument("--save-every", type=int, default=10, help="updates between saves (a few minutes: what a "
                   "preempted learner loses)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--updates", type=int, default=0, help="stop after this many (0: never)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the model's blocks on a GPU (--no-compile: eager)")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.on_save = None
    Learner(arguments).run()
