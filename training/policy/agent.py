"""Policies playing: many seats (a player in a game) decided in one batch per step.

    agents = Agents(model, temperature=1.0)
    seats = [Seat(static, plan) for each player the policy controls]
    decisions = agents.act(seats, observations, blocks)   # one Decision per seat: action, commands, log-probability

`blocks` are each observation's features and masks (policy/native.py): obs.features when the game hook computed
them (GameConfig.features; send each decision back with hook_decision), or native.Features for recorded
observations. The model keeps each seat's memory (SSM state, the attention window of the current 64-step chunk:
chunks start every CHUNK steps from the game's start, as in the data set). A step: one forward pass and the heads
in order for all seats on the device (decide.py), each head's choices limited by the block's masks and the rules of
head_rules(). A seat whose action cannot be finished (no legal actor, say) does nothing that step. Decisions keep
what self-play needs to learn from them: the features, the choices as data set indices, the masks and the
log-probability.
"""

from __future__ import annotations

import collections
import itertools
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .actions import SIZES, TYPES, Action, translate
from .decide import HEADS, NO_KEY, UNUSED, actor_keys, pack, sample, unpack
from .features import MAP, Static, static_map
from .masks import argument_ids
from .model import PATCH, ROW_SIZES, Policy, float32
from .native import chosen_masks, tables
from .plan import empty as empty_plan

CHUNK = 64
_keys = itertools.count(1)
# what the features are sent to the model as: float16 as the learner stores them (data set and chunks), ids in 16
# bits (every id is below 32,768); a third of float32's bytes
SENT = {"units": np.float16, "scalars": np.float16, **{k: np.int16 for k in ("unit_type", "unit_owner", "unit_order",
        "unit_queue", "unit_inventory", "unit_buffs", "unit_abilities", "unit_cell")}}  # fmt: skip
# a block's per-row arrays the model reads (unit_id stays with the actor: choices are rows, turned into ids here)
UNIT_KEYS = ("units", "unit_type", "unit_owner", "unit_order", "unit_queue", "unit_inventory", "unit_buffs",
             "unit_abilities", "unit_cell")  # fmt: skip


@dataclass
class Seat:
    static: Static
    plan: dict | None = None  # training/policy/plan.py arrays, or None for the empty plan
    t: int = 0
    last: Action | None = None  # the action decided on the last observation (None: a noop)
    sent_trees: np.ndarray | None = None  # the tree grid as the model last got it
    # names the seat's model memory where the model runs (unique across the actor processes one model serves)
    key: tuple = field(default_factory=lambda: (os.getpid(), next(_keys)))


@dataclass
class Decision:
    action: Action | None  # None: noop
    commands: list[dict]
    features: dict
    choices: dict  # head -> index as the data set stores it (type, argument index, actor row, ...)
    masks: dict  # type, argument, actor, target: the masks the choices were sampled under
    log_prob: float  # of the whole decision (noop included), at temperature 1 for self-play
    failed: str = ""  # "<type>/<head>" when an action was chosen but that head had no legal choice: it became a noop
    value: np.ndarray | float = float("nan")  # the critic's estimates [CRITIC], when act() was given the opponents' scalars
    chunk_start: bool = False  # the first step of one of the seat's 64-step chunks
    start_state: list | None = None  # then: the model's memory before it (numpy, one array per SSM layer; None at
    # the game's start)


class Agents:
    """Decisions for many seats: the CPU's part here (features, mask tables, the decisions from the choices); the
    model's part in a Model, in this process or another (training/selfplay/inference.py: one process on the GPU
    deciding for every actor)."""

    def __init__(self, model: Policy | None = None, temperature: float = 1.0, seed: int = 0, remote=None,
                 compile: bool = False):
        """`model` to run it here; or `remote`, where it runs: send(inputs) passes a step's inputs on, receive()
        returns the outputs of the oldest step sent (training/selfplay/actor.py: the inference process)."""
        self.model = Model(model, temperature, seed, compile=compile) if model is not None else None
        self.backend = remote if model is None else _Here(self.model)
        self.args = {i: v for v, i in argument_ids().items()}
        self.sent: set = set()  # the map data the model has (by _map_key)

    def act(self, seats: list[Seat], observations: list, blocks: list, opponent_scalars: list | None = None) -> list[Decision]:
        """One step for every seat; with `opponent_scalars` (each seat's opponent's block["scalars"]), also the
        critic's estimate of each seat's position."""
        return self.end(self.begin(seats, observations, blocks, opponent_scalars))

    def begin(self, seats: list[Seat], observations: list, blocks: list, opponent_scalars: list | None = None) -> tuple:
        """act()'s first half: the inputs sent to the model. With the model elsewhere, the caller can do other work
        (another set of games' begin()) before end()."""
        self.backend.send(self._inputs(seats, blocks, opponent_scalars))
        return seats, observations, blocks

    def end(self, begun: tuple) -> list[Decision]:
        """act()'s second half: the model's outputs for the oldest begin() not yet ended, as decisions."""
        seats, observations, blocks = begun
        decisions = self._finish(seats, observations, blocks, self.backend.receive())
        for seat, d in zip(seats, decisions):
            seat.last = d.action
            seat.t += 1
        return decisions

    def _inputs(self, seats, blocks, opponent_scalars) -> dict:
        """What the model needs, as numpy arrays: the blocks' unit rows padded to the fullest seat's count (the
        model pads them on to its row sizes), each seat's key, step and map, the plans, the mask tables, the
        opponents."""
        rows = max(1, max(int(b["unit_count"]) for b in blocks))
        # the tree grids that changed since this seat's last step (the model keeps each seat's last), and every grid at
        # a chunk's start: a snapshot's model made again in the inference process has them within 64 steps
        changed = {}
        for i, (s, b) in enumerate(zip(seats, blocks)):
            if s.sent_trees is None or s.t % CHUNK == 0 or not np.array_equal(s.sent_trees, b["trees"]):
                changed[i] = s.sent_trees = b["trees"]
        features = {"unit_count": np.array([b["unit_count"] for b in blocks], np.int32)}
        for k in UNIT_KEYS:
            first = blocks[0][k]
            out = np.zeros((len(blocks), rows, *first.shape[1:]), SENT.get(k, first.dtype))
            for i, b in enumerate(blocks):
                out[i, : len(b[k])] = b[k]
            features[k] = out
        for k in ("scalars", "previous_type", "recent"):
            features[k] = np.stack([b[k] for b in blocks]).astype(SENT.get(k, blocks[0][k].dtype))
        x = {
            "features": features,
            "trees": changed,
            "keys": [s.key for s in seats],
            "t": [s.t for s in seats],
            "static": [_map_key(s.static) for s in seats],
            "statics": {},
            "tables": [tables(b) for b in blocks],
            "opponents": None if opponent_scalars is None else np.stack(opponent_scalars).astype(np.float16),
            "plans": None,
        }
        for s in seats:
            key = _map_key(s.static)
            if key not in self.sent:
                self.sent.add(key)
                x["statics"][key] = static_map(s.static)
        if any(s.plan is not None for s in seats):
            filled = [s.plan if s.plan is not None else empty_plan() for s in seats]
            x["plans"] = {k: np.stack([np.asarray(p[k]) for p in filled]) for k in filled[0]}
        return x

    def _finish(self, seats, observations, blocks, c) -> list[Decision]:
        """Decisions from the model's choices. A decision keeps the masks of what was chosen, as the learner needs
        them."""
        decisions = []
        for i, (seat, obs, f) in enumerate(zip(seats, observations, blocks)):
            kind, log_prob = int(c["type"][i]), float(c["log_prob"][i])
            failed = f"{TYPES[kind]}/{HEADS[c['failed'][i]]}" if c["failed"][i] >= 0 else ""
            if failed or kind == 0:  # a failed action is recorded as the noop it became, with a noop's masks
                choices = {**UNUSED, "type": 0}
                d = Decision(None, [], f, choices, chosen_masks(f, 0, 0), log_prob, failed)
            else:
                choices = {h: int(c[h][i]) for h in HEADS}
                action = self._action(seat, f, choices)
                masks = chosen_masks(f, kind, choices["argument"])
                d = Decision(action, translate(action, obs), f, choices, masks, log_prob)
            if c.get("value") is not None:
                d.value = c["value"][i]
            d.chunk_start = seat.t % CHUNK == 0
            if d.chunk_start:
                d.start_state = c["start_state"][i]
            decisions.append(d)
        return decisions

    def _action(self, seat: Seat, f: dict, c: dict) -> Action:
        rows = f["unit_id"]
        kind = TYPES[c["type"]]
        fields = {"argument": self._raw(c["argument"])}
        if c["actor"] >= 0:
            fields["actor"] = int(rows[c["actor"]])
            fields["same_type"] = bool(c["same_type"])
            fields["size"] = SIZES[c["size"]]
            fields["queued"] = bool(c["queued"])
        if c["target"] >= 0:
            fields["target"] = int(rows[c["target"]])
        if c["region"] >= 0:  # the middle of the chosen cell
            py = (c["region"] // (MAP // PATCH)) * PATCH + c["cell"] // PATCH
            px = (c["region"] % (MAP // PATCH)) * PATCH + c["cell"] % PATCH
            min_x, min_y, max_x, max_y = seat.static.bounds
            fields["x"] = min_x + (px + 0.5) / MAP * (max_x - min_x)
            fields["y"] = min_y + (py + 0.5) / MAP * (max_y - min_y)
        return Action(kind, **fields)

    def _raw(self, index: int) -> int:
        return self.args.get(index, 0) if index > 0 else 0


class _Here:
    """The model in this process, as a backend: send() runs it, receive() hands back the results in order."""

    def __init__(self, model):
        self.model, self.done = model, collections.deque()

    def send(self, inputs: dict) -> None:
        self.done.append(self.model.run(inputs))

    def receive(self) -> dict:
        return self.done.popleft()


def _map_key(static: Static) -> tuple:
    return (tuple(static.races), tuple(static.bounds))


# a batch's seats and unit rows, padded: few sizes, since each is a graph to make (0.6-3 s; compiled, 20-40 s) and
# padding is cheap (58 seats 4.6 ms, 8 seats 2.7: training/bench/profile_run.py). sp21's batches: ~53 seats, 98% at
# 96 unit rows.
SEAT_SIZES = (64, 128, 256, 512, 1024)
UNIT_ROWS = tuple(r for r in ROW_SIZES if r >= 96)
KEYS_A_SEAT = 64  # actor masks a seat has room for at least (more: the next power of two)
TABLES = ("type", "argument", "actor_keys", "actor_rows")


class Model:
    """The model's part of a step for many seats, on its device: the forward pass from each seat's memory, every
    head (decide.sample) and the critic. Each seat's memory (SSM state, the current chunk's attention window) is a
    row of tables on the device, found by the seat's key, so a batch's memories are gathered and stored with one
    indexing operation each. The time attention has no positions, so a window's steps sit at the front of its row
    with the rest masked. A seat not heard from for FORGET seconds (its game over) gives its row back.

    On a GPU the step is a CUDA graph: launched one kernel at a time it was ~1,070 launches a batch, ~15 of its ~20
    ms the CPU's (training/bench/profile_run.py). A graph replays fixed shapes, so a batch is padded to the seats of
    SEAT_SIZES (the padding seats read and write row 0, kept for them), its unit rows to ROW_SIZES and its actor
    masks to KEYS_A_SEAT a seat (or the next power of two); the window is always CHUNK steps, masked. One graph per such shape, made the
    first time it comes up. The weights are read where they are: loading new ones (in place) needs no new graph.

    `compile` (a GPU): the step's parts fused by torch.compile inside the graphs (the transformer blocks and SSM
    layers, encode_step, the heads): 58 seats at 96 unit rows 3.9 -> 1.8 ms of GPU (training/bench/step_parts.py).
    Compiling a shape takes 20-40 s, so every common shape is made at the first batch, none while games wait."""

    FORGET = 600.0

    def __init__(self, model: Policy, temperature: float = 1.0, seed: int = 0, compile: bool = False):
        self.model, self.temperature = model, temperature
        self.device = next(model.parameters()).device
        self.cuda = self.device.type == "cuda"
        if self.cuda:  # a graph replays the device's default generator; on the CPU, one of this model's
            torch.cuda.manual_seed(seed)
            self.rng = None
        else:
            self.rng = torch.Generator().manual_seed(seed)
        self.compiled = compile and self.cuda
        self.sample = sample
        if self.compiled:
            from torch import _dynamo

            model.compile_blocks()
            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            model.encode_step = torch.compile(model.encode_step, dynamic=False)
            self.sample = torch.compile(sample, dynamic=False)
        self.static_index: dict[tuple, int] = {}  # map data by _map_key -> its row of static_table
        self.static_table = None
        self.slot: dict[tuple, int] = {}  # seat key -> its row
        self.seen: dict[tuple, float] = {}
        self.free: list[int] = []
        self.graphs: dict[tuple, _Step] = {}
        # ms since the last stats() per part of run(): preparing the inputs, the step on the device (and the copy
        # back), the chunk starts' memory; the unit rows of the batches; graphs made
        self.timing: dict[str, list] = {"prep": [], "step": [], "starts": [], "rows": [], "captures": []}
        # rows for every seat seen in the last FORGET seconds: games that end quickly (the ladder's early stages) bring
        # new seats often, and growing the tables means making every graph again
        self._grow(4096)
        self.swept = time.time()

    def _grow(self, rows: int) -> None:
        """Tables of `rows` rows (the old rows kept): state [layers][rows, width], has_state [rows], window
        [rows, CHUNK, d] and its length [rows], and the seat's tree grid [rows, MAP, MAP] (sent only when it
        changed). Row 0 is the padding seats'. New tables: the graphs that read the old ones go."""
        dev, model = self.device, self.model
        old = getattr(self, "rows", 0)
        width, layers = model.memory[0].width, len(model.memory)
        state = [torch.zeros(rows, width, device=dev) for _ in range(layers)]
        has_state = torch.zeros(rows, dtype=torch.bool, device=dev)
        window = torch.zeros(rows, CHUNK, model.d, device=dev)
        length = torch.zeros(rows, dtype=torch.long, device=dev)
        trees = torch.zeros(rows, MAP, MAP, dtype=torch.uint8, device=dev)
        if old:
            for k in range(layers):
                state[k][:old] = self.state[k]
            has_state[:old], window[:old], length[:old] = self.has_state, self.window, self.length
            trees[:old] = self.trees
        self.state, self.has_state, self.window, self.length, self.trees = state, has_state, window, length, trees
        self.free += list(range(rows - 1, max(old, 1) - 1, -1))
        self.rows = rows
        self.graphs.clear()

    def _maps(self, statics: dict, keys: list) -> list[int]:
        """Each seat's row of the map data table (new maps added; a bigger table: the graphs go)."""
        for key, grid in statics.items():
            if key not in self.static_index:
                if self.static_table is None or len(self.static_index) == len(self.static_table):
                    size = 32 if self.static_table is None else 2 * len(self.static_table)
                    table = torch.zeros(size, *grid.shape, dtype=torch.from_numpy(grid).dtype, device=self.device)
                    if self.static_table is not None:
                        table[: len(self.static_table)] = self.static_table
                    self.static_table = table
                    self.graphs.clear()
                self.static_index[key] = len(self.static_index)
            self.static_table[self.static_index[key]] = torch.from_numpy(grid).to(self.device)
        return [self.static_index[k] for k in keys]

    def _slots(self, keys, now) -> list[int]:
        out = []
        for key in keys:
            row = self.slot.get(key)
            if row is None:
                if not self.free:
                    self._grow(2 * self.rows)
                row = self.slot[key] = self.free.pop()
                self.has_state[row], self.length[row] = False, 0
            self.seen[key] = now
            out.append(row)
        return out

    @torch.no_grad()
    def run(self, x: dict) -> dict:
        """Agents._inputs -> {head: [N] choices, log_prob, failed, value (or None), start_state: per seat, the
        memory before this step where the step starts a chunk}, as numpy."""
        began = time.perf_counter()
        now, n = time.time(), len(x["keys"])
        maps = self._maps(x["statics"], x["static"])
        rows = self._slots(x["keys"], now)
        if x["trees"]:  # the seats whose tree grids changed: into their rows
            changed = list(x["trees"])
            grids = torch.from_numpy(np.stack([x["trees"][i] for i in changed])).to(self.device)
            self.trees[torch.tensor([rows[i] for i in changed], device=self.device)] = grids
        seats = next((s for s in SEAT_SIZES if s >= n), -(-n // 256) * 256)
        unit_rows = next(r for r in UNIT_ROWS if r >= max(1, x["features"]["units"].shape[1]))
        per_seat = max(KEYS_A_SEAT, 1 << (-(-actor_keys(x["tables"]) // seats) - 1).bit_length())
        if self.compiled and not self.graphs:  # every common shape now, from this batch's inputs' layout
            for s_, r_ in itertools.product(SEAT_SIZES[:4], UNIT_ROWS):
                capturing = time.perf_counter()
                self.graphs[(s_, r_, KEYS_A_SEAT)] = _Step(self, x, s_, r_, KEYS_A_SEAT)
                self.timing["captures"].append(1000 * (time.perf_counter() - capturing))
            began = time.perf_counter()
        step = self.graphs.get((seats, unit_rows, per_seat))
        if step is None:
            capturing = time.perf_counter()
            step = self.graphs[(seats, unit_rows, per_seat)] = _Step(self, x, seats, unit_rows, per_seat)
            self.timing["captures"].append(1000 * (time.perf_counter() - capturing))
            began = time.perf_counter()
        starts = np.asarray(x["t"]) % CHUNK == 0
        prepared = time.perf_counter()
        out = step(x, rows, maps, starts)
        ints, floats = out["ints"][:, :n].cpu().numpy(), out["floats"][:, :n].cpu().numpy()
        stepped = time.perf_counter()
        c = {h: ints[k] for k, h in enumerate(HEADS)}
        c["failed"], c["log_prob"] = ints[len(HEADS)], floats[0]
        c["value"] = floats[1:].T if x["opponents"] is not None else None  # [n, CRITIC]
        start_state = [None] * n
        first = np.nonzero(starts)[0]
        if len(first):  # the memory each chunk starts from, for the learner
            at = torch.from_numpy(first).to(self.device)
            kept = out["kept"][at].cpu().numpy()
            saved = [h[at].cpu().numpy() for h in out["before"]]
            for j, i in enumerate(first):
                if kept[j]:
                    start_state[i] = [h[j] for h in saved]
        c["start_state"] = start_state
        done = time.perf_counter()
        for k, v in (("prep", prepared - began), ("step", stepped - prepared), ("starts", done - stepped)):
            self.timing[k].append(1000 * v)
        self.timing["rows"].append(unit_rows)
        if now - self.swept > 60:
            self.swept = now
            for key in [k for k, s in self.seen.items() if now - s > self.FORGET]:
                self.free.append(self.slot.pop(key))
                del self.seen[key]
        return c

    def stats(self) -> str:
        """The timings since the last call, as a log line's part (and cleared)."""
        t, self.timing = self.timing, {k: [] for k in self.timing}
        if not t["step"]:
            return "no batches"
        mean = {k: float(np.mean(t[k])) for k in ("prep", "step", "starts")}
        rows = {r: t["rows"].count(r) for r in sorted(set(t["rows"]))}
        return (f"prep {mean['prep']:.1f}, step {mean['step']:.1f} (p90 {np.percentile(t['step'], 90):.1f}), starts "
                f"{mean['starts']:.1f}; rows {rows}; {len(t['captures'])} graphs made {sum(t['captures']):.0f} ms")

    def _step(self, b: dict, unit_rows: int, features: list[str]) -> dict:
        """One step for the seats in the device buffers `b`, all on the device in fixed shapes (no copy to the
        CPU, no shape that depends on the data): what a graph replays."""
        model, dev = self.model, self.device
        rows = b["rows"]
        length = torch.where(b["starts"], torch.zeros_like(rows), self.length[rows])  # a chunk starts: a new window
        kept = self.has_state[rows]
        before = [h[rows] * kept[:, None] for h in self.state]  # zeros for a seat without memory yet
        window_mask = torch.arange(CHUNK, device=dev)[None] < length[:, None]
        batch = {k: b[k][:, None] for k in features}
        batch["static_map"] = self.static_table[b["maps"]]
        batch["trees"] = self.trees[rows][:, None]
        for k in empty_plan():
            batch[f"plan_{k}"] = b[f"plan_{k}"]
        # the trunk in bfloat16 and the heads in float32, as the learner computes them
        with torch.autocast(dev.type, torch.bfloat16, enabled=self.cuda):
            out = model(batch, before, self.window[rows], window_mask, rows=unit_rows)
        out = float32(out)
        for k, h in enumerate(out["state"]):
            self.state[k][rows] = h.float()
        self.has_state.index_fill_(0, rows, True)  # (an index_put of a Python value copies from the host)
        self.window[rows, length] = out["window"][:, -1].float()
        self.length[rows] = length + 1
        tables = unpack({k: b[f"table_{k}"] for k in TABLES})
        c = self.sample(model, out, tables, batch["unit_count"][:, 0], self.temperature, self.rng)
        value = model.value(out, b["opponents"].float()[:, None])[:, 0].float()  # [seats, CRITIC]
        return {"ints": torch.stack([c[h] for h in HEADS] + [c["failed"]]),
                "floats": torch.cat([c["log_prob"][None], value.T]), "before": before, "kept": kept}  # fmt: skip


class _Step:
    """Model._step at one shape (seats, unit rows, actor masks a seat): its inputs' buffers on the device, their
    pinned copies on the host, and on a GPU the step captured as a CUDA graph."""

    def __init__(self, m: Model, x: dict, seats: int, unit_rows: int, per_seat: int):
        self.m, self.seats, self.unit_rows = m, seats, unit_rows
        cut = set(UNIT_KEYS)
        shapes = {}
        for k, v in x["features"].items():  # [seats, (unit rows,) ...] in the dtypes they are sent in
            shapes[k] = ((seats, unit_rows, *v.shape[2:]) if k in cut else (seats, *v.shape[1:]), v.dtype)
        self.features, self.plans = list(x["features"]), list(empty_plan())
        for k, v in empty_plan().items():
            shapes[f"plan_{k}"] = ((seats, *np.shape(v)), np.asarray(v).dtype)
        shapes["opponents"] = ((seats, x["features"]["scalars"].shape[-1]), np.float16)
        tables = pack([], seats, seats * per_seat)
        for k in TABLES:
            shapes[f"table_{k}"] = (tables[k].shape, tables[k].dtype)
        shapes["rows"], shapes["maps"], shapes["starts"] = ((seats,), np.int64), ((seats,), np.int64), ((seats,), bool)
        self.host = {k: torch.from_numpy(np.zeros(s, d)) for k, (s, d) in shapes.items()}
        if m.cuda:
            self.host = {k: v.pin_memory() for k, v in self.host.items()}
        self.host["table_actor_keys"].fill_(int(NO_KEY))
        self.host["starts"].fill_(True)  # a padding seat starts a new window every step (it never fills)
        self.dev = {k: v.to(m.device) for k, v in self.host.items()}
        self.graph = None
        if m.cuda:  # warmed up and captured with every seat a padding seat (row 0): no real seat's memory moves
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    m._step(self.dev, unit_rows, self.features)
            torch.cuda.current_stream().wait_stream(stream)
            # each graph in a memory pool of its own: capturing into a pool another graph's outputs still use failed
            # PyTorch's allocator mid-run (sp29); a capture that fails anyway leaves this shape to run kernel by kernel
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    self.out = m._step(self.dev, unit_rows, self.features)
                self.graph = graph
            except RuntimeError as error:
                print(f"inference: no CUDA graph for {seats} seats x {unit_rows} unit rows ({error}); kernel by kernel",
                      flush=True)  # fmt: skip
                torch.cuda.synchronize()

    def __call__(self, x: dict, rows: list[int], maps: list[int], starts: np.ndarray) -> dict:
        """The step for x's seats: their inputs into the buffers (the rest padding seats), then the graph."""
        n, h, cut = len(rows), self.host, set(UNIT_KEYS)
        for k in self.features:
            v, dst = x["features"][k], h[k].numpy()
            dst[n:] = 0
            if k in cut:
                dst[:n, v.shape[1] :] = 0
                dst[:n, : v.shape[1]] = v
            else:
                dst[:n] = v
        for k in self.plans:
            dst = h[f"plan_{k}"].numpy()
            dst[:] = 0  # the empty plan is zeros
            if x["plans"] is not None:
                dst[:n] = x["plans"][k]
        opponents = h["opponents"].numpy()
        opponents[:] = 0
        if x["opponents"] is not None:
            opponents[:n] = x["opponents"]
        for k, v in pack(x["tables"], self.seats, len(h["table_actor_keys"])).items():
            h[f"table_{k}"].numpy()[:] = v
        for k, v, pad in (("rows", rows, 0), ("maps", maps, 0), ("starts", starts, True)):
            dst = h[k].numpy()
            dst[:] = pad
            dst[:n] = v
        for k, v in self.dev.items():
            v.copy_(h[k], non_blocking=True)
        if self.graph is None:
            return self.m._step(self.dev, self.unit_rows, self.features)
        self.graph.replay()
        return self.out
