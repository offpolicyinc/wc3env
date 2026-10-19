"""An inference process: the model's part of the decisions of the actors connected to it, on the GPU
(docs/training/10-actor-design.md; self-play runs two, each serving half the actors).

    python -m training.selfplay.inference --learner 127.0.0.1:7600 --listen 127.0.0.1:7700
    python -m training.selfplay.actor --learner 127.0.0.1:7600 --inference 127.0.0.1:7700 ...

Actors keep the CPU's part of a decision (mask tables, the commands from the choices: policy/agent.py) and send
the model's part here: each step's inputs for one policy ("latest", or a snapshot's name). Requests that arrive
within --gather seconds (up to --batch games) run as one batch per policy (policy/agent.py's Model, which keeps
every seat's memory). "latest" follows the learner's weights every --refresh seconds; snapshots are fetched from
the learner on first use.

One Model, compiled, serves every policy: a snapshot's part of a batch runs with its weights copied into the
model's in place (its CUDA graphs read them where they are), then the latest's copied back. Each snapshot was its own
Model until 2026-10-12, uncompiled with graphs of its own: cb1's batches went from 6 ms to 30-47 as snapshots joined
(the latest's own step stayed 6-7 ms), and the games' steps a second fell with them. Seats' keys are unique to their
process and game, so the policies share the memory tables.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import OrderedDict
from multiprocessing.connection import wait

import numpy as np
import torch

from ..policy.agent import UNIT_KEYS, Model
from ..policy.model import load
from ..policy.plan import empty as empty_plan
from .transport import Client, Connections, address


class Inference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = Client(address(args.learner), args.authkey.encode(), wait=1800)
        self.version, state, _ = self.client.call("weights", -1)
        # the policy that learns, compiled; the snapshots run in it with their weights swapped in
        self.models = {"latest": Model(load({"model": state}, self.device), 1.0, args.seed, compile=args.compile)}
        self.live = list(self.models["latest"].model.state_dict().values())  # the model's own tensors (in place)
        self.names = list(self.models["latest"].model.state_dict())
        self.latest = [t.clone() for t in self.live]  # the latest weights, put back after a snapshot's turn
        self.snapshots: OrderedDict[str, list] = OrderedDict()  # policy -> its weights on the device, as `live`
        self.passes: dict[str, list] = {"latest": [], "snapshot": []}  # ms per policy's part of a batch
        self.fresh: tuple | None = None  # (version, weights) pulled and not yet loaded
        self.batches = self.decisions = 0
        self.busy = 0.0  # seconds spent running batches (the rest: waiting for requests)
        # every map's data any request brought: a snapshot's model made again after leaving the cache gets them (each
        # actor sends a map's data once per policy)
        self.maps: dict[tuple, np.ndarray] = {}
        # since the last log line, ms: each batch's time, the wait from a batch's first request to its start and from
        # each request's reading to its batch's start; the weights' pulls and loads
        self.logged, self.work, self.gathered, self.queued = time.time(), [], [], []
        self.pulls, self.loads = [], []
        threading.Thread(target=self._pull, daemon=True).start()
        self.links = Connections(address(args.listen), args.authkey.encode(), compress=False)
        self.address = self.links.address

    def run(self) -> None:
        """One thread: wait for requests on every actor's connection, gather a batch, run it per policy, answer each
        connection in the order it asked (an actor may have two requests in flight). No thread per connection: they
        took turns with this one for Python's lock."""
        links = self.links
        print(f"inference listening on {links.address}, version {self.version}", flush=True)
        started, connections = time.time(), []
        while True:
            connections += links.take()
            # requests that come in until the batch has --batch games or --gather seconds went by: a batch costs
            # about the same whatever its size, so a few ms more waiting buys a much bigger one
            pending = self._read(links, connections, timeout=1.0)
            arrived = [time.perf_counter()] * len(pending)
            deadline = time.perf_counter() + self.args.gather
            while (
                pending and sum(len(r[2]["keys"]) for r in pending) < self.args.batch and time.perf_counter() < deadline
            ):
                more = self._read(links, connections, timeout=max(0.0, deadline - time.perf_counter()))
                pending += more
                arrived += [time.perf_counter()] * len(more)
            if not pending:
                continue
            began = time.perf_counter()
            self.gathered.append(1000 * (began - arrived[0]))
            self.queued += [1000 * (began - a) for a in arrived]
            for _, _, inputs in pending:
                self.maps.update(inputs["statics"])
            replies: dict[int, dict] = {}
            model, swapped = self.models["latest"], False
            for policy in sorted(dict.fromkeys(r[1] for r in pending), key=lambda p: p != "latest"):  # latest first
                part = time.perf_counter()
                if policy != "latest":
                    torch._foreach_copy_(self.live, self._snapshot(policy))
                    swapped = True
                group = [(k, r) for k, r in enumerate(pending) if r[1] == policy]
                x = merge([r[2] for _, r in group])
                missing = set(x["static"]) - model.static_index.keys() - x["statics"].keys()
                if missing:  # map data another policy's requests brought first
                    x["statics"] = {**x["statics"], **{k: self.maps[k] for k in missing}}
                out = model.run(x)
                for (k, _), reply in zip(group, split(out, [len(r[2]["keys"]) for _, r in group])):
                    replies[k] = {**reply, "version": self.version}
                self.batches += 1
                self.passes["latest" if policy == "latest" else "snapshot"].append(1000 * (time.perf_counter() - part))
            if swapped:
                torch._foreach_copy_(self.live, self.latest)
            for k, (connection, _, inputs) in enumerate(pending):
                try:
                    links.answer(connection, replies[k])
                except OSError:  # the actor went away
                    pass
                self.decisions += len(inputs["keys"])
            self.busy += time.perf_counter() - began
            self.work.append(1000 * (time.perf_counter() - began))
            if self.fresh is not None:
                (version, state), self.fresh = self.fresh, None
                loading = time.perf_counter()
                self.models["latest"].model.load_state_dict(state, strict=False)
                torch._foreach_copy_(self.latest, self.live)
                self.loads.append(1000 * (time.perf_counter() - loading))
                self.version = version
            if time.time() - self.logged > self.args.log_every:
                self._log(started)

    @staticmethod
    def _read(links, connections: list, timeout: float) -> list:
        """(connection, policy, inputs) of every request waiting on each connection (after up to `timeout` seconds
        for the first), in the order each connection sent them: an actor may have several in flight, and a request
        left unread waits a whole batch; connections that closed are dropped."""
        out = []
        if not connections:  # nothing to wait on (wait() would return at once, and the loop spin)
            time.sleep(min(timeout, 0.1))
            return out
        for connection in wait(connections, timeout):
            try:
                while True:
                    _, (policy, inputs) = links.read(connection)
                    out.append((connection, policy, inputs))
                    if not connection.poll(0):
                        break
            except (EOFError, OSError):
                connections.remove(connection)
        return out

    def _snapshot(self, policy: str) -> list:
        """A snapshot's weights on the device, in the order of the model's own (the least recently used dropped past
        --snapshot-cache)."""
        if policy not in self.snapshots:
            if len(self.snapshots) >= self.args.snapshot_cache:
                self.snapshots.popitem(last=False)
            weights = load({"model": self.client.call("snapshot", policy)}, torch.device("cpu")).state_dict()
            self.snapshots[policy] = [weights[k].to(self.device, t.dtype) for k, t in zip(self.names, self.live)]
        self.snapshots.move_to_end(policy)
        return self.snapshots[policy]

    def _pull(self) -> None:
        """The learner's newest weights every --refresh seconds, on a thread and a connection of their own, so a pull
        (~48 MB) never holds up a batch; loaded between batches."""
        client = Client(address(self.args.learner), self.args.authkey.encode(), wait=60)
        version = self.version
        while True:
            time.sleep(self.args.refresh)
            pulling = time.perf_counter()
            try:
                new, state, _ = client.call("weights", version)
            except Exception:  # noqa: BLE001  the learner is gone (its time limit, or stop): so is this process
                print("inference: the learner is gone; exiting", flush=True)
                os._exit(0)
            if state is not None:
                version, self.fresh = new, (new, state)
                self.pulls.append(1000 * (time.perf_counter() - pulling))

    def _log(self, started: float) -> None:
        self.logged = time.time()
        elapsed = max(time.time() - started, 1e-6)
        rate, per = self.decisions / elapsed, self.decisions / max(self.batches, 1)
        line = (
            f"inference: {self.decisions} decisions, {rate:.0f}/s, {per:.0f} per batch, busy "
            f"{self.busy / elapsed:.0%}, version {self.version}; ms: batch p10/50/90 {_pct(self.work)}, gather "
            f"{_mean(self.gathered)}, queued {_mean(self.queued)}, "
            f"{len(self.pulls)} weight pulls {_mean(self.pulls)} (loads {_mean(self.loads)}); a batch's part: "
            f"latest {len(self.passes['latest'])} x {_mean(self.passes['latest'])}, snapshots "
            f"{len(self.passes['snapshot'])} x {_mean(self.passes['snapshot'])} ({len(self.snapshots)} held); "
            f"model: {self.models['latest'].stats()}"
        )
        self.work, self.gathered, self.queued, self.pulls, self.loads = [], [], [], [], []
        self.passes = {"latest": [], "snapshot": []}
        print(line, flush=True)
        self.client.call("log", line)


def _pct(values: list) -> str:
    return "/".join(f"{np.percentile(values, q):.0f}" for q in (10, 50, 90)) if values else "-"


def _mean(values: list) -> str:
    return f"{np.mean(values):.1f}" if values else "-"


def merge(inputs: list[dict]) -> dict:
    """Several actors' Agents._inputs as one: unit rows padded to the longest, the rest joined."""
    if len(inputs) == 1:
        return inputs[0]
    rows = max(x["features"]["units"].shape[1] for x in inputs)
    cut = set(UNIT_KEYS)
    features = {}
    for k in inputs[0]["features"]:
        parts = []
        for x in inputs:
            v = x["features"][k]
            if k in cut and v.shape[1] < rows:
                v = np.concatenate([v, np.zeros((v.shape[0], rows - v.shape[1], *v.shape[2:]), v.dtype)], 1)
            parts.append(v)
        features[k] = np.concatenate(parts)
    sizes = [len(x["keys"]) for x in inputs]
    out = {
        "features": features,
        "keys": sum((x["keys"] for x in inputs), []),
        "t": sum((x["t"] for x in inputs), []),
        "static": sum((x["static"] for x in inputs), []),
        "statics": {k: v for x in inputs for k, v in x["statics"].items()},
        "tables": sum((x["tables"] for x in inputs), []),
        "trees": {
            start + i: grid for x, start in zip(inputs, np.cumsum([0, *sizes[:-1]])) for i, grid in x["trees"].items()
        },
        "opponents": None,
        "plans": None,
    }
    if any(x["opponents"] is not None for x in inputs):
        width = next(x["opponents"].shape[1] for x in inputs if x["opponents"] is not None)
        out["opponents"] = np.concatenate([x["opponents"] if x["opponents"] is not None else np.zeros((n, width), np.float32)
                                           for x, n in zip(inputs, sizes)])  # fmt: skip
    if any(x["plans"] is not None for x in inputs):
        blank = empty_plan()
        out["plans"] = {
            k: np.concatenate(
                [
                    x["plans"][k] if x["plans"] is not None else np.repeat(np.asarray(blank[k])[None], n, 0)
                    for x, n in zip(inputs, sizes)
                ]
            )  # fmt: skip
            for k in blank
        }
    return out


def split(out: dict, sizes: list[int]) -> list[dict]:
    """Model.run's outputs for the merged batch, back to each request's seats."""
    replies, start = [], 0
    for n in sizes:
        replies.append({k: (v[start : start + n] if v is not None else None) for k, v in out.items()})
        start += n
    return replies


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--learner", required=True, help="host:port")
    p.add_argument("--listen", default="127.0.0.1:7700")
    p.add_argument("--authkey", default=os.environ.get("WC3_SELFPLAY_KEY", "wc3-selfplay"))
    p.add_argument("--gather", type=float, default=0.002, help="seconds a batch waits for more requests at most")
    p.add_argument("--batch", type=int, default=256, help="games a batch stops waiting at")
    p.add_argument("--refresh", type=float, default=1, help="seconds between weight pulls")
    p.add_argument("--log-every", type=float, default=20, help="seconds between log lines")
    p.add_argument("--snapshot-cache", type=int, default=8, help="snapshot policies' models kept at most")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the latest policy's step (policy/agent.py's Model)",
    )
    return p


def main() -> None:
    Inference(parser().parse_args()).run()


if __name__ == "__main__":
    main()
