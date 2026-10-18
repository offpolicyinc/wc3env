"""The learner's batches, made in a process of their own: it takes the actors' chunks, drops those made with weights
more than --max-lag updates old, and writes each batch's arrays (learner.prepare) into a file in memory; the learner
gets the file's name and where each array is in it.

Made on a thread of the learner's own process instead, every update ran beside it and took 0.95 s, against 0.54 s
alone or with the same work in another process (training/bench/learner_prep.py, an H100, mid-game chunks).

The actors send their chunks to the learner's address with the port one lower (chunks_address). The batches take
turns in SLOTS files in memory (/dev/shm), each written in place: the learner holds one batch, the queue between
them a second, and the process fills the third.
"""

from __future__ import annotations

import collections
import contextlib
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import threading
import time

import numpy as np
import torch

from .transport import Server, address

SLOTS = 3
SPREAD = 4  # a batch drawn at random from the oldest SPREAD batches' worth of chunks (a segment's: actor.SEGMENT)


def chunks_address(learner: str) -> str:
    """Where the actors send chunks: the learner's host, its port minus one."""
    host, port = address(learner)
    return f"{host}:{port - 1}"


class _Slot:
    """One of the SLOTS files the batches take turns in: made once, written in place by prepare() (no copy after),
    made again larger when a batch does not fit."""

    def __init__(self, folder: str, k: int, size: int = 1 << 30):
        self.folder, self.k, self.generation = folder, k, 0
        self._make(size)

    def _make(self, size: int) -> None:
        self.path = os.path.join(self.folder, f"slot-{self.k}-{self.generation}.bin")
        self.pages = np.memmap(self.path, np.uint8, "w+", shape=(size,))
        self.pages[:: 4096] = 0  # every page in memory now, not on each batch's first touch

    def write(self, chunks: list[dict], widths: list[int]) -> dict:
        """learner.prepare's batch, written here; each array replaced by (offset, shape, dtype) in self.path."""
        from .learner import prepare  # here: learner imports this module

        while True:
            used = 0

            def allocate(shape, dtype):
                nonlocal used
                offset = -(-used // 64) * 64  # 64-byte aligned
                used = offset + int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                if used > len(self.pages):
                    raise MemoryError(used)
                view = np.ndarray(shape, dtype, buffer=self.pages, offset=offset)
                return (offset, tuple(shape), np.dtype(dtype).str), view

            try:
                return prepare(chunks, widths, allocate)
            except MemoryError:  # a batch larger than the slot (late-game unit counts): a file twice the size
                old = self.path
                self.generation += 1
                self._make(2 * len(self.pages))
                with contextlib.suppress(OSError):  # on Linux a file still mapped can be removed anyway
                    os.remove(old)


def read(batch: dict, pages: np.memmap) -> dict:
    """The learner's side: the batch with each (offset, shape, dtype) as a tensor on its file's pages (no copy)."""
    def tensor(item):
        offset, shape, dtype = item
        return torch.from_numpy(np.ndarray(shape, np.dtype(dtype), buffer=pages, offset=offset))

    out = dict(batch)
    for key in ("dense", "units", "steps"):
        out[key] = {k: tensor(v) for k, v in batch[key].items()}
    out["where"] = [tensor(v) for v in batch["where"]]
    if batch["state"] is not None:
        out["state"] = [tensor(v) for v in batch["state"]]
    return out


def upgo(reward: np.ndarray, value: np.ndarray, done: np.ndarray, bootstrap: np.ndarray, gamma) -> np.ndarray:
    """UPGO's advantages [T, CHANNELS] (AlphaStar; DI-star's upgo loss): a step's return follows what actually came
    after it while each next step did at least as well as the critic expected (its reward and the value after it, against
    the value before it), and takes the critic's value at the first that did worse: a good action is not charged for a
    later mistake it did not make. Each channel on its own, as GAE."""
    out = np.zeros_like(reward)
    after = follow = np.asarray(bootstrap, np.float32)  # V(s_t+1), and what step t's return goes on with
    for i in range(len(reward) - 1, -1, -1):
        on = 0.0 if done[i] else 1.0
        g = reward[i] + gamma * on * follow
        out[i] = g - value[i]
        follow = np.where(reward[i] + gamma * on * after >= value[i], g, value[i])  # step i as good as expected?
        after = value[i]
    return out


def score(segment: list[dict], gamma, lam: float, critic_lam: float) -> None:
    """A segment's chunks (one player's, in order) given their advantages [T, CHANNELS] (GAE at `lam`) and the critic's
    targets (at `critic_lam`), over the whole segment: a step sees real rewards to the segment's end, then the critic's
    value after it (the last chunk's bootstrap). Left as they are when a chunk has no actor values."""
    if not all(np.isfinite(c["value"]).all() for c in segment):
        return
    reward = np.concatenate([c["reward"] for c in segment]).astype(np.float32)
    value = np.concatenate([c["value"] for c in segment]).astype(np.float32)
    done = np.concatenate([c["done"] for c in segment])
    out = {}
    for name, lam_ in (("advantage", lam), ("returns", critic_lam)):
        a = np.zeros_like(reward)
        last, after = np.zeros(reward.shape[1], np.float32), np.asarray(segment[-1]["bootstrap"], np.float32)
        for i in range(len(reward) - 1, -1, -1):
            on = 0.0 if done[i] else 1.0
            last = reward[i] + gamma * after * on - value[i] + gamma * lam_ * on * last
            a[i], after = last, value[i]
        out[name] = a if name == "advantage" else a + value
    out["upgo"] = upgo(reward, value, done, segment[-1]["bootstrap"], gamma)
    at = 0
    for c in segment:
        n = len(c["reward"])
        c["advantage"], c["returns"], c["upgo"] = (out[k][at : at + n] for k in ("advantage", "returns", "upgo"))
        at += n


def _serve(listen: str, authkey: bytes, batch: int, buffer: int, max_lag: int, widths: list[int], folder: str,
           scoring: tuple | None, version, counts, ready) -> None:  # fmt: skip
    """The process: chunks in from the actors (a thread per connection), batches out to the learner. `scoring`
    (each channel's gamma, lambda, the critic's lambda): each segment's advantages computed as it comes in (score)."""
    chunks: collections.deque = collections.deque(maxlen=buffer)
    lock = threading.Lock()
    rng = np.random.default_rng()

    def handle(kind: str, payload):
        if kind != "chunks":
            raise ValueError(kind)
        if scoring is not None:  # a segment's chunks come in one payload, in order
            segments: dict = {}
            for c in payload:
                if "segment" in c:
                    segments.setdefault(int(c["segment"]), []).append(c)
            for segment in segments.values():
                score(segment, *scoring)
        with lock:
            chunks.extend(payload)
            counts["received"].value += len(payload)
        return "ok"

    size = 1 << 30 if folder.startswith("/dev/shm") else 1 << 26  # elsewhere (a laptop's disk) small, grown as needed
    slots = [_Slot(folder, k, size) for k in range(SLOTS)]
    Server(address(listen), authkey, handle)
    made = 0
    while True:
        with lock:
            while chunks and version.value - int(chunks[0]["version"]) > max_lag:
                chunks.popleft()
                counts["dropped"].value += 1
            taken = None
            if len(chunks) >= batch:  # from the oldest SPREAD batches' worth at random: a segment's chunks land in
                # different batches, each batch as many separate moments as before segments
                pool = [chunks.popleft() for _ in range(min(len(chunks), SPREAD * batch))]
                pick = set(rng.choice(len(pool), batch, replace=False).tolist())
                taken = [c for k, c in enumerate(pool) if k in pick]
                chunks.extendleft(reversed([c for k, c in enumerate(pool) if k not in pick]))
        if taken is None:
            time.sleep(0.05)
            continue
        began = time.perf_counter()
        slot = slots[made % SLOTS]
        out = slot.write(taken, widths)
        counts["seconds"].value = time.perf_counter() - began
        ready.put((slot.path, out))  # waits while the learner has not taken the last one
        made += 1


class Batches:
    """The learner's handle on the batch process: get() the next batch, set `version` after each update. A process
    that died is started again (the chunks it held are lost; the actors keep sending)."""

    def __init__(self, listen: str, authkey: bytes, batch: int, buffer: int, max_lag: int, widths: list[int],
                 scoring: tuple | None = None):
        # in memory (Linux's /dev/shm): a batch written to the container's /tmp, on disk, took 0.8 s of its 1.1
        root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        folder = self.folder = tempfile.mkdtemp(prefix="wc3-batches-", dir=root)  # its own: no other learner reads these
        self.mapped: dict = {}  # path -> its pages, mapped once here (and registered with CUDA)
        self.ctx = mp.get_context("spawn")  # the learner's CUDA context must not be forked
        self.version = self.ctx.Value("i", 0)
        self.counts = {k: self.ctx.Value("d", 0.0) for k in ("received", "dropped", "seconds")}
        self.args = (listen, authkey, batch, buffer, max_lag, widths, folder, scoring)
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        self.ready = self.ctx.Queue(maxsize=1)
        self.proc = self.ctx.Process(target=_serve, args=(*self.args, self.version, self.counts, self.ready), daemon=True)
        self.proc.start()

    def get(self, timeout: float) -> dict | None:
        """The next batch (learner.prepare's, as tensors on its file), or None after `timeout` seconds."""
        try:
            path, batch = self.ready.get(timeout=timeout)
        except queue.Empty:
            if not self.proc.is_alive():
                self.restarts += 1
                self._start()
            return None
        return read(batch, self._pages(path))

    def _pages(self, path: str) -> np.memmap:
        """A slot file's pages, mapped the first time; on a GPU machine registered as page-locked then, so copies from
        it run at full speed without waiting (unregistered: 0.13 s of each update, registered ~0.02)."""
        pages = self.mapped.get(path)
        if pages is None:
            pages = self.mapped[path] = np.memmap(path, np.uint8, "r+")
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostRegister(pages.ctypes.data, pages.nbytes, 0)
        return pages

    def close(self) -> None:
        """The process stopped, the files unregistered, unmapped and removed (a later learner in this process may get
        the same addresses, which CUDA will not register twice)."""
        self.proc.terminate()
        for pages in self.mapped.values():
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostUnregister(pages.ctypes.data)
            pages._mmap.close()
        self.mapped.clear()
        shutil.rmtree(self.folder, ignore_errors=True)

    @property
    def dropped(self) -> int:
        return int(self.counts["dropped"].value)

    @property
    def seconds(self) -> float:
        return self.counts["seconds"].value
