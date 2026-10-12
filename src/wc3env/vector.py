"""Many games stepped asynchronously by worker processes, for RL rollouts.

    with VectorSession([config] * 16, group_size=2) as games:
        for i, observations in games.reset():          # every game's first observations
            games.send(i, policy(observations))
        while training:
            i, observations, done, info = games.recv()  # whichever game finishes its step first
            games.send(i, policy(observations))

Each worker process hosts a group of games (one thread and GameSession each; use
`observation="binary"`). `send` returns at once; `recv` returns the next finished step from any game,
so the policy can batch whatever is ready while the others simulate. A finished episode resets inside
its worker: `recv` then reports `done` with the final observations in `info["final_observations"]` and
the new episode's first observations as `observations`, so a reload never stalls the other games.

Workers speak length-prefixed pickles over stdin/stdout, so the host needs no Windows: on a Linux
host, `launch=wine_workers()` runs each worker as Windows Python under Wine in its own copy of the Wine
prefix. Each prefix has its own wineserver, which without ntsync stops scaling at about four games;
two games per worker measured best (docs/compatibility.md).

The workers can also be on another machine: `remote_workers(port)` makes each worker a TCP connection
that machine opens to this one, where `python -m wc3env.vector --connect host:port --workers N` runs N
local workers and carries each one's stream over its own connection, zstd-compressed. The games then
need only CPUs where they run, and the policy deciding their steps can sit next to a GPU elsewhere.
The other way round, `python -m wc3env.vector --listen PORT --workers N` on the games' machine and
`dial_workers(address)` on the host: each worker is a connection the host opens to the games' machine. On Modal
this is the way that stays fast: connections into a container's tunnel (modal.forward) slowed from ~50 ms to
1,000+ ms a round trip after 10-15 minutes (sp41, training/bench/tunnel_latency.py), connections out to a
sandbox's tunnel (unencrypted_ports) held ~11 ms.
"""

from __future__ import annotations

import os
import pickle
import queue
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .session import GameConfig, GameSession

# worker index -> argv and environment (None: inherit) of a local process, or a connected remote one
Launch = Callable[[int], "tuple[list[str], dict | None] | RemoteWorker"]


def local_workers(worker: int) -> tuple[list[str], dict | None]:
    """Workers in this Python (Windows)."""
    return [sys.executable, "-m", "wc3env.vector"], None


def wine_workers(prefix: str | None = None, python: str = r"C:\Python311\python.exe") -> Launch:
    """Workers as Windows Python under Wine, from a Linux host: worker 0 in `prefix` ($WINEPREFIX), the
    others in copies of it in the temporary directory, each with its own wineserver."""
    base = Path(prefix or os.environ["WINEPREFIX"])

    def launch(worker: int) -> tuple[list[str], dict | None]:
        path = base if worker == 0 else Path(tempfile.gettempdir()) / f"wine-prefix-{worker}"
        if not path.exists():
            link_prefix(base, path)
        return ["wine", python, "-m", "wc3env.vector"], {**os.environ, "WINEPREFIX": str(path)}

    return launch


def link_prefix(base: Path, path: Path, small: int = 256 * 1024) -> None:
    """A Wine prefix like `base` at `path`: its large files hard-linked (read-only DLLs, Python, game data; 1.6 GB
    of the image's prefix), the registry and small files copied, since Wine and the games write those. Falls
    back to copying where linking fails (another filesystem). Written beside `path`, then renamed: a copy cut
    short is never mistaken for a whole one."""
    part = path.with_name(path.name + ".part")
    shutil.rmtree(part, ignore_errors=True)

    def place(src: str, dst: str) -> None:
        if not src.endswith(".reg") and os.path.getsize(src) >= small:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    shutil.copytree(base, part, symlinks=True, copy_function=place)
    os.rename(part, path)


# zstd's window: 8 MB, many steps of a game. Each step's message is mostly the one before (a step changes few units),
# so with a window that reaches back to it a step costs ~1 KB instead of zlib's ~12 (its window is 32 KB, under
# one message): ~90% less traffic, and the network between sandboxes and their host is billed by the byte
WINDOW_LOG = 23


class _Inflating:
    """A readable stream of what a zstd-compressing peer (_Deflating) sent over `raw`."""

    def __init__(self, raw):
        import zstandard  # (only where streams cross machines: pip install zstandard)

        self.raw, self.buffer = raw, bytearray()
        self.z = zstandard.ZstdDecompressor(max_window_size=1 << WINDOW_LOG).decompressobj()

    def read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            data = self.raw.recv(1 << 16)
            if not data:
                break
            self.buffer += self.z.decompress(data)
        out = bytes(self.buffer[:n])
        del self.buffer[:n]
        return out


class _Deflating:
    """A writable stream that zstd-compresses into `raw`, every flush a complete block for the reader."""

    def __init__(self, raw):
        import zstandard

        self.zstandard, self.raw = zstandard, raw
        params = zstandard.ZstdCompressionParameters.from_level(3, window_log=WINDOW_LOG)
        self.z = zstandard.ZstdCompressor(compression_params=params).compressobj()

    def write(self, data: bytes) -> None:
        self.raw.sendall(self.z.compress(data))

    def flush(self) -> None:
        self.raw.sendall(self.z.flush(self.zstandard.COMPRESSOBJ_FLUSH_BLOCK))

    def close(self) -> None:
        try:
            self.raw.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class RemoteWorker:
    """A worker on another machine, over one connection: what VectorSession uses of a local process."""

    def __init__(self, conn: socket.socket):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conn, self.stdin, self.stdout = conn, _Deflating(conn), _Inflating(conn)

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.conn.close()


def remote_workers(port: int, host: str = "0.0.0.0") -> Launch:
    """Workers that connect in to `port` (python -m wc3env.vector --connect ...), one connection each."""
    listener = socket.create_server((host, port), reuse_port=False)

    def launch(worker: int) -> RemoteWorker:
        conn, _ = listener.accept()
        return RemoteWorker(conn)

    return launch


HELLO = b""  # what a listener sends on each connection it carries (dial_workers waits for it)


def dial_workers(address: str, give_up: float = 600.0) -> Launch:
    """Workers on another machine that listens at `address` (python -m wc3env.vector --listen ...), one
    connection each, opened from here; retried until it answers, for up to `give_up` seconds."""
    host, port = address.rsplit(":", 1)

    def launch(worker: int) -> RemoteWorker:
        began = time.time()
        while True:
            try:
                conn = socket.create_connection((host, int(port)), timeout=30)
                # the listener's hello: a tunnel in front of it (Modal's) accepts a connection before anything listens
                # behind it, then closes it (cbe05: every actor's first reset failed while the bridges set up Wine)
                if conn.recv(1) == HELLO:
                    conn.settimeout(None)
                    return RemoteWorker(conn)
                conn.close()
            except OSError:
                pass
            if time.time() - began > give_up:
                raise ConnectionError(f"no worker listening at {address} after {give_up:.0f} s")
            time.sleep(5)

    return launch


def _relay(address: str, launch: Launch, worker: int, give_up: float = 600.0) -> None:
    """Local workers' streams over connections to `address`, compressed both ways, one after another: when the
    host ends a session its connection closes and the next one is opened, until the host has refused new
    connections for `give_up` seconds (it is gone)."""

    host, port = address.rsplit(":", 1)
    refused_since = None
    while True:
        try:
            conn = socket.create_connection((host, int(port)), timeout=30)
        except OSError:
            refused_since = refused_since or time.time()
            if time.time() - refused_since > give_up:
                return
            time.sleep(5)
            continue
        refused_since = None
        conn.settimeout(None)
        _carry(conn, launch, worker)


def _carry(conn: socket.socket, launch: Launch, worker: int) -> None:
    """One session: a fresh local worker, its stream over `conn` until either side ends."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    argv, env = launch(worker)
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    inflate, deflate = _Inflating(conn), _Deflating(conn)

    def inbound() -> None:  # messages to the worker
        while (message := _read(inflate)) is not None:
            _write(proc.stdin, message)
        proc.stdin.close()

    threading.Thread(target=inbound, daemon=True).start()
    try:
        while (message := _read(proc.stdout)) is not None:  # its replies, out
            _write(deflate, message)
    except OSError:  # the host went away mid-session
        proc.kill()
    deflate.close()
    proc.wait()
    conn.close()


def listen(port: int, workers: int, launch: Launch = local_workers, give_up: float = 600.0) -> None:
    """Local workers for a host that connects in (dial_workers): each connection carried by a fresh worker under the
    lowest worker index not in use (at most `workers` at once: a Wine worker's index names its prefix). Ends once
    no connection has been open for `give_up` seconds after the first one (the host is gone)."""
    listener = socket.create_server(("0.0.0.0", port))
    listener.settimeout(5)
    free = list(range(workers))
    lock, idle_since, served = threading.Lock(), time.time(), False

    def session(conn: socket.socket, worker: int) -> None:
        nonlocal idle_since
        try:
            _carry(conn, launch, worker)
        finally:
            with lock:
                free.append(worker)
                free.sort()
                if len(free) == workers:
                    idle_since = time.time()

    while True:
        try:
            conn, _ = listener.accept()
        except socket.timeout:  # noqa: UP041  (TimeoutError only from 3.10; the bridges' Linux Python may be older)
            with lock:
                if served and len(free) == workers and time.time() - idle_since > give_up:
                    return
            continue
        conn.settimeout(None)
        with lock:
            if not free:  # more connections than workers: the host's mistake, not a reason to share a prefix
                conn.close()
                continue
            worker, served = free.pop(0), True
        try:
            conn.sendall(HELLO)
        except OSError:  # the host already gave up on it
            with lock:
                free.append(worker)
                free.sort()
            conn.close()
            continue
        threading.Thread(target=session, args=(conn, worker), daemon=True).start()


def connect(address: str, workers: int, launch: Launch = local_workers) -> None:
    """`workers` local workers, each carried over its own connection to the host at `address`."""
    threads = [threading.Thread(target=_relay, args=(address, launch, w)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _hops(message) -> dict | None:
    """A step reply's timings (info["hops"]: ms per part of the way, each part adding its own), or None."""
    if len(message) == 3 and message[1] == "ok" and isinstance(message[2], tuple) and len(message[2]) == 3:
        info = message[2][2]
        if isinstance(info, dict):
            return info.get("hops")
    return None


def _write(stream, message) -> None:
    data = pickle.dumps(message, pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<I", len(data)) + data)
    stream.flush()


def _read(stream):
    """The next message, or None at end of stream."""
    head = stream.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(stream.read(struct.unpack("<I", head)[0]))


class VectorSession:
    def __init__(
        self,
        configs: Sequence[GameConfig],
        *,
        group_size: int = 1,
        launch: Launch = local_workers,
        speed: float = 2048,
        wait_floor: int = 1,
    ):
        if not configs or group_size < 1:
            raise ValueError("VectorSession needs at least one GameConfig and a positive group_size")
        self._games = len(configs)
        self._where = [(i // group_size, i % group_size) for i in range(self._games)]  # game -> (worker, slot)
        self._game = {where: i for i, where in enumerate(self._where)}
        self._results: queue.Queue = queue.Queue()
        self._waiting: list = []  # results read while waiting for a particular game
        self._busy: set[int] = set()
        self._sent: dict[int, float] = {}  # game -> when its last message went out (perf_counter)
        self._workers = []
        for w in range(0, self._games, group_size):
            launched = launch(w // group_size)
            if isinstance(launched, RemoteWorker):
                proc = launched
            else:
                argv, env = launched
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            _write(proc.stdin, (list(configs[w : w + group_size]), speed, wait_floor))
            threading.Thread(target=self._drain, args=(len(self._workers), proc), daemon=True).start()
            self._workers.append(proc)

    def __len__(self) -> int:
        return self._games

    def _drain(self, worker: int, proc) -> None:
        """Every reply of one worker into the shared queue; a worker that exits fails its games."""
        while (message := _read(proc.stdout)) is not None:
            slot, status, payload = message
            i = self._game[worker, slot]
            hops = _hops(message)
            if hops is not None and i in self._sent:
                hops["h_round"] = 1000 * (time.perf_counter() - self._sent[i])
                hops["_arrived"] = time.perf_counter()
            self._results.put((i, status, payload))
        error = f"worker {worker} exited with {proc.wait()}"
        for (w, _), i in self._game.items():
            if w == worker:
                self._results.put((i, "error", error))

    def _take(self, game: int | None = None, timeout: float | None = None):
        """The next reply (for `game` only, if given; others wait for recv) as (index, payload)."""
        while True:
            match = next((r for r in self._waiting if game is None or r[0] == game), None)
            if match:
                self._waiting.remove(match)
                i, status, payload = match
                break
            try:
                self._waiting.append(self._results.get(timeout=timeout))
            except queue.Empty:
                return None
        self._busy.discard(i)
        if status == "error":
            raise RuntimeError(f"game {i}: {payload}; call reset_one({i})")
        hops = _hops((None, status, payload))
        if hops is not None and "_arrived" in hops:
            hops["h_queue"] = 1000 * (time.perf_counter() - hops.pop("_arrived"))
        return i, payload

    def _send(self, i: int, message) -> None:
        worker, slot = self._where[i]
        self._busy.add(i)
        _write(self._workers[worker].stdin, (slot, message))
        self._sent[i] = time.perf_counter()

    def reset(self) -> list[tuple[int, dict]]:
        """Reset every game (in parallel); returns (index, observations) for each, in index order."""
        for i in range(self._games):
            self._send(i, "reset")
        return [self._take(i) for i in range(self._games)]

    def reset_one(self, i: int) -> dict:
        self._send(i, "reset")
        return self._take(i)[1]

    def debug(self, i: int, op: str, **args) -> dict:
        """A staging or tuning op (GameSession.debug) on game i, between its steps; returns its result."""
        if i in self._busy:
            raise RuntimeError(f"game {i} is stepping; recv() its result first")
        self._send(i, ("debug", op, args))
        return self._take(i)[1]

    def stage(self, i: int, ops: list, actions: dict) -> None:
        """Start game i's next step after staging ops ([(op, args)], GameSession.debug in order) in its worker; returns
        at once, and recv reports the step as any other. One message, so a staged episode costs no round trips of its
        own. A "spawn" may also carry, for each unit it makes: "level" (a hero's), "mana", "items" (type ids, given),
        and "learn" (a hero's skill ids, learned in that order before the step, one a step after a first step that
        makes the new units commandable: those steps' observations are not reported)."""
        if i in self._busy:
            raise RuntimeError(f"game {i} is already stepping; recv() its result first")
        self._send(i, ("stage", ops, actions))

    def send(self, i: int, actions: dict, decisions: dict | None = None, replay: str | None = None) -> None:
        """Start game i's next step with one action batch per agent slot (and the decisions behind them, for the
        hook's features: GameSession.step); returns at once. `replay`: a path where the game's worker sees it, for
        the native replay of an episode this step ends (saved before the worker starts the next)."""
        if i in self._busy:
            raise RuntimeError(f"game {i} is already stepping; recv() its result first")
        self._send(i, ("step", actions, decisions, replay))

    def save_replay(self, i: int, path: str) -> str:
        """Game i's episode so far as a native replay (GameSession.save_replay) at `path`, as its worker sees it; for
        an episode stopped by the caller (a time limit). Recording stops: reset the game next."""
        if i in self._busy:
            raise RuntimeError(f"game {i} is stepping; recv() its result first")
        self._send(i, ("replay", path))
        return self._take(i)[1]

    def recv(self, timeout: float | None = None):
        """(index, observations, done, info) of the first game to finish a step, or None on timeout."""
        if not self._busy:
            raise RuntimeError("no game is stepping; send() first")
        taken = self._take(timeout=timeout)
        return None if taken is None else (taken[0], *taken[1])

    def close(self) -> None:
        for proc in self._workers:
            try:
                proc.stdin.close()  # end of input: the worker closes its games and exits
            except OSError:
                pass
        for proc in self._workers:
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _debug(session: GameSession, op: str, args: dict) -> dict:
    """GameSession.debug, its error naming the op and its arguments."""
    try:
        return session.debug(op, **args) or {}
    except Exception as exc:
        raise RuntimeError(f"{op} {args}: {exc}") from exc


def _stage(session: GameSession, ops: list, actions: dict) -> None:
    """VectorSession.stage's ops, in the worker."""
    learns = {}  # hero -> (its player, its skills to learn)
    for op, args in ops:
        args = dict(args)
        extra = (
            ("level", "mana", "items", "learn") if op == "spawn" else ()
        )  # what each unit made gets (a research has a level of its own)
        level, mana, items, learn = (
            args.pop(k, None) if k in extra else None for k in ("level", "mana", "items", "learn")
        )
        for unit in _debug(session, op, args).get("unit_ids", []):
            if level is not None:
                _debug(session, "level", {"unit_id": unit, "level": level})
            if mana is not None:
                _debug(session, "mana", {"unit_id": unit, "value": mana})
            for item in items or ():
                _debug(session, "give", {"unit_id": unit, "type_id": item})
            if learn:
                learns[unit] = (args["player"], list(learn))
    if learns:  # the new units in the players' observations first: only units seen there take commands
        session.step({slot: [] for slot in actions})
    for k in range(max((len(s) for _, s in learns.values()), default=0)):
        batch = {slot: [] for slot in actions}
        for unit, (player, skills) in learns.items():
            if k < len(skills):
                batch[player].append({"unit_id": unit, "command": "learn", "arguments": {"ability_id": skills[k]}})
        session.step(batch)


def _serve() -> None:
    """A worker: the games of one group, each on its own thread, answering over stdin/stdout."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the replies
    configs, speed, wait_floor = _read(stdin)
    if any(c.observation == "binary" for c in configs):
        # Before the blocking reads of stdin below: on Windows, numpy's import touches the standard
        # handles and would wait behind a pending read, deadlocking the first game thread that imports it.
        from . import binary  # noqa: F401
    lock = threading.Lock()
    inboxes = [queue.Queue() for _ in configs]

    def game(slot: int, config: GameConfig) -> None:
        session, tuned = GameSession(config), False

        def reset():
            nonlocal tuned
            observations = session.reset()
            if not tuned:  # the session replays tuning on the processes it launches later
                session.debug("speed", factor=speed)
                session.debug("waitfloor", ms=wait_floor)
                tuned = True
            return observations

        try:
            while (message := inboxes[slot].get()) is not None:
                began = time.perf_counter()
                try:
                    if message == "reset":
                        reply = reset()
                    elif message[0] == "debug":
                        reply = session.debug(message[1], **message[2])
                    elif message[0] == "replay":
                        reply = str(session.save_replay(message[1]))
                    else:  # ("step", actions, decisions, replay path or None), or ("stage", debug ops, actions)
                        if message[0] == "stage":
                            _stage(session, message[1], message[2])
                            message = ("step", message[2], None)
                        observations, done, info = session.step(message[1], decisions=message[2])
                        if done:
                            info["final_observations"] = observations
                            if len(message) > 3 and message[3]:
                                info["replay"] = str(session.save_replay(message[3]))
                            observations = reset()
                        # ms per part of the way (the host adds its own): here, the game's step
                        info["hops"] = {"w_step": 1000 * (time.perf_counter() - began)}
                        reply = (observations, done, info)
                    status = "ok"
                except Exception as exc:  # noqa: BLE001  report; the host asks for a fresh episode
                    status, reply = "error", f"{type(exc).__name__}: {exc}"
                with lock:
                    _write(stdout, (slot, status, reply))
        finally:
            session.close()

    threads = [threading.Thread(target=game, args=(k, c)) for k, c in enumerate(configs)]
    for thread in threads:
        thread.start()
    while (message := _read(stdin)) is not None:
        slot, payload = message
        inboxes[slot].put(payload)
    for inbox in inboxes:
        inbox.put(None)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    if "--connect" in sys.argv or "--listen" in sys.argv:  # a machine whose workers serve a host elsewhere
        import argparse

        p = argparse.ArgumentParser()
        side = p.add_mutually_exclusive_group(required=True)
        side.add_argument("--connect", help="host:port of the VectorSession (remote_workers)")
        side.add_argument("--listen", type=int, help="port the VectorSession connects in to (dial_workers)")
        p.add_argument("--workers", type=int, required=True)
        p.add_argument("--wine", action="store_true", help="the workers as Windows Python under Wine")
        a = p.parse_args()
        launch = wine_workers() if a.wine else local_workers
        if a.listen:
            listen(a.listen, a.workers, launch)
        else:
            connect(a.connect, a.workers, launch)
    else:
        _serve()
