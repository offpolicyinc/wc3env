"""The learner's link to its actors: a TCP service any machine can reach (the same code on one laptop, Modal or
GCP), with Python's multiprocessing.connection (pickles, length-prefixed, an auth key).

    server = Server(("0.0.0.0", 7600), b"key", learner)   # in the learner; learner.handle(kind, payload) answers
    client = Client(("10.0.0.5", 7600), b"key")           # in an actor
    client.call("chunks", [chunk, ...])                   # -> the learner's answer

Payloads are compressed (zlib): chunks carry mostly zeros and repeated tree grids. Within one machine
(compress=False on both ends: the inference process and its actors) they are not: the copy is cheaper than zlib.
"""

from __future__ import annotations

import pickle
import socket
import threading
import time
import zlib
from multiprocessing.connection import Client as _Client
from multiprocessing.connection import Listener, answer_challenge, deliver_challenge


def _nodelay(connection) -> None:
    """TCP_NODELAY on a connection's socket. send_bytes writes a message's length and its body separately (above 16 KB),
    and Nagle's algorithm then holds the body until the length's ACK, which the reader delays up to ~40 ms: 5% of the
    inference process's reads took that long, 88% of its reading time. With it, ~1,860 -> ~2,240 decided steps/s at
    96 games on one machine (2026-10-19)."""
    with socket.socket(fileno=connection.fileno()) as s:
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        finally:
            s.detach()  # the connection keeps its socket


def _pack(value, compress: bool = True) -> bytes:
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(data, 1) if compress else data


def _unpack(data: bytes, compress: bool = True):
    # a reply packed without zlib (Packed(..., compress=False)) is a plain pickle: protocol 2+ starts with 0x80,
    # zlib's streams with 0x78
    return pickle.loads(zlib.decompress(data) if compress and data[:1] != b"\x80" else data)


class Packed:
    """A reply packed once and sent as is to every caller who asks for it (the learner's weights: ~48 MB, once per
    version instead of once per pull; without zlib, whose ~1 s on ~48 MB of float32 saved little, to callers on the
    same machine)."""

    def __init__(self, value, compress: bool):
        self.data = _pack(value, compress)


class Connections:
    """Authenticated connections: accepted on a thread of their own, each key handshake on a short thread of its
    own, so a client that connects and says nothing, or hangs up mid-handshake (the actors' liveness probe), cannot
    stop new connections (sp4's learner stopped accepting after one such reset). take() hands over the new ones;
    the owner answers them (Server: a thread each; the inference process: all from one thread)."""

    def __init__(self, address: tuple[str, int], authkey: bytes, compress: bool = True):
        self.listener = Listener(address)  # no key here: Listener.accept would run the handshake on this thread
        self.authkey, self.compress = authkey, compress
        self.address = self.listener.address
        self.closed = False
        self.new: list = []
        self.lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self.closed:
            try:
                connection = self.listener.accept()
            except OSError:  # closed, or a client gone before it was taken
                time.sleep(0.1)
                continue
            threading.Thread(target=self._handshake, args=(connection,), daemon=True).start()

    def _handshake(self, connection) -> None:
        try:  # what Listener.accept does with a key
            deliver_challenge(connection, self.authkey)
            answer_challenge(connection, self.authkey)
        except Exception:  # noqa: BLE001  a wrong key, a probe, a client gone
            connection.close()
            return
        _nodelay(connection)
        self.arrived(connection)

    def arrived(self, connection) -> None:
        """A new connection, on its handshake's thread: kept for take() (Server serves it there instead)."""
        with self.lock:
            self.new.append(connection)

    def take(self) -> list:
        with self.lock:
            new, self.new = self.new, []
        return new

    def read(self, connection):
        return _unpack(connection.recv_bytes(), self.compress)

    def answer(self, connection, value) -> None:
        connection.send_bytes(value.data if isinstance(value, Packed) else _pack(value, self.compress))

    def close(self) -> None:
        self.closed = True
        self.listener.close()


class Server(Connections):
    """Answers each connection on its own thread: handler(kind, payload) -> reply (the learner)."""

    def __init__(self, address: tuple[str, int], authkey: bytes, handler, compress: bool = True):
        self.handler = handler
        super().__init__(address, authkey, compress)

    def arrived(self, connection) -> None:
        with connection:
            while True:
                try:
                    kind, payload = self.read(connection)
                    self.answer(connection, self.handler(kind, payload))
                except Exception:  # noqa: BLE001  the client went away, or its request failed: this connection ends
                    return


class Client:
    def __init__(self, address: tuple[str, int], authkey: bytes, wait: float = 300, compress: bool = True):
        """Connects, retrying for `wait` seconds while the learner starts."""
        self.compress = compress
        deadline = time.time() + wait
        while True:
            try:
                self.connection = _Client(address, authkey=authkey)
                _nodelay(self.connection)
                break
            except (ConnectionRefusedError, OSError):
                if time.time() > deadline:
                    raise
                time.sleep(2)
        self.lock = threading.Lock()

    def call(self, kind: str, payload=None):
        with self.lock:
            self.connection.send_bytes(_pack((kind, payload), self.compress))
            return _unpack(self.connection.recv_bytes(), self.compress)

    def send(self, kind: str, payload=None) -> None:
        """A request whose answer receive() collects later, in order (one thread; the server answers each
        connection's requests in turn): several may be in flight."""
        self.connection.send_bytes(_pack((kind, payload), self.compress))

    def receive(self):
        return _unpack(self.connection.recv_bytes(), self.compress)

    def close(self) -> None:
        self.connection.close()


def address(text: str) -> tuple[str, int]:
    host, port = text.rsplit(":", 1)
    return host, int(port)
