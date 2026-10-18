"""Does the round trip through Modal's tunnels grow over a run with no game and no model in it, and what avoids it?

sp41 and cw1: the games' round trip (an actor's command to the sandbox and the observation back) went from ~50 ms to
1,000-1,600 ms after 10-13 minutes, while the sandboxes replied in 5-7 ms (strace of the bridge), their TCP round trip
to the tunnel's entry was 7-40 ms, and both ends sat idle. The first version of this bench (16 tunnels into a CPU
container, 512 connections from 8 sandboxes) showed the same with nothing else in it: 48 ms p50 at minute 9, 581 at
13, 814 at 14.

Every connection plays a game's loop: the container sends a COMMAND-byte command, the sandbox answers REPLY bytes at
once, the container waits DECIDE ms, and again. Three arms at once, SANDBOXES sandboxes each (VM runtime, as the games
sandboxes, in us-east as the container), FLOWS connections a sandbox:
  in:        the sandboxes connect into the container's tunnels (modal.forward, unencrypted), as self-play does;
  reconnect: the same, each connection closed and opened again every RECONNECT seconds;
  out:       each sandbox opens a tunnel of its own (unencrypted_ports) and the container connects out to it.
  tls-out:   the same as out over an encrypted tunnel (encrypted_ports, TLS to its tls_socket);
  tls-in:    the sandboxes connect over TLS into the container's encrypted tunnels (modal.forward).
  ext:       the container connects out to listeners outside Modal (--external host:port,...: GCP VMs,
             training/bench/gcp_link.py), FLOWS x SANDBOXES connections spread over them.
BENCH_REGION (default us-east) places the container and the sandboxes.
Modal (2026-10-12): unencrypted tunnels go through shared relays, some routed through the West Coast from us-east;
encrypted ones connect to the container directly. --arms picks the arms (default: out, tls-out, tls-in).
The container prints each arm's round trip p50 / p90 per minute for MINUTES minutes.

    python -m modal run training/bench/tunnel_latency.py [--arms out,tls-out]
"""

from __future__ import annotations

import os

import modal

app = modal.App("wc3-tunnel-latency")
image = modal.Image.debian_slim(python_version="3.11")
ARMS = ("in", "reconnect", "out", "tls-out", "tls-in", "ext")
REGION = os.environ.get("BENCH_REGION", "us-east")
SANDBOXES, FLOWS, TUNNELS = 2, 64, 2  # per arm; "in" and "reconnect": TUNNELS tunnels each
COMMAND, REPLY, DECIDE, MINUTES, RECONNECT = 300, 4000, 0.012, 25, 120
PORT, BOX_PORT = 7700, 7800

CLIENT = r'''
import os, random, socket, ssl, sys, threading, time
mode, reply = sys.argv[1], os.urandom(int(sys.argv[2]))
TLS = ssl.create_default_context()
def answer(s, until=None):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while until is None or time.time() < until:
        head = b""
        while len(head) < 4:
            d = s.recv(4 - len(head))
            if not d: return
            head += d
        n, got = int.from_bytes(head, "little"), 0
        while got < n:
            d = s.recv(min(65536, n - got))
            if not d: return
            got += len(d)
        s.sendall(len(reply).to_bytes(4, "little") + reply)
def flow(host, port, every, tls=False):
    first = True
    while True:
        s = socket.create_connection((host, port))
        if tls: s = TLS.wrap_socket(s, server_hostname=host)
        until = None if not every else time.time() + (random.uniform(0, every) if first else every)
        first = False
        answer(s, until)
        s.close()
        if not every: return
if mode == "listen":
    listener = socket.create_server(("0.0.0.0", int(sys.argv[3])))
    print(os.environ.get("MODAL_REGION", "?"), "listening", flush=True)
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=answer, args=(conn,), daemon=True).start()
tls = mode == "connect-tls"
flows, every, threads = int(sys.argv[3]), float(sys.argv[4]), []
addresses = sys.argv[5:]
for k in range(flows):
    host, port = addresses[k % len(addresses)].rsplit(":", 1)
    threads.append(threading.Thread(target=flow, args=(host, int(port), every, tls), daemon=True)); threads[-1].start()
print(os.environ.get("MODAL_REGION", "?"), len(threads), "flows", flush=True)
for t in threads: t.join()
'''


def _play(conn, arm: str, samples: list, lock) -> None:
    """One connection's game loop from the container's side, until the sandbox closes it."""
    import os
    import socket
    import time

    command = os.urandom(COMMAND)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            sent = time.perf_counter()
            conn.sendall(len(command).to_bytes(4, "little") + command)
            head = b""
            while len(head) < 4:
                d = conn.recv(4 - len(head))
                if not d:
                    return
                head += d
            n, got = int.from_bytes(head, "little"), 0
            while got < n:
                d = conn.recv(min(65536, n - got))
                if not d:
                    return
                got += len(d)
            with lock:
                samples.append((time.time(), arm, 1000 * (time.perf_counter() - sent)))
            time.sleep(DECIDE)
    except OSError:
        return
    finally:
        conn.close()


def _listen(port: int, arm: str, samples: list, lock) -> None:
    import socket
    import threading

    listener = socket.create_server(("0.0.0.0", port))
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=_play, args=(conn, arm, samples, lock), daemon=True).start()


@app.function(image=image, cpu=16, memory=16384, timeout=(MINUTES + 15) * 60, region=REGION)
def container(arms: tuple = ("out", "tls-out", "tls-in"), external: tuple = ()) -> list:
    import contextlib
    import os
    import socket
    import ssl
    import threading
    import time

    samples, lock = [], threading.Lock()
    print("container", os.environ.get("MODAL_REGION"), os.environ.get("MODAL_CLOUD_PROVIDER"), flush=True)
    boxes_app = modal.App.lookup("wc3-tunnel-latency-boxes", create_if_missing=True)

    with contextlib.ExitStack() as stack:
        boxes: list = []
        stack.callback(lambda: [b.terminate() for b in boxes])

        def box(*argv: str, ports: tuple = (), tls: bool = False) -> modal.Sandbox:
            made = modal.Sandbox.create("python", "-c", CLIENT, *argv, app=boxes_app, image=image, cpu=2, memory=2048,
                                        timeout=(MINUTES + 10) * 60, region=REGION,
                                        **({"encrypted_ports": list(ports)} if tls else {"unencrypted_ports": list(ports)}),
                                        experimental_options={"vm_runtime": True})  # fmt: skip
            boxes.append(made)
            return made

        for i, arm in enumerate(("in", "reconnect", "tls-in")):
            if arm not in arms:
                continue
            arm_ports = [PORT + 10 * i + k for k in range(TUNNELS)]
            tunnels = [stack.enter_context(modal.forward(p, unencrypted=arm != "tls-in")) for p in arm_ports]
            for p in arm_ports:
                threading.Thread(target=_listen, args=(p, arm, samples, lock), daemon=True).start()
            addresses = ["%s:%d" % (t.tls_socket if arm == "tls-in" else t.tcp_socket) for t in tunnels]
            for _ in range(SANDBOXES):
                made = box("connect-tls" if arm == "tls-in" else "connect", str(REPLY), str(FLOWS),
                           str(RECONNECT if arm == "reconnect" else 0), *addresses)
                print("sandbox", arm, next(iter(made.stdout), "?").strip(), flush=True)
        tls = ssl.create_default_context()
        for arm in ("out", "tls-out"):
            if arm not in arms:
                continue
            for _ in range(SANDBOXES):
                made = box("listen", str(REPLY), str(BOX_PORT), ports=(BOX_PORT,), tls=arm == "tls-out")
                print("sandbox", arm, next(iter(made.stdout), "?").strip(), flush=True)  # its listener is up
                tunnel = made.tunnels()[BOX_PORT]
                host, port = tunnel.tls_socket if arm == "tls-out" else tunnel.tcp_socket
                for _ in range(FLOWS):
                    conn = socket.create_connection((host, port))
                    if arm == "tls-out":
                        conn = tls.wrap_socket(conn, server_hostname=host)
                    threading.Thread(target=_play, args=(conn, arm, samples, lock), daemon=True).start()
        if "ext" in arms:
            for k in range(SANDBOXES * FLOWS):
                host, port = external[k % len(external)].rsplit(":", 1)
                conn = socket.create_connection((host, int(port)))
                threading.Thread(target=_play, args=(conn, "ext", samples, lock), daemon=True).start()
        began, rows = time.time(), []
        while time.time() - began < MINUTES * 60:
            time.sleep(60)
            now = time.time()
            with lock:
                recent = [(arm, ms) for t, arm, ms in samples if t > now - 60]
                samples[:] = [s for s in samples if s[0] > now - 60]
            minute = round((now - began) / 60)
            line = []
            for arm in ARMS:
                ms = sorted(m for a, m in recent if a == arm)
                if ms:
                    p50, p90 = ms[len(ms) // 2], ms[int(0.9 * (len(ms) - 1))]
                    rows.append((minute, arm, len(ms) / 60, p50, p90))
                    line.append(f"{arm} {len(ms) / 60:5.0f}/s p50 {p50:6.1f} p90 {p90:6.1f}")
            print(f"minute {minute:2d}: " + " | ".join(line), flush=True)
    return rows


@app.local_entrypoint()
def main(arms: str = "out,tls-out,tls-in", external: str = "") -> None:
    for minute, arm, rate, p50, p90 in container.remote(tuple(arms.split(",")), tuple(filter(None, external.split(",")))):
        print(f"minute {minute:2d} {arm:9s}: {rate:6.0f}/s p50 {p50:6.1f} ms p90 {p90:6.1f} ms")
