"""Sandbox -> Modal container through a TCP tunnel: round-trip time for one step's messages (actions ~1 KB out,
observations of 16 games x 2 players x 9 KB = 288 KB back) and throughput. A CPU function serves; a plain
sandbox (no Wine) is the client."""

import os
import socket
import struct
import time

import modal

app = modal.App("wc3-bench-tunnel")
image = modal.Image.debian_slim(python_version="3.11")

REPLY = int(os.environ.get("REPLY", 16 * 2 * 9000))


def _exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return bytes(data)


@app.function(image=image.env({"REPLY": os.environ.get("REPLY", str(16 * 2 * 9000))}), cpu=2, timeout=1200)
def server(registry: str) -> str:
    reply = b"x" * REPLY
    with modal.forward(7700, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        modal.Dict.from_name(registry, create_if_missing=True)["address"] = f"{host}:{port}"
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 7700))
        listener.listen(64)
        import threading

        served = [0]

        def handle(conn):
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    size = struct.unpack("!I", _exact(conn, 4))[0]
                except ConnectionError:
                    return
                _exact(conn, size)
                conn.sendall(struct.pack("!I", len(reply)) + reply)
                served[0] += 1

        listener.settimeout(120)
        threads = []
        try:
            while True:
                conn, _ = listener.accept()
                t = threading.Thread(target=handle, args=(conn,))
                t.start()
                threads.append(t)
        except TimeoutError:
            pass
        for t in threads:
            t.join()
        return f"served {served[0]}"


CLIENT = r"""
import socket, struct, time, sys
host, port = sys.argv[1].split(":")
def exact(s, n):
    d = bytearray()
    while len(d) < n:
        c = s.recv(n - len(d))
        if not c: raise ConnectionError
        d += c
    return bytes(d)
import threading
msg = b"a" * 1000
CONNS = int(sys.argv[2])
got = [0] * CONNS
def run(k):
    s = socket.create_connection((host, int(port)))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(60):
        s.sendall(struct.pack("!I", len(msg)) + msg)
        n = struct.unpack("!I", exact(s, 4))[0]
        exact(s, n)
        got[k] += n
t = time.time()
threads = [threading.Thread(target=run, args=(k,)) for k in range(CONNS)]
[x.start() for x in threads]; [x.join() for x in threads]
dt = time.time() - t
print(f"{CONNS} connections: {sum(got) / dt / 1e6:.1f} MB/s in total, {60 / dt * CONNS:.0f} round trips/s")
"""

if __name__ == "__main__":
    registry = "wc3-tunnel-test"
    with modal.enable_output(), app.run():
        call = server.spawn(registry)
        reg = modal.Dict.from_name(registry, create_if_missing=True)
        for _ in range(120):
            if "address" in reg:
                break
            time.sleep(1)
        address = reg["address"]
        box = modal.Sandbox.create(app=app, image=image, cpu=2, timeout=600)
        try:
            p = box.exec("python", "-c", CLIENT, address, os.environ.get("CONNS", "8"))
            p.wait()
            print(p.stdout.read().strip(), p.stderr.read().strip()[-300:])
        finally:
            box.terminate()
        print(call.get())
        reg.clear()
