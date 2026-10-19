"""Self-play on Modal: the learner, the inference process and the actors in one GPU container, the games in Wine
sandboxes (the worker image of docker/modal_run.py).

    python -m training.cloud.selfplay start --name sp1 --init runs/bc10/last.pt --actors 16 --per-box 2 --games 32 \\
        --ai-games 32 --group 2 --cpu 4 --memory 32768 --box-cpu 24 -- --hours 1
    python -m training.cloud.selfplay log --name sp1          # the learner's log and the last lines of the others
    python -m training.cloud.selfplay stop --name sp1         # the container and every sandbox
    python -m training.cloud.selfplay start --name cb1 --mode combat ...   # fights, not games (training/selfplay/modes)
    python -m training.cloud.selfplay start --name fx1 --mode fast ...     # 3x-speed games (python -m tools.prepare.fastmap)

```
 SANDBOX x (actors / per-box)           GPU CONTAINER
 ┌───────────────────────────┐  tunnel   ┌────────────────────────────────────────────────┐
 │ per_box bridges           │ ◄───────► │ ACTOR x actors (CPU only)   INFERENCE (GPU)      │
 │ (wc3env.vector --listen)  │  one per  │   features, masks  ───────►  model for all     │
 │ each: games / group Wine  │  actor    │   commands         ◄───────                    │
 │ workers                   │           │   64-step chunks  ───────►  LEARNER (GPU)      │
 └───────────────────────────┘           └────────────────────────────────────────────────┘
```

Each sandbox opens one TCP tunnel per bridge (one bridge per actor) and the launcher records their addresses in
the Modal Dict wc3-selfplay; the container starts each actor once its address is there, and the actor connects out
to it, one connection per Wine worker. Out, not in: connections into the container's own tunnels (modal.forward)
slowed from ~50 ms a round trip to 1,000-1,600 after 10-15 minutes (sp41, cw1), out to the sandboxes' held ~11 ms
(training/bench/tunnel_latency.py). The license files are streamed
into each sandbox, never put in an image or on a command line. Everything a run writes is in the Volume under
runs/<name>/ (the actors' and the inference process's logs in runs/<name>/actors/).

A slow sandbox is replaced mid-run: a tunnel can be slow from the start or turn slow later (cb1: one host's round
trips 90-140 ms, another's 6-8.6 s after half an hour, the games' own step 5 ms on both). Every minute the container
counts each actor's game steps; a sandbox one of whose actors made under SLOW_SHARE of the actors' median for
SLOW_CHECKS minutes running (a stuck one, none) is asked for again (the registry's "<name>:slow:<box>"); the launcher,
which stays on after the launch (`watch`), makes a new one, publishes its addresses ("<name>:box:<box>") and stops
the old; the container starts those actors again on the new one. At most MAX_REPLACED times a sandbox.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from ..bench.common import FLASH_ATTN_WHEEL
from ..selfplay import modes
from ..selfplay.modes.game import Games
from ..train.track import modal_secrets
from .scripts import bridge, learner_hours

ROOT = Path(__file__).resolve().parents[2]
VOLUME = modal.Volume.from_name("wc3-dataset")
REGISTRY = "wc3-selfplay"  # Modal Dict: run name -> {games: tunnel addresses, sandboxes, call, app}
PORT = 7600  # inside the container: the learner
BRIDGE_PORT = 7800  # in a games sandbox: BRIDGE_PORT + j, bridge j's listener and its tunnel
MAX_RESTARTS = 5  # an inference process started again at most this often
KERNELS = Path("/dataset/cache/inductor.tar")  # torch.compile's cache, kept between runs (one file: a folder of
# thousands took the Volume over 10 minutes to copy)
TRIES = (
    2  # GPU containers made in --region before one is pinned to --want (sp31: 3 of 3 landed in us-west and us-central)
)
INFERENCE_PORT = 7500  # inside the container: INFERENCE_PORT + i, the actors' link to inference process i
SLOW_SHARE, SLOW_CHECKS = 0.5, 3  # a sandbox replaced: an actor of it under this share of the median, minutes running
MAX_REPLACED = 3  # replacements of one sandbox at most (a slow stretch on Modal's side does not loop)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.8.0", "zstandard", "wandb", "wandb-workspaces")
    .run_commands(f"pip install {FLASH_ATTN_WHEEL}")  # the learner's packed attention (without it: padded)
    .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
    .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
)
app = modal.App("wc3-selfplay", image=image)


@app.function(
    gpu="H100", cpu=32, memory=131072, timeout=24 * 3600, volumes={"/dataset": VOLUME}, secrets=modal_secrets()
)
def container(
    name: str, learner_argv: list[str], actors: int, actor_argv: list[str], inference: int = 2, per_box: int = 2
) -> None:
    """The learner, `inference` inference processes (actor k decides through process k % inference: each one
    does its CPU work while another's batch is on the GPU) and `actors` actors (CPU only: no CUDA context each).
    The run ends when the learner does (its --hours, or stop); an inference process that dies is started again (its
    actors start new games against it), up to MAX_RESTARTS times."""
    import contextlib
    import os
    import shutil

    key = secrets.token_hex(16)
    # torch.compile's kernels (the learner's and the inference processes'): on local disk, from the Volume's copy, and
    # back into it once this run's are made. With them the inference step's shapes take ~1 minute to make, not
    # 6-12 (training/bench/compile_cost.py); read from the Volume directly it was as slow as none (sp22)
    cache = Path("/tmp/inductor")
    cache.mkdir(parents=True, exist_ok=True)
    if KERNELS.exists():
        began = time.time()
        subprocess.run(["tar", "-xf", str(KERNELS), "-C", str(cache)], check=False)
        print(f"torch.compile's cache unpacked from the Volume in {time.time() - began:.0f} s", flush=True)
    env = {**os.environ, "PYTHONPATH": "/root", "TORCHINDUCTOR_CACHE_DIR": str(cache)}
    logs = Path(f"/dataset/runs/{name}/actors")  # in the Volume, committed every minute below: readable mid-run
    logs.mkdir(parents=True, exist_ok=True)

    def start(module: str, *argv: str, log: str | None = None, gpu: str | None = None) -> subprocess.Popen:
        """gpu: the GPUs the process sees ("" for none: the actors); None, all of them."""
        out = open(logs / log, "w") if log else None  # noqa: SIM115 (open while the process runs)
        return subprocess.Popen(["python", "-m", module, *argv, "--authkey", key], stdout=out, stderr=subprocess.STDOUT,
                                env=env if gpu is None else {**env, "CUDA_VISIBLE_DEVICES": gpu})  # fmt: skip

    # with two GPUs (--gpu H100:2), the inference processes get the second one to themselves: on a shared GPU their
    # batches take turns with the learner's kernels (sp14: batch p90 106 ms with the learner, 45 without), and MPS
    # did not help (a small batch next to a busy process: 7.5 ms alone, 17 taking turns, 50 under MPS, 14 under MPS
    # with the busy process held to half the GPU)
    gpus = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines())
    learner_gpu, inference_gpu = ("0", str(gpus - 1)) if gpus > 1 else (None, None)
    learner = f"127.0.0.1:{PORT}"
    with contextlib.ExitStack() as stack:
        # the games sandboxes end with this container, however it ends (their bridges would wait 10 minutes)
        stack.callback(_stop_sandboxes, name)
        registry = modal.Dict.from_name(REGISTRY, create_if_missing=True)
        registry[name] = {**registry.get(name, {}), "container_at": _placement(os.environ), "hardware": _hardware()}
        learner_proc = start("training.selfplay.learner", *learner_argv, "--listen", learner, gpu=learner_gpu)
        time.sleep(20)  # the learner loads its checkpoint before the others ask for weights

        def decider(i: int, restart: int = 0) -> subprocess.Popen:
            log = ("inference" if i == 0 else f"inference-{i}") + (f"-restart{restart}" if restart else "") + ".log"
            return start("training.selfplay.inference", "--learner", learner, "--listen", f"127.0.0.1:{INFERENCE_PORT + i}",
                         log=log, gpu=inference_gpu)  # fmt: skip

        deciders = [decider(i) for i in range(inference)]
        restarts = [0] * inference
        waited = time.time()
        while len(addresses := registry.get(name, {}).get("games", [])) < actors:  # the launcher makes the sandboxes
            if time.time() - waited > 1800:
                raise RuntimeError("the games sandboxes' addresses did not come in 30 minutes")
            time.sleep(5)

        def actor(k: int, restart: int = 0) -> subprocess.Popen:
            log = f"actor-{k}" + (f"-restart{restart}" if restart else "") + ".log"
            return start("training.selfplay.actor", "--learner", learner, "--inference", f"127.0.0.1:{INFERENCE_PORT + k % inference}",
                         "--dial", addresses[k], "--seed", str(k), *actor_argv, log=log, gpu="")  # fmt: skip

        actor_procs = [actor(k) for k in range(actors)]
        actor_logs = [logs / f"actor-{k}.log" for k in range(actors)]
        boxes = -(-actors // per_box)
        slow, asked, swaps = [0] * boxes, [False] * boxes, [0] * boxes
        counted = None  # each actor's game steps at the last check
        ticks = 0
        while learner_proc.poll() is None:
            time.sleep(10)
            ticks += 1
            # an inference process that died is started again (its actors reconnect to it), up to MAX_RESTARTS
            # times each; sp29 lost all its games to one that crashed
            for i, d in enumerate(deciders):
                if d.poll() is not None:
                    if restarts[i] == MAX_RESTARTS:
                        print(f"inference {i} died {MAX_RESTARTS + 1} times: the run ends", flush=True)
                        learner_proc.kill()
                        break
                    restarts[i] += 1
                    print(f"inference {i} exited ({d.returncode}): started again", flush=True)
                    deciders[i] = decider(i, restarts[i])
            if ticks == 48:  # everything compiled by now (the inference shapes are made at the first batch)
                packed = Path("/tmp/inductor.tar")  # packed locally, then copied (a rename in the Volume did not take)
                if subprocess.run(["tar", "-cf", str(packed), "-C", str(cache), "."], check=False).returncode == 0:
                    shutil.copyfile(packed, KERNELS)
            if ticks % 6 == 0:
                VOLUME.commit()  # the logs and saves visible outside, every minute
                # a slow sandbox's replacement in: its actors started again on it
                for b in range(boxes):
                    mine = range(b * per_box, min((b + 1) * per_box, actors))
                    new = registry.get(f"{name}:box:{b}")
                    if new and new != [addresses[k] for k in mine]:
                        swaps[b] += 1
                        for j, k in enumerate(mine):
                            actor_procs[k].kill()
                            addresses[k] = new[j]
                            actor_procs[k] = actor(k, swaps[b])
                            actor_logs[k] = logs / f"actor-{k}-restart{swaps[b]}.log"
                        slow[b], asked[b] = 0, False
                        print(f"games sandbox {b} replaced: actors {list(mine)} started again on it", flush=True)
                # a slow one asked for again
                now = [_steps(path) for path in actor_logs]
                if counted is not None:
                    for b in slow_boxes([n - c for n, c in zip(now, counted)], per_box, slow, asked):
                        asked[b] = True
                        registry[f"{name}:slow:{b}"] = time.time()
                        print(f"games sandbox {b} slow for {SLOW_CHECKS} minutes: asked for a new one", flush=True)
                counted = now
        for p in deciders + actor_procs:
            p.kill()
    VOLUME.commit()


def _stop_sandboxes(name: str) -> None:
    for box_id in modal.Dict.from_name(REGISTRY).get(name, {}).get("sandboxes", []):
        try:
            modal.Sandbox.from_id(box_id).terminate()
        except Exception:  # noqa: BLE001  already gone
            pass


def _steps(path: Path) -> int:
    """An actor's game steps so far, from the last report in its log (0 before the first)."""
    import re

    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8192))
            found = re.findall(rb"actor \d+: (\d+) game steps", f.read())
    except OSError:
        return 0
    return int(found[-1]) if found else 0


def slow_boxes(made: list[int], per_box: int, slow: list[int], asked: list[bool]) -> list[int]:
    """The sandboxes to ask for again, given each actor's game steps in the last minute (`made`; one started again
    has fewer than before: not counted this time). `slow` counts each sandbox's slow minutes running, in place; one
    already `asked` for waits."""
    import statistics

    typical = statistics.median([m for m in made if m >= 0] or [0])
    out = []
    for b in range(len(slow)):
        mine = made[b * per_box : (b + 1) * per_box]
        if typical <= 0 or any(m < 0 for m in mine):  # nothing running yet (compiling), or an actor just restarted
            slow[b] = 0
            continue
        slow[b] = slow[b] + 1 if min(mine) < SLOW_SHARE * typical else 0
        if slow[b] >= SLOW_CHECKS and not asked[b]:
            out.append(b)
    return out


def _hardware() -> str:
    """The container's GPUs and CPU as nvidia-smi and lscpu name them: the same GPU type has run ~1.8x slower on some
    hosts (sp24 and sp26 against the benchmarks), so every run records what it got."""
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,power.limit,pcie.link.gen.current",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()  # fmt: skip
    cpu = next((line.split(":", 1)[1].strip() for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
                if line.startswith("Model name")), "?")  # fmt: skip
    return f"{' | '.join(dict.fromkeys(gpus))} x{len(gpus)}; {cpu}"


def _placement(environ) -> str:
    """Where Modal put this container or sandbox: its region and cloud (the tunnels' round trip depends on it)."""
    return f"{environ.get('MODAL_REGION', '?')} {environ.get('MODAL_CLOUD_PROVIDER', '?')}"


def _local():
    """The helpers that only exist on the machine starting a run (the container imports this module without them)."""
    sys.path.insert(0, str(ROOT / "docker"))
    from modal_run import IMAGE_ID, stream

    from .modal import _code, _send

    return IMAGE_ID, stream, _code, _send


def games_image(image_id: str = "") -> modal.Image:
    """The worker image (build/modal-image.txt's, or `image_id`) with this checkout's wc3env (the DLL included) on both
    sides: the bridges' Linux Python and the games' Windows Python, so the games run the env this code was written
    against."""
    site = "/home/worker/.wine/drive_c/Python311/Lib/site-packages/wc3env"
    return (
        modal.Image.from_id(image_id or _local()[0].read_text().strip())
        .run_commands(
            "apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*",
            "python3 -m pip install --break-system-packages --no-cache-dir numpy zstandard",
        )
        .add_local_dir(ROOT / "src" / "wc3env", "/opt/env/wc3env", copy=True, ignore=["**/__pycache__/**"])
        .run_commands(
            f"cp -r /opt/env/wc3env/. /opt/host/wc3env/ && cp -r /opt/env/wc3env/. {site}/ && chown -R 10001:10001 {site}"
        )
    )


def sandbox_memory(games: int) -> int:
    """MiB to request for a sandbox of `games` games. Sandbox memory is billed on the request, not on use, so ask for
    what they use: at most 0.32 GiB a game with the lowest video options (64 games played to the end against
    Warcraft's AI), plus a fifth for bigger self-play armies and 1 GiB for the bridges and Wine's services."""
    return int(games * 0.32 * 1.2 * 1024) + 1024


def games_sandbox(args, image: modal.Image, code: bytes, made: list | None = None, volumes: dict | None = None, gpu: str | None = None,
                  ports: list[int] | None = None) -> tuple[modal.Sandbox, str]:  # fmt: skip
    """A games sandbox with the licence and this checkout's code, its bridges not started yet; where it landed.
    It joins `made` as soon as it exists (so a failure later in the launch still stops it). ports: each one
    reachable from outside through a TCP tunnel of its own (box.tunnels())."""
    from wc3env.settings import settings

    _, stream, _, send = _local()
    box = modal.Sandbox.create(
        "sleep", "infinity", app=modal.App.lookup("wc3-selfplay", create_if_missing=True), image=image,
        cpu=args.cpu, memory=args.memory or sandbox_memory(args.games * args.per_box), timeout=min(24 * 3600, int((learner_hours(args.rest, 23) + 1) * 3600)),
        experimental_options={"vm_runtime": True}, volumes=volumes or {}, gpu=gpu,
        **({"encrypted_ports": ports or []} if getattr(args, "tls", False) else {"unencrypted_ports": ports or []}),
    )  # fmt: skip
    if made is not None:
        made.append(box)
    where = box.exec("bash", "-c", "echo ${MODAL_REGION:-?} ${MODAL_CLOUD_PROVIDER:-?}")
    where.wait()
    at = where.stdout.read().strip()
    stream(box, "mkdir -p /run/wc3-license /sessions/out /work && chown -R 10001:10001 /sessions /work", root=True)
    for license_file in ("roc.w3k", "tft.w3k"):
        send(box, f"/run/wc3-license/{license_file}", (settings().game_dir / license_file).read_bytes())
    send(box, "/work/code.tar", code)
    stream(box, "chmod 444 /run/wc3-license/* && cd /work && tar -xf code.tar && chown -R 10001:10001 /work", root=True)
    return box, at


def start_bridges(box: modal.Sandbox, addresses: list[str], args, listen: bool = False) -> None:
    """The sandbox's bridges, one per actor's tunnel address (listen: one per port here, the actors connecting in):
    its games start as the actors connect."""
    _, stream, _, send = _local()
    send(box, "/work/bridge.sh", bridge(addresses, args.games // args.group, listen=listen).encode())
    stream(box, "chown 10001:10001 /work/bridge.sh", root=True)
    # when the bridges end (the container is gone), so does the sandbox: its main process is `sleep infinity`
    box.exec(
        "bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/bridge.sh'; kill 1"
    )


def game_map(args) -> str:
    """The games' map under the game's folder: --map's, else the mode's (training/selfplay/modes)."""
    return f"Maps/FrozenThrone/{args.map}" if args.map else modes.get(args.mode).map


def install_map(box, name: str, data: bytes) -> None:
    """A map of build/maps (the flat arena, python -m tools.prepare.arena; the fast maps, tools.prepare.fastmap)
    where the games' Warcraft finds it, /opt/game/<name>."""
    _local()[3](box, "/tmp/map.w3x", data)
    _local()[1](box, f"install -m 644 /tmp/map.w3x '/opt/game/{name}'", root=True)


def start(args) -> None:
    """The GPU container (pinned to --region, broadly "us": Modal bills a broad region 1.15x, a narrow one like us-east
    1.75x; one landing outside --want, us-east by default, where the tunnel is shortest, is made again, and after TRIES
    it is pinned there),
    then the games sandboxes; one
    landing on another continent from the container is made again (up to 3 times: sp14's eu-west games stepped at
    half the speed of the US ones). The tunnel goes through Modal's relays on AWS near the container: us-east to
    us-east ~14 ms, us-central to us-east ~27-34, us-central to us-central ~50, eu-west to eu-west ~42
    (9-benchmarks.md), so following the sandboxes does not pay."""
    registry = modal.Dict.from_name(REGISTRY, create_if_missing=True)
    registry[args.name] = {"sandboxes": []}
    for key in [k for k in registry.keys() if k.startswith(f"{args.name}:")]:  # a slow sandbox's keys of a run before
        registry.pop(key)
    learner_argv = ["--init", f"/dataset/{args.init}", "--out", f"/dataset/runs/{args.name}", "--mode", args.mode,
                    *(["--start", f"/dataset/{args.start}"] if args.start else []),
                    *(["--evaluate", ",".join(f"/dataset/{c}" for c in args.evaluate)] if args.evaluate else []),
                    *args.rest]  # fmt: skip
    actor_argv = ["--games", str(args.games), "--ai-games", str(args.ai_games), "--group", str(args.group),
                  "--eval-games", str(args.eval_games),
                  "--plans", "/dataset/plans.npz", *(["--races", *args.races] if args.races else []),
                  "--mode", args.mode, *(["--map", game_map(args)] if args.map else [])]  # fmt: skip
    with modal.enable_output(), app.run(detach=True):
        options = {"gpu": args.gpu, "cpu": args.box_cpu, **({"region": args.region} if args.region else {})}
        for attempt in range(TRIES + 1):
            if attempt == TRIES and args.want:  # still elsewhere: pinned to where it should land (narrow: 1.75x)
                options["region"] = args.want
            call = container.with_options(**options).spawn(args.name, learner_argv, args.actors, actor_argv, args.inference,
                                                           args.per_box)  # fmt: skip
            registry[args.name] = {**registry[args.name], "call": call.object_id, "app": app.app_id}
            print(f"container {call.object_id}; waiting for where it landed", flush=True)
            while "container_at" not in registry.get(args.name, {}):
                time.sleep(5)
            at = registry[args.name].get("container_at", "?")
            if not args.want or at.startswith(args.want) or attempt == TRIES:
                break
            # a minute or two of a container (~$0.3) against a slower tunnel all run long: us-central to us-central
            # ~50 ms against us-east's ~14, ~20-30% fewer steps a second
            print(f"container landed in {at}, not {args.want}: made again", flush=True)
            call.cancel(terminate_containers=True)
            registry[args.name] = {"sandboxes": []}
    entry = registry[args.name]
    continent = entry.get("container_at", "?").split("-")[0]  # "us" of "us-east CLOUD_PROVIDER_GCP"
    image, code = games_image(), _local()[2]()
    # every map but the stock one is made here (build/maps: tools.prepare.arena, tools.prepare.fastmap) and sent along
    custom = (ROOT / "build" / "maps" / Path(game_map(args)).name).read_bytes() if game_map(args) != Games.map else b""
    count = -(-args.actors // args.per_box)
    every: list[modal.Sandbox] = []  # every sandbox made: all stopped if the launch fails (they would wait for nothing)

    def place(k: int) -> tuple[modal.Sandbox, str, list[str]]:
        return place_box(args, k, image, code, continent, custom, every)

    try:
        with ThreadPoolExecutor(min(count, 16)) as pool:
            made = list(pool.map(place, range(count)))
    except BaseException:
        for box in every:
            box.terminate()
        raise
    entry = registry[args.name]
    registry[args.name] = {**entry, "sandboxes": [b.object_id for b, _, _ in made], "sandboxes_at": [at for _, at, _ in made],
                           "games": [a for _, _, addresses in made for a in addresses]}  # the actors start now  # fmt: skip
    print(f"container in {entry.get('container_at', '?')}: {entry.get('hardware', '?')}", flush=True)
    for box, at, _ in made:
        print(f"games sandbox {box.object_id} in {at}", flush=True)
    print(f"read the log with: python -m training.cloud.selfplay log --name {args.name}")
    if args.watch:
        watch(args, image, code, continent, custom, [b.object_id for b, _, _ in made])


def place_box(
    args, k: int, image, code: bytes, continent: str, custom: bytes, every: list
) -> tuple[modal.Sandbox, str, list[str]]:
    """Games sandbox k with its bridges listening, on the container's continent (made again up to 3 times); where it
    landed and its tunnels' addresses, one per actor of it."""
    ports = [BRIDGE_PORT + j for j in range(min(args.per_box, args.actors - k * args.per_box))]
    box, at = games_sandbox(args, image, code, every, ports=ports)
    for _ in range(3):
        if continent == "?" or at.split("-")[0] == continent:
            break
        print(f"games sandbox {k} landed in {at}, the container in {continent}: made again", flush=True)
        box.terminate()
        box, at = games_sandbox(args, image, code, every, ports=ports)
    if custom:
        install_map(box, game_map(args), custom)
    start_bridges(box, [str(p) for p in ports], args, listen=True)
    tunnels = box.tunnels()
    if args.tls:  # TLS to the tunnel, which hands the bridge plain TCP (vector.dial_workers: tls://)
        return box, at, ["tls://%s:%d" % tunnels[p].tls_socket for p in ports]
    return box, at, ["%s:%d" % tunnels[p].tcp_socket for p in ports]


def watch(args, image, code: bytes, continent: str, custom: bytes, boxes: list[str]) -> None:
    """Until the run's sandboxes are gone (it ended, or stop): each one the container asks for again
    ("<name>:slow:<box>") made anew, its addresses published ("<name>:box:<box>"), the old one stopped."""
    registry = modal.Dict.from_name(REGISTRY)
    swaps = [0] * len(boxes)
    print(
        f"watching for slow sandboxes (ctrl-c leaves the run going; python -m training.cloud.selfplay stop --name "
        f"{args.name} ends it)",
        flush=True,
    )
    while any(_alive(box_id) for box_id in boxes):
        for b in range(len(boxes)):
            if registry.get(f"{args.name}:slow:{b}") is None:
                continue
            if swaps[b] == MAX_REPLACED:
                print(f"games sandbox {b} slow again: replaced {MAX_REPLACED} times already, kept", flush=True)
                registry.pop(f"{args.name}:slow:{b}")
                continue
            swaps[b] += 1
            made: list[modal.Sandbox] = []
            try:
                box, at, addresses = place_box(args, b, image, code, continent, custom, made)
            except Exception as exc:  # noqa: BLE001  the old one stays; asked again later
                print(
                    f"games sandbox {b}: a new one failed ({type(exc).__name__}: {exc}); the old one kept", flush=True
                )
                for box in made:
                    box.terminate()
                registry.pop(f"{args.name}:slow:{b}")
                continue
            entry = registry[args.name]
            registry[args.name] = {**entry, "sandboxes": [*entry.get("sandboxes", []), box.object_id]}  # stop's too
            registry[f"{args.name}:box:{b}"] = addresses
            registry.pop(f"{args.name}:slow:{b}")
            try:
                modal.Sandbox.from_id(boxes[b]).terminate()
            except Exception:  # noqa: BLE001  already gone
                pass
            print(f"games sandbox {b} replaced ({swaps[b]} of {MAX_REPLACED}): {box.object_id} in {at}", flush=True)
            boxes[b] = box.object_id
        time.sleep(20)
    print("the run's sandboxes are gone: done watching", flush=True)


def _alive(box_id: str) -> bool:
    try:
        return modal.Sandbox.from_id(box_id).poll() is None
    except Exception:  # noqa: BLE001  gone
        return False


def stop(args) -> None:
    """The run's sandboxes and app stopped, then checked: exits non-zero if the app or a sandbox is still alive."""
    entry = modal.Dict.from_name(REGISTRY).get(args.name, {})
    _stop_sandboxes(args.name)
    if "app" in entry:  # cancelling the call alone left its container running: stop the whole app
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", entry["app"], "--yes"], capture_output=True)
    listed = subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True, check=True
    )
    state = next((a["state"] for a in json.loads(listed.stdout) if a["app_id"] == entry.get("app")), "gone")
    alive = sum(_alive(box_id) for box_id in entry.get("sandboxes", []))
    if state not in ("stopped", "gone") or alive:
        raise SystemExit(f"app {entry.get('app')}: {state}; {alive} sandboxes still alive")
    print(f"stopped the container and {len(entry.get('sandboxes', []))} sandboxes")


def log(args) -> None:
    entry = modal.Dict.from_name(REGISTRY).get(args.name, {})
    print(
        f"placement: container {entry.get('container_at', '?')} ({entry.get('hardware', '?')}); sandboxes "
        f"{entry.get('sandboxes_at', '?')}"
    )
    sys.stdout.buffer.write(b"".join(VOLUME.read_file(f"runs/{args.name}/log.txt")))
    for entry in VOLUME.listdir(f"runs/{args.name}/actors"):  # the actors' and the inference process's last lines
        text = b"".join(VOLUME.read_file(entry.path)).decode(errors="replace").splitlines()
        print(f"== {entry.path}", *text[-3:], sep="\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("start", "log", "stop"))
    p.add_argument("--name", required=True)
    p.add_argument(
        "--init",
        default="runs/bc10/step8k.pt",
        help="the KL's anchor, inside the Volume (the start too, without --start)",
    )
    p.add_argument(
        "--start",
        default="",
        help="the weights to start from, inside the Volume (e.g. runs/cb9/last.pt): "
        "its Adam state comes along; the anchor stays --init",
    )
    p.add_argument("--actors", type=int, default=16)
    p.add_argument("--games", type=int, default=32, help="per actor")
    p.add_argument("--ai-games", type=int, default=32, help="per actor: against the league's ladder rung")
    p.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="per actor, of --games: only evaluate (the current policy "
        "and each --evaluate checkpoint in turn, at the learner's --eval-stage; not learned from)",
    )
    p.add_argument(
        "--evaluate",
        nargs="*",
        default=[],
        help="checkpoints inside the Volume the evaluation games play "
        "beside the current policy (e.g. runs/cb8/last.pt runs/cb9/last.pt)",
    )
    p.add_argument("--group", type=int, default=2, help="games per Wine worker")
    p.add_argument(
        "--races", nargs=2, metavar=("OWN", "OPPONENT"), help="one matchup in every game (default: each in turn)"
    )
    p.add_argument("--per-box", type=int, default=2, help="actors whose games one sandbox runs")
    p.add_argument("--cpu", type=float, default=4, help="a games sandbox's cores (billed on use, above this too)")
    p.add_argument(
        "--memory", type=int, default=0, help="a games sandbox's memory, MiB (0: sandbox_memory of its games)"
    )
    p.add_argument(
        "--box-cpu",
        type=float,
        default=24,
        help="the GPU container's cores (billed on the request: sp27 "
        "used ~16, 16 actors at ~0.75, 2 inference processes at ~1, the learner ~2.3)",
    )
    p.add_argument(
        "--inference",
        type=int,
        default=2,
        help="inference processes on the GPU (actors split among them): "
        "two decide 1.7x as fast as one (9-benchmarks.md)",
    )
    p.add_argument(
        "--gpu",
        default="H100:2",
        help="the GPU container's GPUs: with two, inference has one of its own (sp15: 1.7x the steps/s of one H100 for ~25%% more an hour); H100 for one",
    )
    p.add_argument(
        "--region",
        default="us",
        help="pin the GPU container there (Modal bills a broad region like us "
        "1.15x, a narrow one like us-east 1.75x); '' for anywhere",
    )
    p.add_argument(
        "--want",
        default="us-east",
        help="where the container should land (the shortest tunnel to the "
        "sandboxes, 9-benchmarks.md): one landing elsewhere is made again, pinned there after TRIES; '' to take any",
    )
    p.add_argument(
        "--mode",
        choices=modes.NAMES,
        default="game",
        help="whole games, 3x-speed games or fights on the "
        "flat arena (training/selfplay/modes; build fast's map first: python -m tools.prepare.fastmap, "
        "combat's: python -m tools.prepare.arena)",
    )
    p.add_argument(
        "--map",
        default="",
        help="the games' map instead of the mode's: a map of build/maps (its map data in src/wc3env/data/maps)",
    )
    p.add_argument(
        "--no-tls",
        dest="tls",
        action="store_false",
        help="the sandboxes' tunnels unencrypted, as before "
        "2026-10-12. By default encrypted (encrypted_ports, the actors connecting over TLS): Modal says "
        "unencrypted tunnels go through shared relays (some through the West Coast from us-east), encrypted "
        "ones to the sandbox directly; bench/tunnel_latency 2026-10-12, 20 min: unencrypted fell from 2717 "
        "round trips/s (p50 23 ms) to 59/s (p50 1.9 s) by minute 20, encrypted held 3160/s (p50 10 ms)",
    )
    p.add_argument(
        "--no-watch",
        dest="watch",
        action="store_false",
        help="leave after the launch instead of staying on to replace slow sandboxes",
    )
    p.add_argument("rest", nargs="*", help="learner flags, after --")
    args = p.parse_args()
    {"start": start, "stop": stop, "log": log}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
