"""A checkpoint's games against Warcraft's AI, handicapped or not (training/play/evaluate.py) on Modal: the model on
one small GPU container, the games in Wine sandboxes (Modal has no GPUs with the VM runtime Wine needs), joined by
one tunnel, as in self-play (docs/training/machines.md).

    python -m training.cloud.evaluate runs/sp34/last.pt --label sp34 --games 20 --races orc orc --minutes 15
    python -m training.cloud.evaluate runs/bc11/last.pt --label bc11-hc20 --games 32 --handicap 0.2
    python -m training.cloud.evaluate runs/bc11/last.pt --label bc11-plan --games 16 --empty-half   # the plan used?
    python -m training.cloud.evaluate runs/fx1/last.pt --label fx1 --games 16 --map "(2)EchoIsles_x3_s20n.w3x"  # fast

Each game's record prints as it ends, progress every 200 rounds (with the model's share of a round), the table at
the end; the records also go to the Volume, runs/evals/<label>.jsonl, and every game's native replay to
build/evals/<label>/ here (box-<k>/game-<i>.w3g: the record's "replay" names it).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
import time
import types
from pathlib import Path

import modal
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOLUME = modal.Volume.from_name("wc3-dataset")
REGISTRY = "wc3-eval"  # Modal Dict: label -> the games' tunnel address
PORT = 7700
GAMES_A_BOX = 64  # as in self-play
CORES_A_GAME = 0.16  # self-play's games: ~9-11 cores a sandbox of 64 (docs/training/9-benchmarks.md)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.8.0", "zstandard")
    .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
    .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
)
app = modal.App("wc3-eval", image=image)


# pinned to us-east, as self-play's container (training/cloud/selfplay.py): unpinned, a game's step took ~120 ms there
# and back (1.4 s with the sandbox in the UK) against ~15 from us-east to us-east (9-benchmarks.md); its own step 4 ms
@app.function(gpu="L4", cpu=4, memory=16384, timeout=4 * 3600, volumes={"/dataset": VOLUME}, region="us-east")
def play(label: str, checkpoint: str, options: dict) -> list[dict]:
    """The model's side: listens for the games' workers through a tunnel (its address in REGISTRY), plays them."""
    sys.path.insert(0, "/root")  # the image's copy of training/, importable only once this is on the path
    from training.play.evaluate import evaluate, following, summary

    rows = []
    with modal.forward(PORT, unencrypted=True) as tunnel:
        modal.Dict.from_name(REGISTRY, create_if_missing=True)[label] = "%s:%d" % tunnel.tcp_socket
        print(f"games come in at {tunnel.tcp_socket}", flush=True)
        for row in evaluate(f"/dataset/{checkpoint}", options["games"], options["difficulty"], options["minutes"], 1.0,
                            False, "/dataset/plans.npz", f"Maps/FrozenThrone/{options.get('map') or '(2)EchoIsles.w3x'}", label=label,
                            opponent="ai", pair=options["races"], device="cuda",
                            remote=PORT, replays=r"Z:\sessions\out\replays", empty_half=options["empty_half"], handicap=options["handicap"],
                            record=options.get("record", False)):  # fmt: skip
            data = row.pop("data", None)
            if (
                data is not None
            ):  # each step's decision, the critic's values and the reward: runs/evals/<label>/game-<k>.npz
                (Path("/dataset/runs/evals") / label).mkdir(parents=True, exist_ok=True)
                keep = ("type", "argument", "game_time", "value", "reward", "done", "log_prob")
                np.savez_compressed(f"/dataset/runs/evals/{label}/game-{len(rows)}.npz", **{k: data[k] for k in keep})
            rows.append(row)
            print(json.dumps(row), file=sys.__stdout__, flush=True)  # (evaluate silences stdout while games run)
    print(summary(rows), flush=True)
    print(following(rows), flush=True)
    out = Path("/dataset/runs/evals")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    VOLUME.commit()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="its path on the wc3-dataset Volume")
    p.add_argument("--label", default="")
    p.add_argument("--games", type=int, default=20)
    p.add_argument(
        "--boxes", type=int, default=0, help="games sandboxes, the games split evenly (0: one per GAMES_A_BOX)"
    )
    p.add_argument("--cpu", type=float, default=0, help="each sandbox's cores (0: from its games, CORES_A_GAME)")
    p.add_argument("--races", nargs=2, default=("orc", "orc"))
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--handicap", type=float, default=1.0, help="player 1's hit points as a share of their own")
    p.add_argument(
        "--map", default="", help="a map of build/maps instead of Echo Isles (python -m tools.prepare.fastmap)"
    )
    p.add_argument(
        "--empty-half", action="store_true", help="every other game with the empty plan, to see the plan used"
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="each game's steps (decisions, critic values, rewards as "
        "self-play pays them) to runs/evals/<label>/game-<k>.npz on the Volume",
    )
    a = p.parse_args()
    a.boxes = a.boxes or -(-a.games // GAMES_A_BOX)
    if a.games % (2 * a.boxes):
        sys.exit("--games must split into pairs evenly over --boxes (two games a Wine worker)")
    sys.path.insert(0, str(ROOT))  # here, not at the top: the container imports this file without them
    from training.cloud.scripts import bridge
    from training.cloud.selfplay import _local, games_image, games_sandbox, install_map

    label = a.label or f"{a.checkpoint.replace('/', '-')}-{int(time.time())}"
    registry = modal.Dict.from_name(REGISTRY, create_if_missing=True)
    registry.pop(label, None)
    options = {"games": a.games, "difficulty": a.difficulty, "minutes": a.minutes,
               "races": list(a.races), "empty_half": a.empty_half, "handicap": a.handicap, "record": a.record,
               "map": a.map}  # fmt: skip
    _, stream, code, send = _local()
    per_box = a.games // a.boxes
    a.cpu = a.cpu or max(2.0, math.ceil(per_box * CORES_A_GAME) + 1.0)  # +1: Wine's services and the bridge
    print(f"{a.boxes} sandbox(es), {per_box} games and {a.cpu:g} cores each", flush=True)
    args = types.SimpleNamespace(cpu=a.cpu, memory=0, games=per_box, per_box=1, rest=["--hours", "3"])
    boxes: list = []
    with modal.enable_output(), app.run():
        call = play.spawn(label, a.checkpoint, options)
        while label not in registry:
            time.sleep(3)
        address = registry[label]
        try:
            for k in range(a.boxes):
                box, at = games_sandbox(args, games_image(), code(), boxes)
                for _ in range(3):  # one on another continent from the container is made again, as in self-play
                    if at.startswith("us"):
                        break
                    print(f"games sandbox {k} landed in {at}: made again", flush=True)
                    boxes.remove(box)
                    box.terminate()
                    box, at = games_sandbox(args, games_image(), code(), boxes)
                print(f"games sandbox {k} in {at}", flush=True)
                if a.map:
                    install_map(box, f"Maps/FrozenThrone/{a.map}", (ROOT / "build" / "maps" / a.map).read_bytes())
                send(box, "/work/bridge.sh", bridge([address], per_box // 2).encode())
                stream(
                    box,
                    "mkdir -p /sessions/out/replays && chown -R 10001:10001 /work/bridge.sh /sessions/out",
                    root=True,
                )
                box.exec(
                    "bash", "-c", "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/bridge.sh'"
                )
            call.get()
            for k, box in enumerate(boxes):  # the replays, before the sandboxes go
                folder = ROOT / "build" / "evals" / label / f"box-{k}"
                folder.mkdir(parents=True, exist_ok=True)
                tar = box.exec("bash", "-c", "cd /sessions/out/replays && tar -cf - .", text=False)
                with tarfile.open(fileobj=io.BytesIO(tar.stdout.read()), mode="r:") as archive:
                    archive.extractall(folder, filter="data")
                print(f"replays: {folder} ({len(list(folder.glob('*.w3g')))})", flush=True)
        finally:
            for box in boxes:
                box.terminate()
            registry.pop(label, None)


if __name__ == "__main__":
    main()
