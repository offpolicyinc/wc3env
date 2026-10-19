"""Write the data set on Modal (the worker image of docker/modal_run.py, in the peter-nico workspace).

    python -m training.cloud.modal upload                 # the good replays into the Volume wc3-replays
    python -m training.cloud.modal check --shards 40      # training.replays.check on every replay
    python -m training.cloud.modal run --shards 40        # sandboxes write their share into wc3-dataset
    python -m training.cloud.modal bots --shards 20 --games 40   # games of Warcraft's AI against itself, checked
    python -m training.cloud.modal verify                 # training.replays.check_dataset on the Volume's files
    python -m training.cloud.modal status                 # files and index lines in wc3-dataset

Each sandbox gets its share of replay ids, the training code and the activation files (streamed in, never in
an image, as the Docker README requires), runs `python -m training.replays.dataset` under Wine with two games at a
time (one Wine prefix scales to about four), and copies what it wrote into the Volume. Replays already in the
Volume's data set are skipped, so a run can be repeated after a failure.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
REPLAYS = ROOT / "runs" / "replays" / "1.29"
sys.path.insert(0, str(ROOT / "docker"))
from modal_run import ENTRYPOINT, IMAGE_ID, stream  # noqa: E402

REPLAY_VOLUME, DATA_VOLUME = "wc3-replays", "wc3-dataset"


def upload() -> None:
    ids = [line.split()[0] for line in (REPLAYS / "good.txt").read_text().splitlines() if line.strip()]
    volume = modal.Volume.from_name(REPLAY_VOLUME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for i in ids:
            batch.put_file(str(REPLAYS / f"{i}.w3g"), f"/{i}.w3g")
        batch.put_file(str(REPLAYS / "good.txt"), "/good.txt")
    print(f"{len(ids)} replays in the Volume {REPLAY_VOLUME}")


def _code() -> bytes:
    """The training package as a tar archive (its sources and the order table)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted((ROOT / "training").rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    return buffer.getvalue()


def _send(box: modal.Sandbox, target: str, data: bytes) -> None:
    process = box.exec("bash", "-c", f"cat > {target}", text=False)  # contents stay off the command line
    for k in range(0, len(data), 1 << 20):  # in pieces: the stream refuses big writes before a drain
        process.stdin.write(data[k : k + (1 << 20)])
        process.stdin.drain()
    process.stdin.write_eof()
    process.stdin.drain()
    if process.wait() != 0:
        raise RuntimeError(f"could not write {target}")


def shard(
    number: int, ids: list[str], image: modal.Image, code: bytes, cpu: float, memory: int, job: str, folder: str = ""
) -> str:
    from wc3env.settings import settings

    volumes = {"/replays": modal.Volume.from_name(REPLAY_VOLUME), "/dataset": modal.Volume.from_name(DATA_VOLUME)}
    box = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=modal.App.lookup("wc3-dataset", create_if_missing=True),
        image=image,
        cpu=cpu,
        memory=memory,
        timeout=6 * 3600,
        volumes=volumes,
        experimental_options={"vm_runtime": True},
    )
    try:
        stream(box, "mkdir -p /run/wc3-license /sessions/out /work && chown -R 10001:10001 /sessions /work", root=True)
        for name in ("roc.w3k", "tft.w3k"):
            _send(box, f"/run/wc3-license/{name}", (settings().game_dir / name).read_bytes())
        _send(box, "/work/code.tar", code)
        _send(box, "/work/ids.txt", "\n".join(ids).encode() + b"\n")
        stream(
            box, "chmod 444 /run/wc3-license/* && cd /work && tar -xf code.tar && chown -R 10001:10001 /work", root=True
        )
        if job == "bots":
            first, count = ids[0].split(":")
            command = (
                rf"python -m training.replays.check 'Z:\replays' --bots {count} --first-seed {first} --workers 2 "
                rf"--out 'Z:\sessions\out\bots-{number:03d}.jsonl'"
            )
        elif job == "dataset":
            command = (
                rf"python -m training.replays.dataset 'Z:\replays' --out 'Z:\sessions\out' --workers 2 "
                rf"--ids 'Z:\work\ids.txt' --index index-{number:03d}.jsonl"
            )
        else:
            command = (
                rf"python -m training.replays.check 'Z:\replays' --workers 2 --ids 'Z:\work\ids.txt' "
                rf"--out 'Z:\sessions\out\check-{number:03d}.jsonl'"
            )
        code_run = stream(box, f"{ENTRYPOINT} {command}")
        folder = folder or job
        stream(box, f"mkdir -p /dataset/{folder} && cp /sessions/out/* /dataset/{folder}/ && sync", root=True)
        return f"shard {number}: exit {code_run}"
    finally:
        box.terminate()


def run_bots(shards: int, games: int, cpu: float, memory: int) -> None:
    """`games` games of Warcraft's AI against itself (seeds 0..games-1, all race pairs in turn), checked."""
    image = modal.Image.from_id(IMAGE_ID.read_text().strip())
    code = _code()
    per = -(-games // shards)
    parts = [[f"{k * per}:{min(per, games - k * per)}"] for k in range(shards) if k * per < games]
    with ThreadPoolExecutor(len(parts)) as pool:
        for line in pool.map(lambda k: shard(k, parts[k], image, code, cpu, memory, "bots"), range(len(parts))):
            print(line, flush=True)


def verify() -> None:
    from .verify import app, check_files

    with modal.enable_output(), app.run():
        print(check_files.remote())


def run(shards: int, cpu: float, memory: int, job: str, limit: int = 0, folder: str = "") -> None:
    ids = [line.split()[0] for line in (REPLAYS / "good.txt").read_text().splitlines() if line.strip()]
    volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    try:
        done = {
            e.path.rsplit("/", 1)[-1].split("_")[0]
            for e in volume.listdir(f"/{folder or job}")
            if e.path.endswith(".npz")
        }
    except Exception:  # noqa: BLE001 (the folder is not there yet)
        done = set()
    ids = [i for i in ids if i not in done][: limit or None]
    print(f"{job}: {len(ids)} replays in {shards} sandboxes", flush=True)
    image = modal.Image.from_id(IMAGE_ID.read_text().strip())
    code = _code()
    parts = [ids[k::shards] for k in range(shards)]
    with ThreadPoolExecutor(shards) as pool:
        for line in pool.map(lambda k: shard(k, parts[k], image, code, cpu, memory, job, folder), range(shards)):
            print(line, flush=True)


def status() -> None:
    entries = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True).listdir("/", recursive=True)
    files = [e for e in entries if e.path.endswith(".npz")]
    print(f"{len(files)} files, {sum(e.size for e in files) / 1e9:.2f} GB in {DATA_VOLUME}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("upload", "check", "run", "bots", "verify", "status"))
    parser.add_argument("--games", type=int, default=40, help="bots: how many games")
    parser.add_argument("--shards", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="only this many replays (a trial)")
    parser.add_argument("--cpu", type=float, default=4)
    parser.add_argument("--memory", type=int, default=8192, help="MiB")
    parser.add_argument("--folder", default="", help="run: the Volume folder to write (default: the job's name)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    jobs = {
        "upload": upload,
        "status": status,
        "verify": verify,
        "bots": lambda: run_bots(args.shards, args.games, args.cpu, args.memory),
        "check": lambda: run(args.shards, args.cpu, args.memory, "check", args.limit),
        "run": lambda: run(args.shards, args.cpu, args.memory, "dataset", args.limit, args.folder),
    }
    jobs[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
