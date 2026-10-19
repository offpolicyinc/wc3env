"""Training jobs on GCP VMs (on-demand, or --spot at about half the price), built from training/cloud/vm.py:

    python -m training.cloud.gcp clone --name bc10 --data gs://.../pro2/data,gs://.../bots3k/data --init runs/bc9/last.pt -- --steps 30000 --micro 4
    python -m training.cloud.gcp log --name bc10           # a clone's log, synced every minute
    python -m training.cloud.gcp stop --name g1            # every VM of the run
    python -m training.cloud.gcp eval --name e1 --games 8 -- gs://.../a.pt runs/b.pt   # one VM per checkpoint
    python -m training.cloud.gcp eval --name e1 --collect  # the table so far, and which checkpoints are missing
    python -m training.cloud.gcp record --name r1 --games 400 --vms 8   # Warcraft's AI against itself, kept as replays
    python -m training.cloud.gcp convert --name c1 --replays gs://.../replays-1.29 --vms 3   # replays to data again

A run's files live in gs://<project>-sessions/<kind>/<name>/ (kind: clone, eval, record, convert): this checkout's
code (training/ and src/wc3env, so the games run its DLL), the plan bank, and what the VMs write. A clone writes
<run>/out/: its log, saves, and exit.txt when it stops. Self-play runs on Modal (training/cloud/selfplay.py).

clone: one GPU VM trains training.train.cloning on --data (bucket folders of data set files, copied to the VM) from
--init; run it again with the same --name to carry on from the last save if the VM was lost. An L4 (24 GB) needs
-- --micro 4.

eval, record, convert: CPU VMs, each doing its share, uploading every two minutes and deleting itself.
"""

from __future__ import annotations

import argparse
import json
import sys

from .scripts import conversion, evaluation, recording
from .vm import cat, cpu_job, delete_all, gcloud, gpu_job, prepare, run_folder, upload

EVAL_HOURS = 2.5


def _bucket_file(path: str, url: str) -> str:
    """`path` (local or gs://) in the bucket: copied to `url` when local; its gs:// address."""
    if path.startswith("gs://"):
        return path
    gcloud("storage", "cp", path, url)
    return url


# ---- clone


def clone(args) -> None:
    """One GPU VM training a clone (see the module's docstring)."""
    run = prepare(args, "clone")
    folders = [d.strip().rstrip("/") for d in args.data.split(",") if d.strip()]
    inputs = [(f"{d}/", f"data/{k}") for k, d in enumerate(folders)]
    flags = list(args.rest)
    if args.init:
        inputs.append((_bucket_file(args.init, f"{run}/init.pt"), "init.pt"))
        flags = ["--init", "/work/init.pt", *flags]
    size_gb = sum(int(gcloud("storage", "du", "-s", f"{d}/", capture=True).split()[0]) for d in folders) / 1e9
    command = " ".join(["training.train.cloning", *(f"/work/data/{k}" for k in range(len(folders))), "--out", "/work/out",
                        *flags])  # fmt: skip
    name = f"wc3-{args.name}-clone"
    zone = gpu_job(args, run, name, command, inputs, args.hours, disk_gb=100 + int(size_gb * 1.3))
    if zone:
        print(f"{name} in {zone} on {args.gpu}: {len(folders)} data folders, {size_gb:.0f} GB; "
              f"log: python -m training.cloud.gcp log --name {args.name}")  # fmt: skip


# ---- self-play


# ---- CPU jobs: eval, record, convert


def evaluate(args) -> None:
    """One CPU VM per checkpoint, playing --games at once against the built-in AI; each uploads its eval.jsonl
    (one line per finished game) to <run>/out/<vm>/ and deletes itself. --collect prints the table so far."""
    from ..play.evaluate import summary

    run = run_folder(args, "eval")
    if args.collect:
        manifest = [line.split(" ", 1) for line in cat(f"{run}/checkpoints.txt").splitlines()]
        rows, missing = [], []
        for vm, checkpoint in manifest:
            lines = [json.loads(x) for x in cat(f"{run}/out/{vm}/eval.jsonl").splitlines() if x.strip()]
            rows += lines
            if not lines:
                missing.append(
                    f"{vm} ({checkpoint}): no games yet; its eval.err:\n{cat(f'{run}/out/{vm}/eval.err')[-800:]}"
                )
        print(summary(rows))
        print("\n".join(missing))
        return
    if cat(f"{run}/checkpoints.txt"):
        sys.exit(f"an evaluation named {args.name} exists; pick another --name")
    prepare(args, "eval")
    checkpoints = [_bucket_file(c, f"{run}/checkpoints/{k}.pt") for k, c in enumerate(args.rest)]
    vms = [f"wc3-{args.name}-eval-{k}" for k in range(len(checkpoints))]
    # the manifest first: VMs started before a failed one still get collected
    upload("\n".join(f"{vm} {c}" for vm, c in zip(vms, checkpoints)).encode(), f"{run}/checkpoints.txt")
    for vm, checkpoint in zip(vms, checkpoints):
        script = evaluation(checkpoint, args.games, args.minutes, args.difficulty, args.threads)
        cpu_job(args, run, vm, script, EVAL_HOURS, checkpoint=checkpoint)
        print(f"{vm}: {checkpoint}", flush=True)
    print(f"results: python -m training.cloud.gcp eval --name {args.name} --collect")


def record(args) -> None:
    """--vms CPU VMs, each recording its share of --games games of Warcraft's AI against itself (seeds in turn, every
    race pair) as training data into <run>/data/ and replays into <run>/replays/, then deleting itself."""
    run = prepare(args, "record")
    per = -(-args.games // args.vms)
    for k in range(args.vms):
        first, count = args.first_seed + k * per, min(per, args.games - k * per)
        if count <= 0:
            break
        name = f"wc3-{args.name}-record-{k}"
        # --workers games at once, each up to --minutes of game time, run about as fast as real time on a C2D core
        hours = count / max(args.workers, 1) * args.minutes / 60 + 1
        if cpu_job(args, run, name, recording(first, count, args.workers, args.minutes, args.maps), hours):
            print(f"{name}: seeds {first}..{first + count - 1}", flush=True)
        else:
            print(f"{name}: no room for seeds {first}..{first + count - 1}; record again later with --games {count} "
                  "on its own name, or another --actor-machine", flush=True)  # fmt: skip
    print(f"data: {run}/data/, replays: {run}/replays/")


def convert(args) -> None:
    """--vms CPU VMs, each writing the data set files of its share of the replays in --replays (a bucket folder of
    .w3g files: those its good.txt lists, else all) into <run>/data/, then deleting itself: the data again after
    a feature change, from the replays, without playing new games."""
    run = prepare(args, "convert")
    listing = cat(f"{args.replays}/good.txt")
    if listing:
        ids = [line.split()[0] for line in listing.splitlines() if line.strip()]
    else:
        names = gcloud("storage", "ls", f"{args.replays}/*.w3g", capture=True).split()
        ids = sorted(n.rsplit("/", 1)[1].removesuffix(".w3g") for n in names)
    per = -(-len(ids) // args.vms)
    only = {int(k) for k in args.only.split(",")} if args.only else set(range(args.vms))
    for k in sorted(only):  # --only: the shares still without a VM (the region's quota held the rest)
        share = ids[k * per : (k + 1) * per]
        if not share:
            break
        name = f"wc3-{args.name}-convert-{k}"
        # a replay converts at about 10x its game time; pro games run about 20 minutes, bot games up to --minutes
        hours = len(share) / max(args.workers, 1) * 0.1 + 1
        if cpu_job(args, run, name, conversion(share, args.workers), hours, inputs=args.replays):
            print(f"{name}: {len(share)} replays, {share[0]}..{share[-1]}", flush=True)
        else:
            print(f"{name}: no room for {len(share)} replays", flush=True)
    print(f"data: {run}/data/")


# ---- any run


def stop(args) -> None:
    print(f"deleted {delete_all(args)} VMs", flush=True)


def log(args) -> None:
    """A GPU job's log: the clone's or the self-play learner's."""
    for kind in ("clone", "selfplay"):
        out = f"{run_folder(args, kind)}/out"
        texts = {name: cat(f"{out}/{name}") for name in ("log.txt", "job.out", "gpu.txt", "exit.txt")}
        if any(texts.values()):
            for name, text in texts.items():
                if text:
                    print(f"== {name}\n{text[-6000:]}")
            return
    print(f"no log for {args.name} yet")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("clone", "log", "stop", "eval", "record", "convert"))
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="wc3env-pw")
    p.add_argument("--zone", default="us-central1-a", help="its region needs a Cloud NAT (see vm.prepare)")
    p.add_argument("--zones", default="", help="tried in order for new VMs (default: every zone in --zone's region)")
    p.add_argument("--spot", action="store_true", help="spot VMs, about half the price, which Google may take back "
                   "(spot A100s were taken twice in 30 minutes); default: on-demand")  # fmt: skip
    p.add_argument("--gpu", choices=("l4", "a100"), default="l4", help="clone: the GPU (l4 needs -- --micro 4)")
    p.add_argument("--init", default="", help="clone: the starting checkpoint, local or gs://")
    p.add_argument("--actor-machine", default="c2d-highcpu-8", help="CPU VMs (C2D: the CPU quota this project has)")
    p.add_argument("--threads", type=int, default=1, help="eval: torch threads")
    p.add_argument("--games", type=int, default=8, help="eval: per checkpoint, at once; record: in all")
    p.add_argument("--minutes", type=float, default=30, help="eval, record: each game's time limit, in game minutes")
    p.add_argument("rest", nargs="*", help="after --: cloning flags; eval: checkpoints, gs:// or local")
    cl = p.add_argument_group("clone")
    cl.add_argument("--data", default="", help="bucket folders of data set files, comma-separated (gs://...)")
    cl.add_argument("--hours", type=float, default=12, help="the VM's limit: it is deleted after this, done or not")
    ev = p.add_argument_group("eval")
    ev.add_argument("--difficulty", type=int, default=0, help="the built-in AI's: 0 easy, 1 normal, 2 insane")
    ev.add_argument("--collect", action="store_true", help="print the table so far")
    rec = p.add_argument_group("record, convert")
    rec.add_argument("--vms", type=int, default=4, help="CPU VMs recording or converting")
    rec.add_argument("--replays", help="convert: a bucket folder of .w3g replays (gs://...)")
    rec.add_argument("--workers", type=int, default=4, help="games at once per VM (one Wine prefix handles about four)")
    rec.add_argument(
        "--first-seed", type=int, default=0, help="the games' seeds start here (to record missing ones again)"
    )
    rec.add_argument(
        "--only",
        default="",
        help="convert: only these shares, comma-separated (those of a run that got "
        "no VM the first time; same --name and --vms)",
    )
    rec.add_argument("--maps", default="", help="map names under Maps/FrozenThrone, comma-separated, taken in turn "
                     "((2)EchoIsles,(4)TurtleRock,...; default: Echo Isles)")  # fmt: skip
    args = p.parse_args()
    if args.command == "clone" and not args.data:
        p.error("clone needs --data")
    commands = {"clone": clone, "stop": stop, "log": log, "eval": evaluate, "record": record, "convert": convert}
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
