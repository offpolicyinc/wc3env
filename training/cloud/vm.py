"""GCP building blocks for training/cloud/gcp.py: gcloud, the bucket, and VMs that run one job and delete
themselves. Two kinds of VM, each with a startup script:

    CPU_JOB  the worker image (Warcraft under Wine) running a script from training/cloud/scripts.py: actors, an
             evaluation, a recording, a conversion; logs and data go to the run's folder every two minutes
    GPU_JOB  a Deep Learning image (PyTorch, CUDA, the driver) running `python -m <command>` from this checkout's
             code with its inputs copied in: the self-play learner, a cloning run; /work/out goes to <run>/out
             every minute and an earlier VM's /work/out comes back first, so a new VM carries on from its saves

Every VM carries the label wc3-run=<name>, has no public address (it reaches the internet through the region's
Cloud NAT wc3-nat and Google's services through private access) and a maximum run duration, and deletes itself
when its job ends.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GCLOUD = shutil.which("gcloud") or "gcloud"
SETUP_HOURS = 1.5  # a VM's start before its job: packages, the image, the inputs
GPUS = {  # machine type and its creation flags (an A100 machine comes with its GPU)
    "a100": ("a2-ultragpu-1g", []),  # 80 GB
    "l4": ("g2-standard-8", ["--accelerator=type=nvidia-l4,count=1"]),  # 24 GB: --micro for the learner and cloning
}

# both kinds: metadata values, the VM's name and zone
COMMON = """#!/bin/bash
set -u
meta() {{ curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }}
NAME=$(meta name); ZONE=$(meta zone | cut -d/ -f4); RUN=$(meta attributes/wc3-run-path)
"""
DELETE_SELF = 'gcloud compute instances delete "$NAME" --zone "$ZONE" -q\n'

GPU_JOB = (
    COMMON
    + """INPUTS=$(meta attributes/wc3-inputs); COMMAND=$(meta attributes/wc3-command)
mkdir -p /work/out && cd /work
gcloud storage cp "$RUN/code.tar" . && tar -xf code.tar
gcloud storage rsync -r "$RUN/out" /work/out > /dev/null 2>&1 || true   # an earlier VM of this job: its saves
rm -f /work/out/exit.txt
# one "gs://source target" per line: a folder (ending in /) or a file, to /work/<target>
echo "$INPUTS" | while read -r source target; do
    [ -z "$source" ] && continue
    case "$source" in
        */) mkdir -p "/work/$target" && gcloud storage rsync -r "$source" "/work/$target" > /dev/null ;;
        *) mkdir -p "$(dirname "/work/$target")" && gcloud storage cp "$source" "/work/$target" > /dev/null ;;
    esac || echo "could not fetch $source" >> /work/out/job.out
done
PY=/opt/conda/bin/python; [ -x "$PY" ] || PY=$(command -v python3)
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'" > /work/out/gpu.txt 2>&1
(while sleep 60; do gcloud storage rsync -r /work/out "$RUN/out" > /dev/null 2>&1; done) &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/work:/work/src $PY -m $COMMAND >> /work/out/job.out 2>&1
echo $? > /work/out/exit.txt   # 0 done; the learner: 3 it stalled; else it crashed
gcloud storage rsync -r /work/out "$RUN/out"
"""
    + DELETE_SELF
)

CPU_JOB = (
    COMMON
    + """IMAGE=$(meta attributes/wc3-image); CHECKPOINT=$(meta attributes/wc3-checkpoint); INPUTS=$(meta attributes/wc3-inputs)
apt-get update -q && apt-get install -yq docker.io python3-pip
gcloud auth configure-docker "${{IMAGE%%/*}}" -q
docker pull "$IMAGE" & PULL=$!
mkdir -p /run/wc3-license /opt/work /sessions/out && chown -R 10001:10001 /sessions
for key in roc tft; do gcloud secrets versions access latest --secret "wc3-$key" > "/run/wc3-license/$key.w3k"; done
chmod 444 /run/wc3-license/*
# CPU PyTorch for the host's Python 3.12, which the image's Linux Python matches
python3 -m pip install -q --target /opt/pylib torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
cd /opt/work && gcloud storage cp "$RUN/code.tar" "$RUN/plans.npz" . && tar -xf code.tar
if [ -n "$CHECKPOINT" ]; then gcloud storage cp "$CHECKPOINT" /opt/work/checkpoint.pt; fi
if [ -n "$INPUTS" ]; then gcloud storage rsync -r "$INPUTS" /opt/work/inputs > /dev/null; fi
meta attributes/wc3-script > /opt/work/script.sh
chown -R 10001:10001 /opt/work
upload() {{ shopt -s nullglob; files=(/sessions/*.log /sessions/out/*.log /sessions/out/eval.*); [ ${{#files[@]}} -gt 0 ] && gcloud storage cp "${{files[@]}}" "$RUN/out/$NAME/" > /dev/null 2>&1; }}
data() {{ for d in data replays; do [ -d /sessions/out/$d ] && gcloud storage rsync /sessions/out/$d "$RUN/$d" -x '.*\\.jsonl$' > /dev/null 2>&1; done; true; }}
replays() {{ shopt -s nullglob; for r in /sessions/out/game-*/*/Replay/TempReplay.w3g; do g=${{r#/sessions/out/}}; gcloud storage cp "$r" "$RUN/out/$NAME/replays/${{g%%/*}}.w3g" > /dev/null 2>&1; done; }}
(while sleep 120; do upload; data; done) &
wait $PULL
docker run --rm --init --network host --shm-size 1g --user root --entrypoint bash \\
    --mount type=bind,src=/run/wc3-license,dst=/run/wc3-license,readonly \\
    --mount type=bind,src=/opt/work,dst=/work --mount type=bind,src=/opt/pylib,dst=/opt/pylib,readonly \\
    --mount type=bind,src=/sessions,dst=/sessions "$IMAGE" -c \\
    "{overlay} && su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/script.sh'" \\
    > /sessions/worker.log 2>&1
upload; data
if [ -n "$CHECKPOINT" ]; then replays; fi   # an evaluation's games, to watch
"""
    + DELETE_SELF
)


# ---- gcloud and the bucket


def gcloud(*args: str, check: bool = True, capture: bool = False) -> str:
    result = subprocess.run([GCLOUD, *args], check=check, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def cat(url: str) -> str:
    """A bucket file's text, or "" when there is none."""
    return gcloud("storage", "cat", url, check=False, capture=True)


def upload(data: bytes, url: str) -> None:
    subprocess.run([GCLOUD, "storage", "cp", "-", url], input=data, check=True)


def code() -> bytes:
    """training/ and src/wc3env (sources, data and the built DLLs), and wc3hook's C sources, as a tar archive: the
    features DLL is rebuilt when a source is newer, so its sources must be there (without them every replay of
    pro3's first try failed on the missing wc3hook/features.c)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for top, kinds in ((ROOT / "training", (".py", ".json", ".dll", ".npz")), (ROOT / "src" / "wc3env", (".py", ".json", ".dll", ".npz")),
                           (ROOT / "wc3hook", (".c", ".h"))):  # fmt: skip
            for path in sorted(top.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix in kinds:
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
    return buffer.getvalue()


def run_folder(args, kind: str) -> str:
    """The run's bucket folder: gs://<project>-sessions/<kind>/<name>."""
    return f"gs://{args.project}-sessions/{kind}/{args.name}"


def worker_image(args) -> str:
    return f"us-central1-docker.pkg.dev/{args.project}/wc3/worker:latest"  # VMs in other regions pull it from there


def prepare(args, kind: str) -> str:
    """What every launch needs first: the region's NAT, and this checkout's code and the plan bank in the run's
    folder; the folder."""
    region = args.zone.rsplit("-", 1)[0]
    nat = ("compute", "routers", "nats", "describe", "wc3-nat", "--router=wc3-router", f"--region={region}",
           f"--project={args.project}")  # fmt: skip
    if subprocess.run([GCLOUD, *nat], capture_output=True).returncode != 0:
        sys.exit(f"no Cloud NAT in {region}; once per region:\n"
                 f"  gcloud compute routers create wc3-router --region={region} --network=default\n"
                 f"  gcloud compute routers nats create wc3-nat --router=wc3-router --region={region} "
                 "--auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges\n"
                 f"  gcloud compute networks subnets update default --region={region} --enable-private-ip-google-access")  # fmt: skip
    run = run_folder(args, kind)
    upload(code(), f"{run}/code.tar")
    gcloud("storage", "cp", str(ROOT / "runs" / "plans.npz"), f"{run}/plans.npz")
    return run


# ---- VMs


def create(args, zone: str, name: str, machine: str, startup: str, metadata: dict, hours: float,
           gpu: str = "", disk_gb: int = 0) -> None:  # fmt: skip
    """One VM: `gpu` a GPUS key for a Deep Learning image (else Ubuntu), deleted after `hours` at the latest."""
    script = ROOT / "build" / f"gcp-{name}.sh"
    script.parent.mkdir(exist_ok=True)
    script.write_text(startup, newline="\n")
    files = [f"startup-script={script}"]
    for key, value in metadata.items():  # values from files: scripts and flags keep their characters
        path = script.with_suffix(f".{key}")
        path.write_text(value, newline="\n")
        files.append(f"{key}={path}")
    if gpu:
        image = ["--image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580", "--image-project=deeplearning-platform-release",
                 *GPUS[gpu][1], "--maintenance-policy=TERMINATE", f"--boot-disk-size={disk_gb or 100}GB"]  # fmt: skip
    else:
        image = ["--image-family=ubuntu-2404-lts-amd64", "--image-project=ubuntu-os-cloud", f"--boot-disk-size={disk_gb or 60}GB"]
    gcloud("compute", "instances", "create", name, f"--project={args.project}", f"--zone={zone}",
           f"--machine-type={machine}", f"--provisioning-model={'SPOT' if args.spot else 'STANDARD'}",
           "--instance-termination-action=DELETE", f"--max-run-duration={int(hours * 3600)}s", *image, "--no-address",
           "--scopes=cloud-platform", f"--labels=wc3-run={args.name}", f"--metadata-from-file={','.join(files)}")  # fmt: skip


def create_anywhere(args, name: str, *rest, first: str = "", **kwargs) -> str:
    """create() in the first zone of --zones (default: every zone of --zone's region; `first` before the others)
    with room; the zone it got, or "". A create can report a failure yet leave the VM running: it counts as made."""
    region = args.zone.rsplit("-", 1)[0]
    zones = args.zones.split(",") if args.zones else gcloud(
        "compute", "zones", "list", f"--filter=region:{region}", "--format=value(name)", f"--project={args.project}",
        capture=True).split()  # fmt: skip
    for zone in sorted(zones, key=lambda z: z != first):
        try:
            create(args, zone, name, *rest, **kwargs)
            return zone
        except subprocess.CalledProcessError:
            if running(args).get(name) == zone:
                return zone
            print(f"{name}: no VM in {zone}", flush=True)
    print(f"{name}: no zone had room", flush=True)
    return ""


def running(args) -> dict[str, str]:
    """The run's VMs that exist: name -> zone."""
    rows = gcloud("compute", "instances", "list", f"--project={args.project}", f"--filter=labels.wc3-run={args.name}",
                  "--format=value(name,zone.basename())", capture=True).splitlines()  # fmt: skip
    return dict(row.split() for row in rows)


def delete_all(args) -> int:
    """Deletes the run's VMs; how many there were."""
    vms = running(args)
    for zone in set(vms.values()):  # one call per zone deletes its VMs together (one at a time took 15 minutes)
        names = [name for name, z in vms.items() if z == zone]
        gcloud("compute", "instances", "delete", *names, f"--project={args.project}", f"--zone={zone}", "--quiet",
               check=False)  # fmt: skip
    return len(vms)


def gpu_job(args, run: str, name: str, command: str, inputs: list[tuple[str, str]], hours: float,
            disk_gb: int = 0, metadata: dict | None = None) -> str:  # fmt: skip
    """A GPU_JOB VM on --gpu for about `hours`: `python -m <command>` with `inputs` ((gs:// source, target under
    /work)) copied in; the zone it got, or ""."""
    meta = {"wc3-run-path": run, "wc3-command": command, "wc3-inputs": "\n".join(f"{s} {t}" for s, t in inputs),
            **(metadata or {})}  # fmt: skip
    return create_anywhere(args, name, GPUS[args.gpu][0], GPU_JOB.format(), meta, hours + SETUP_HOURS,
                           gpu=args.gpu, disk_gb=disk_gb)  # fmt: skip


def cpu_job(args, run: str, name: str, script: str, hours: float, first: str = "", checkpoint: str = "",
            inputs: str = "") -> str:  # fmt: skip
    """A CPU_JOB VM on --actor-machine running `script` (training/cloud/scripts.py) in the worker image for about
    `hours`; the zone it got, or ""."""
    from .scripts import overlay_env

    meta = {"wc3-run-path": run, "wc3-image": worker_image(args), "wc3-script": script, "wc3-checkpoint": checkpoint,
            "wc3-inputs": inputs}  # fmt: skip
    return create_anywhere(args, name, args.actor_machine, CPU_JOB.format(overlay=overlay_env()), meta,
                           hours + SETUP_HOURS, first=first)  # fmt: skip
