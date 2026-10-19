"""Behavioral cloning (training/train/cloning.py) on a Modal H100, reading the data set from the Volume wc3-dataset.

    python -m training.cloud.train start --name bc1 [--steps 7200] [--data dataset3,bots1] [-- more cloning.py flags]
    python -m training.cloud.train start --name bc1w --steps 1000 -- --init runs/bc1/last.pt --only-wins --lr 1e-4
    python -m training.cloud.train log --name bc1          # the run's log.txt, from the Volume
    python -m training.cloud.train eval --name bc1         # held-out accuracy of the run's last.pt

The run writes /dataset/runs/<name>/ (log.txt, last.pt) and commits the Volume at every checkpoint, so the log
can be read while it trains; it keeps running after this command returns. A restarted container resumes from
last.pt.
"""

import argparse
import sys
from pathlib import Path

import modal

from ..bench.common import FLASH_ATTN_WHEEL
from ..train.track import modal_secrets

ROOT = Path(__file__).resolve().parents[2]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.8.0", "wandb", "wandb-workspaces")
    .run_commands(f"pip install {FLASH_ATTN_WHEEL}")  # packed attention (without it: padded)
    .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
    .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
)
app = modal.App("wc3-cloning", image=image)
VOLUME = modal.Volume.from_name("wc3-dataset")


@app.function(
    gpu="H100",
    cpu=16,
    memory=98304,
    timeout=24 * 3600,
    volumes={"/dataset": VOLUME},
    retries=modal.Retries(max_retries=2),
    secrets=modal_secrets(),
)
def cloning(argv: list[str]) -> None:
    from training.train.cloning import parser, train

    args = parser().parse_args(argv)
    if args.init and not args.init.startswith("/"):  # a run's checkpoint, named inside the Volume
        args.init = f"/dataset/{args.init}"
    args.on_save = VOLUME.commit
    train(args)
    VOLUME.commit()


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3600, volumes={"/dataset": VOLUME})
def evaluate_run(name: str, games: int, data: str) -> str:
    from pathlib import Path

    import torch

    from training.policy.model import Policy
    from training.train.cloning import evaluate, split

    device = torch.device("cuda")
    held = sorted((f for d in data.split(",") for f in split(Path(f"/dataset/{d}"))[1]), key=lambda f: f.name)[:games]
    model = Policy().to(device)
    saved = torch.load(f"/dataset/runs/{name}/last.pt", map_location=device)
    model.load_state_dict(saved["model"])
    acc = evaluate(model, held, 64, device)
    return f"{name} at step {saved['step']} on {len(held)} held-out files: " + " ".join(
        f"{k} {v:.3f}" for k, v in acc.items()
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("start", "log", "eval"))
    p.add_argument("--games", type=int, default=64, help="eval: held-out files")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=7200, help="optimizer steps (~2,400 per pass over the data)")
    p.add_argument(
        "--data", default="dataset", help="the Volume folders of data set files, comma-separated (dataset3,bots1)"
    )
    p.add_argument("rest", nargs="*", help="more training/train/cloning.py flags, after --")
    args = p.parse_args()
    if args.command == "log":
        data = b"".join(VOLUME.read_file(f"runs/{args.name}/log.txt"))
        sys.stdout.buffer.write(data)
        return 0
    if args.command == "eval":
        with modal.enable_output(), app.run():
            print(evaluate_run.remote(args.name, args.games, args.data))
        return 0
    argv = [
        *(f"/dataset/{d}" for d in args.data.split(",")),
        "--out",
        f"/dataset/runs/{args.name}",
        "--steps",
        str(args.steps),
        *args.rest,
    ]
    with modal.enable_output(), app.run(detach=True):
        call = cloning.spawn(argv)
        print(
            f"started {args.name}: {call.object_id}; read the log with: python -m training.cloud.train log --name {args.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
