"""A checkpoint's games against Warcraft's AI (training/play/evaluate.py) in one Modal games sandbox, the model on
the sandbox's CPU (98% of a round: 533 ms of 544 at 20 games): no tunnel. --gpu fails: Modal has no GPUs with
the VM runtime Wine needs. Prints each game's record as it ends, progress (with how much of a round of
steps the model takes) every 200 rounds, and the table at the end.

    python -m training.bench.modal_eval runs/sp34/last.pt --label sp34 --games 20 --races orc orc --minutes 15
"""

import argparse
import sys
import types
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.cloud.scripts import _preamble, low_video  # noqa: E402
from training.cloud.selfplay import _local, games_image, games_sandbox  # noqa: E402

TORCH = "python3 -m pip install --break-system-packages --no-cache-dir torch"
TORCH_CPU = TORCH + " --index-url https://download.pytorch.org/whl/cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="its path on the wc3-dataset volume")
    p.add_argument("--label", default="")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--races", nargs=2, default=("orc", "orc"))
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--opponent", default="ai")
    p.add_argument("--cpu", type=float, default=8)
    p.add_argument("--threads", type=int, default=4, help="the model's torch threads")
    p.add_argument("--gpu", default="", help='the sandbox\'s GPU ("": none, the model on its CPU)')
    a = p.parse_args()
    _, stream, _code, send = _local()
    args = types.SimpleNamespace(cpu=a.cpu, memory=16384, games=a.games, per_box=1, rest=["--hours", "1"])
    made: list = []
    try:
        box, at = games_sandbox(args, games_image().run_commands(TORCH if a.gpu else TORCH_CPU), _code(), made,
                                {"/dataset": modal.Volume.from_name("wc3-dataset")}, a.gpu or None)  # fmt: skip
        print("sandbox in", at, flush=True)
        stream(
            box,
            f"mkdir -p /work/runs && cp /dataset/{a.checkpoint} /work/checkpoint.pt && chown -R 10001:10001 /work",
            root=True,
        )
        send(box, "/work/runs/plans.npz", (ROOT / "runs" / "plans.npz").read_bytes())
        script = (
            _preamble("/work:/opt/host", "/sessions/out")
            + low_video()
            + (
                f"python3 -m training.play.evaluate /work/checkpoint.pt --label {a.label or a.checkpoint} --games {a.games} "
                f"--races {a.races[0]} {a.races[1]} --minutes {a.minutes} --difficulty {a.difficulty} --opponent {a.opponent} "
                f"--threads {a.threads} --device {'cuda' if a.gpu else 'cpu'} --wine --plans /work/runs/plans.npz | tee /sessions/out/eval.jsonl\n"
                "python3 -m training.play.evaluate --table /sessions/out/eval.jsonl\n"
            )
        )
        send(box, "/work/eval.sh", script.encode())
        stream(box, "chown 10001:10001 /work/eval.sh /work/runs/plans.npz", root=True)
        stream(
            box,
            "su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/eval.sh' 2>&1 | grep --line-buffered -v -E '^(0[0-9a-f]+:|ERROR: +Vk)'",
            root=True,
        )
    finally:
        for box in made:
            box.terminate()


if __name__ == "__main__":
    main()
