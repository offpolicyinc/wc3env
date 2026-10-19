"""Self-play on one machine: the learner here, the inference process and the actors as child processes, all
connecting over localhost (the same layout as on Modal: training/cloud/selfplay.py).

    python -m training.selfplay.run --init runs/cloning/bc6w/last.pt --out runs/selfplay/local --actors 2 --games 4

Arguments after `--` go to the learner (training/selfplay/learner.py), e.g. `-- --batch 8 --warmup 5`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import learner


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--actors", type=int, default=1)
    p.add_argument("--games", type=int, default=4, help="per actor")
    p.add_argument("--ai-games", type=int, default=1, help="per actor")
    p.add_argument("--eval-games", type=int, default=0, help="per actor, of --games: only evaluate (learner's --evaluate)")
    p.add_argument("--mode", default="game", help="whole games or fights (training/selfplay/modes)")
    p.add_argument("--races", nargs=2, metavar=("OWN", "OPPONENT"), help="one matchup in every game (default: each in turn)")
    p.add_argument("--steps", type=int, default=0, help="per actor: stop after this many game steps (0: never; for tests)")
    p.add_argument("--wine", action="store_true", help="the actors' games under Wine (a Linux host; $WINEPREFIX)")
    p.add_argument("--nice", type=int, default=0, help="with --wine: the games' niceness")
    p.add_argument("--port", type=int, default=7600, help="the learner's; the inference process's is the next one")
    p.add_argument("--learner-gpu-share", type=int, default=65, help="under NVIDIA MPS, the percent of the GPU's SMs "
                   "the learner may use (100: all), so inference's batches are not stuck behind its kernels: on one 3090 "
                   "Ti, 65 took ~10%% more decided steps a second, inference's batch p90 18 -> 11 ms (rsi/log.md E15). "
                   "Without MPS it does nothing")
    p.add_argument("rest", nargs="*", help="learner flags, after --")
    args = p.parse_args()
    learner_args = learner.parser().parse_args(
        ["--init", args.init, "--out", args.out, "--listen", f"127.0.0.1:{args.port}", "--mode", args.mode, *args.rest]
    )
    learner_args.on_save = None
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.getcwd(), "src", os.environ.get("PYTHONPATH")]))}
    if args.learner_gpu_share < 100:  # read when this process's CUDA context is made (in Learner); not the children's
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.learner_gpu_share)
    the_learner = learner.Learner(learner_args)
    at_learner, at_inference = f"127.0.0.1:{args.port}", f"127.0.0.1:{args.port + 1}"

    def start(name: str, *argv: str) -> subprocess.Popen:
        log = open(os.path.join(args.out, f"{name}.log"), "a")  # noqa: SIM115 (open while the process runs)
        return subprocess.Popen([sys.executable, "-m", *argv, "--authkey", learner_args.authkey], env=env, stdout=log,
                                stderr=subprocess.STDOUT)  # fmt: skip

    children = [start("inference", "training.selfplay.inference", "--learner", at_learner, "--listen", at_inference)]
    children += [start(f"actor-{k}", "training.selfplay.actor", "--learner", at_learner, "--inference", at_inference, "--games",
                       str(args.games), "--ai-games", str(args.ai_games), "--eval-games", str(args.eval_games),
                       "--mode", args.mode, "--seed", str(k), "--steps", str(args.steps),
                       *(["--races", *args.races] if args.races else []), *(["--wine"] if args.wine else []), "--nice", str(args.nice))
                for k in range(args.actors)]  # fmt: skip
    try:
        the_learner.run(stop=lambda: all(proc.poll() is not None for proc in children[1:]))
    finally:
        the_learner.server.close()
        for proc in children:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
