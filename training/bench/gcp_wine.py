"""wine.py's bench on a GCP CPU VM (the worker image, as evaluations): N no-op games stepped through wine_workers,
per run the wall time per step and the CPU time per game-step, to set against a Modal sandbox's (9.3 ms,
docs/training/9-benchmarks.md) at each one's price per vCPU. A vCPU here is a hyperthread (C2D, N2D) or a whole
core (T2D); /proc/stat counts the threads the VM has.

    python -m training.bench.gcp_wine --machine c2d-highcpu-32 [--spot] [--games 16,32,48]
    python -m training.bench.gcp_wine --machine c2d-highcpu-32 --collect   # its bench.log, once uploaded

The VM uploads /sessions/out/bench.log every two minutes and deletes itself when done."""

from __future__ import annotations

import argparse
import types

from training.bench.wine import BENCH
from training.cloud.scripts import GCP_PYTHONPATH, _preamble
from training.cloud.vm import cat, cpu_job, prepare, run_folder

STEPS = 300


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default="c2d-highcpu-32")
    p.add_argument("--games", default="16,32,48")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--zone", default="us-central1-a")
    p.add_argument("--collect", action="store_true")
    a = p.parse_args()
    args = types.SimpleNamespace(project="wc3env-pw", zone=a.zone, zones="", spot=a.spot, actor_machine=a.machine,
                                 name=f"wine-{a.machine}")  # fmt: skip
    vm = f"wc3-bench-{a.machine}"
    if a.collect:
        print(cat(f"{run_folder(args, 'bench')}/out/{vm}/bench.log") or "nothing yet")
        return
    run = prepare(args, "bench")
    script = _preamble(GCP_PYTHONPATH, "/sessions/out") + f"cat > /work/bench.py <<'EOF'\n{BENCH}\nEOF\n" + "".join(
        f"python3 /work/bench.py {n} {STEPS} >> /sessions/out/bench.log 2>> /sessions/out/bench.err.log\n"
        for n in a.games.split(","))  # fmt: skip
    zone = cpu_job(args, run, vm, script, hours=1)
    print(f"{vm} in {zone or 'no zone'}; then: python -m training.bench.gcp_wine --machine {a.machine} --collect")


if __name__ == "__main__":
    main()
