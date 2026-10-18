"""The shell scripts the worker image runs for training: the games of self-play actors (Modal), evaluation games
against the built-in AI and recordings of the AI against itself (GCP).

The worker image's entrypoint sets up the license and the Windows paths for its Windows-Python modes. A bridge or an
evaluation is Linux Python driving Windows game workers, so those scripts take the same steps themselves; a recording
runs the data set's code under the image's Windows Python, through the entrypoint. Run one as the image's worker
user inside its display wrapper:

    su worker -s /bin/bash -p -c 'bash /opt/worker/with-display.sh bash /work/script.sh'
"""

from __future__ import annotations

import argparse

WINDOWS_SITE = "/home/worker/.wine/drive_c/Python311/Lib/site-packages/wc3env"
GCP_PYTHONPATH = "/work:/opt/host:/opt/pylib"  # on GCP the host's CPU PyTorch is mounted at /opt/pylib


def _preamble(python_path: str, log_dir: str) -> str:
    """The license, the Windows paths and the output folder, for Linux Python driving Windows games."""
    return f"""#!/bin/bash
set -e
cd /work
export HOME=/home/worker PYTHONPATH={python_path} OMP_WAIT_POLICY=passive
export WC3_GAME_DIR='C:\\wc3' WC3_USER_DIR='C:\\scratch\\Warcraft III' AGENT_SESSIONS_DIR='Z:\\sessions'
python3 /opt/worker/license.py
mkdir -p {log_dir}
export WC3_OUTPUT_DIR="$(winepath -w {log_dir})"
"""


# Warcraft's lowest video options, in the prefix's registry: with drawing off they change nothing the games compute,
# and a game holds 246 MiB instead of 287 (textures and models it loads anyway; docs/training/9-benchmarks.md)
LOW_VIDEO = {"texquality": 0, "modeldetail": 0, "animquality": 0, "particles": 0, "lights": 0, "unitshadows": 0,
             "occlusion": 0, "spellfilter": 0, "texcolordepth": 16, "colordepth": 16, "reswidth": 640, "resheight": 480}


def low_video() -> str:
    """Shell lines setting LOW_VIDEO in $WINEPREFIX (before prefixes are copied from it)."""
    key = r"HKCU\Software\Blizzard Entertainment\Warcraft III\Video"
    adds = [f'wine reg add "{key}" /v {name} /t REG_DWORD /d {value} /f > /dev/null 2>&1 || true' for name, value in LOW_VIDEO.items()]
    return "\n".join([*adds, "wineserver -w", ""])


def bridge(addresses: list[str], workers: int, python_path: str = "/work:/opt/host", log_dir: str = "/sessions/out",
           listen: bool = False) -> str:  # fmt: skip
    """The games of actors that run elsewhere (training.selfplay.actor --remote), one per address: `workers` Wine
    workers each, every one carried over its own connection (python -m wc3env.vector --connect). listen: the
    addresses are ports here, and the actors connect in (python -m wc3env.vector --listen; wc3env.vector.dial_workers)."""
    # each bridge's workers in Wine prefixes of their own: sharing the image's prefix and the temporary copies
    # (wine-prefix-<w>) between bridges in one sandbox killed their workers at once. The prefixes are link copies
    # (wc3env.vector.link_prefix): 40 full copies of the 1.6 GB prefix kept test 5's games from starting
    lines = [f"(export TMPDIR=/tmp/bridge-{k} && mkdir -p $TMPDIR && python3 -c 'import os, pathlib, wc3env.vector as v; "
             f"v.link_prefix(pathlib.Path(os.environ[\"WINEPREFIX\"]), pathlib.Path(\"/tmp/bridge-{k}/base\"))' && "
             f"WINEPREFIX=$TMPDIR/base python3 -m wc3env.vector {'--listen' if listen else '--connect'} {a} --workers {workers} --wine) "
             f"> {log_dir}/bridge-{k}.log 2>&1 &" for k, a in enumerate(addresses)]  # fmt: skip
    return _preamble(python_path, log_dir) + low_video() + "\n".join([*lines, "wait", ""])


def evaluation(label: str, games: int, minutes: float, difficulty: int, threads: int, python_path: str = GCP_PYTHONPATH,
               log_dir: str = "/sessions/out") -> str:  # fmt: skip
    """`games` games at once of /work/checkpoint.pt against the built-in AI (training.play.evaluate): one JSON
    line per game in {log_dir}/eval.jsonl as each one ends, labelled with `label`."""
    return _preamble(python_path, log_dir) + f"""python3 -m training.play.evaluate /work/checkpoint.pt --label '{label}' --wine --games {games} --minutes {minutes} --difficulty {difficulty} --plans /work/plans.npz --threads {threads} > {log_dir}/eval.jsonl 2> {log_dir}/eval.err
"""


def recording(first_seed: int, games: int, workers: int, minutes: float, maps: str = "") -> str:
    """Games of Warcraft's AI against itself written as training data (training.replays.dataset --bots) into
    /sessions/out/data, by the image's Windows Python (the data set's code runs where the games do)."""
    return rf"""#!/bin/bash
set -e
cd /work
bash /opt/worker/entrypoint.sh python -m training.replays.dataset --bots {games} --first-seed {first_seed} --workers {workers} --minutes {minutes} --maps '{maps}' --out 'Z:\sessions\out\data' > /sessions/out/record.log 2>&1
"""


def conversion(ids: list[str], workers: int) -> str:
    """The data set files of replays /work/inputs/<id>.w3g (training.replays.dataset) into /sessions/out/data."""
    listing = "\n".join(ids)
    return rf"""#!/bin/bash
set -e
cd /work
cat > /work/ids.txt <<'IDS'
{listing}
IDS
bash /opt/worker/entrypoint.sh python -m training.replays.dataset 'Z:\work\inputs' --ids 'Z:\work\ids.txt' --workers {workers} --out 'Z:\sessions\out\data' > /sessions/out/convert.log 2>&1
"""


def learner_hours(learner_flags: list[str], default: float) -> float:
    """The learner's --hours among its flags (either spelling, --hours 4 or --hours=4), else `default`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--hours", type=float, default=default)
    return p.parse_known_args(learner_flags)[0].hours or default


def overlay_env(source: str = "/work/src/wc3env") -> str:
    """Root commands putting this checkout's wc3env (the DLL included) over the image's, on both sides: the actors'
    Linux Python (/opt/host) and the games' Windows Python."""
    return f"cp -r {source}/. /opt/host/wc3env/ && cp -r {source}/. {WINDOWS_SITE}/ && chown -R 10001:10001 {WINDOWS_SITE}"
