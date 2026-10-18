"""Check every file training/replays/dataset.py wrote.

    python -m training.replays.check_dataset runs/dataset --workers 6

Per file: the rows add up to the step counts; every number is finite; labels are in range; each label is
allowed by its own mask (type, argument, actor, target); the loader rebuilds the map features() made at the
stored check steps (walkable/buildable fractions are stored as bytes: within 0.005). Prints the totals and
the files with problems.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def check(path: str) -> dict:
    from ..policy.actions import SIZES, TYPES
    from ..policy.features import MAX_UNITS
    from ..policy.masks import argument_ids
    from ..train.loader import ReplayData

    out = Counter()
    try:
        data = ReplayData(path)
        d = data.d
        t = data.steps
        out["files"] += 1
        out["steps"] += t
        out["actions"] += int((d["type"] != 0).sum())
        out["problem: rows do not add up"] += int(d["unit_count"].sum() != len(d["units"]))
        out["problem: rows over the cap"] += int((d["unit_count"] > MAX_UNITS).sum())
        for k in ("units", "scalars"):
            out[f"problem: non-finite {k}"] += int((~np.isfinite(d[k].astype(np.float32))).sum())
        n_args = len(argument_ids()) + 1
        out["problem: type out of range"] += int(((d["type"] < 0) | (d["type"] >= len(TYPES))).sum())
        out["problem: argument out of range"] += int(((d["argument"] < 0) | (d["argument"] >= n_args)).sum())
        out["problem: size out of range"] += int(((d["size"] < 0) | (d["size"] >= len(SIZES))).sum())
        rows = d["unit_count"].astype(np.int64)
        for head in ("actor", "target"):
            v = d[head].astype(np.int64)
            out[f"problem: {head} beyond the rows"] += int(((v >= 0) & (v >= rows)).sum())
        steps = np.arange(t)
        out["labels: type"] += t
        out["labels: type allowed"] += int(d["type_mask"][steps, d["type"].astype(np.int64)].sum())
        argument_mask = np.unpackbits(d["argument_mask"], axis=1)[:, :n_args].astype(bool)
        has = d["argument"] > 0
        out["labels: argument"] += int(has.sum())
        out["labels: argument allowed"] += int(argument_mask[steps[has], d["argument"][has]].sum())
        for head in ("actor", "target"):
            mask = np.unpackbits(d[f"{head}_mask"], axis=1)[:, :MAX_UNITS].astype(bool)
            v = d[head].astype(np.int64)
            has = v >= 0
            out[f"labels: {head}"] += int(has.sum())
            out[f"labels: {head} allowed"] += int(mask[steps[has], v[has]].sum())
        worst = 0.0
        for s, expected in zip(d.get("check_steps", ()), d.get("check_maps", ())):  # files from before the hook's features
            got = data.step(int(s), map=True)["map"][: len(expected)]  # files written before a channel was added
            worst = max(worst, float(np.abs(got - expected.astype(np.float32)).max()))
        out["maps checked"] += len(d.get("check_steps", ()))
        out["problem: map rebuilt differently"] += int(worst > 0.005)
        out["result known"] += int(d["result"] >= 0)
    except Exception as exc:  # noqa: BLE001
        out[f"problem: unreadable ({type(exc).__name__})"] += 1
    return {"file": Path(path).name, **out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    files = sorted(str(p) for p in args.folder.glob("*.npz"))
    total, bad = Counter(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(check, files, chunksize=4):
            name = row.pop("file")
            total.update(row)
            if any(k.startswith("problem") and v for k, v in row.items()):
                bad.append((name, {k: v for k, v in row.items() if k.startswith("problem") and v}))
    size = sum(Path(f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e9:.2f} GB")
    print(json.dumps(dict(sorted(total.items())), indent=1))
    for head in ("type", "argument", "actor", "target"):
        n = total[f"labels: {head}"]
        print(f"{head}: {total[f'labels: {head} allowed'] / max(n, 1):.4%} of {n} labels allowed by their mask")
    print(f"files with problems: {len(bad)}")
    for name, problems in bad[:20]:
        print(" ", name, problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
