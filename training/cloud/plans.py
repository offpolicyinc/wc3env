"""The plan bank (training/policy/plan.py PlanBank) from a data set folder in the Volume, in a Modal function.

    python -m training.cloud.plans dataset3 runs/plans.npz     # also kept in the Volume as /dataset/plans.npz
"""

import io
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
    .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
)
app = modal.App("wc3-plans", image=image)


@app.function(volumes={"/dataset": modal.Volume.from_name("wc3-dataset")}, cpu=4, timeout=1800)
def build(folder: str) -> bytes:
    import numpy as np

    from training.policy.plan import bank_arrays

    arrays = bank_arrays(sorted(Path(f"/dataset/{folder}").glob("*.npz")))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    Path("/dataset/plans.npz").write_bytes(buffer.getvalue())
    return buffer.getvalue()


if __name__ == "__main__":
    folder, out = sys.argv[1], Path(sys.argv[2])
    with modal.enable_output(), app.run():
        data = build.remote(folder)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{out}: {len(data) / 1e6:.1f} MB")
