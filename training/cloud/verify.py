"""training.replays.check_dataset over the data set in the Modal Volume, in a function beside it (no Wine).

python -m training.cloud.verify [dataset2]
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .add_local_dir(ROOT / "src" / "wc3env", "/root/wc3env", ignore=["**/*.dll", "**/__pycache__/**"])
    .add_local_dir(ROOT / "training", "/root/training", ignore=["**/__pycache__/**", "tests/**"])
)
app = modal.App("wc3-verify", image=image)


@app.function(volumes={"/dataset": modal.Volume.from_name("wc3-dataset")}, cpu=16, memory=32768, timeout=3 * 3600)
def check_files(folder: str = "dataset") -> str:
    import contextlib
    import io
    import sys

    from training.replays.check_dataset import main

    sys.argv = ["check_dataset", f"/dataset/{folder}", "--workers", "16"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main()
    Path(f"/dataset/verify-{folder}.txt").write_text(out.getvalue())  # kept in the Volume if the caller loses its connection
    return out.getvalue()


if __name__ == "__main__":
    import sys as _sys

    with modal.enable_output(), app.run():
        print(check_files.remote(_sys.argv[1] if len(_sys.argv) > 1 else "dataset"))
