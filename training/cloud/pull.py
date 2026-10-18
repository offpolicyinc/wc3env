"""Copy a folder of the GCS bucket into the Modal Volume wc3-dataset, from inside Modal (cloud to cloud, nothing
through this machine): data recorded on GCP (training.cloud.gcp record) for cloning on Modal's H100.

    python -m training.cloud.pull gs://wc3env-pw-sessions/record/r1/data bots1     # -> /dataset/bots1/

The job reads the bucket with a one-hour access token of the read-only service account wc3-bucket-reader (allowed
to read this bucket only), made here by impersonation: no key file exists anywhere, and the token is handed to the
job as an ephemeral secret, never stored in Modal or put on a command line. Files already in the Volume with the
same size are skipped and the Volume is committed every 200 files, so a pull can be repeated (a token runs out after
an hour: run it again).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import modal

READER = "wc3-bucket-reader@wc3env-pw.iam.gserviceaccount.com"
VOLUME = modal.Volume.from_name("wc3-dataset")
app = modal.App("wc3-pull", image=modal.Image.debian_slim(python_version="3.11").pip_install("google-cloud-storage"))


@app.function(cpu=8, memory=8192, timeout=3600, volumes={"/dataset": VOLUME})
def pull(source: str, folder: str) -> str:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    bucket_name, _, prefix = source.removeprefix("gs://").partition("/")
    client = storage.Client(project="wc3env-pw", credentials=Credentials(os.environ["GCS_TOKEN"]))
    blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/") if not b.name.endswith("/")]
    out = Path("/dataset") / folder
    out.mkdir(parents=True, exist_ok=True)

    def get(blob) -> int:
        target = out / blob.name[len(prefix.rstrip("/")) + 1 :]  # its path under the folder, kept
        if target.exists() and target.stat().st_size == blob.size:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        return blob.size

    sizes = []
    with ThreadPoolExecutor(32) as pool:
        for k in range(0, len(blobs), 200):  # committed as it goes: a pull cut off by the hour keeps what it copied
            sizes += pool.map(get, blobs[k : k + 200])
            VOLUME.commit()
    copied = [s for s in sizes if s]
    return f"{len(copied)} files copied ({sum(copied) / 1e9:.2f} GB), {len(blobs) - len(copied)} already there, into /dataset/{folder}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="gs://bucket/folder")
    p.add_argument("folder", help="the Volume folder to fill")
    args = p.parse_args()
    gcloud = "gcloud.cmd" if sys.platform == "win32" else "gcloud"
    token = subprocess.run([gcloud, "auth", "print-access-token", f"--impersonate-service-account={READER}"],
                           capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    with modal.enable_output(), app.run():
        print(pull.with_options(secrets=[modal.Secret.from_dict({"GCS_TOKEN": token})]).remote(args.source, args.folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
