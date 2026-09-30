"""Download 1v1 replays for one patch from warcraft3.info's replay database.

    python tools/fetch_replays.py --version 29 --out runs/replays/1.29

The site's JSON API lists replays with their patch (`version`: 29 for 1.29.x), game type, origin and
file type; this keeps Battle.net `.w3g` 1on1 games (NetEase's `.nwg` files need converting first). Files
are written as <id>.w3g next to index.json, the listing's records; existing files are skipped. Whether a
replay plays on this client is a separate question: tools/check_replays.py.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

API = "https://warcraft3.info/api/v1/replays"
HEADERS = {"User-Agent": "wc3env", "Content-Type": "application/json"}


def fetch(url: str, body: dict | None = None) -> bytes:
    request = urllib.request.Request(url, json.dumps(body).encode() if body else None, HEADERS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3 + 5 * attempt)
    raise AssertionError("unreachable")


def listing(version: str) -> list[dict]:
    rows, page, last = [], 1, 1
    while page <= last:
        result = json.loads(fetch(f"{API}/", {"page": page, "version": version}))
        rows += result["data"]
        last = result["last_page"]
        page += 1
        time.sleep(0.3)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="29", help="warcraft3.info's patch field: 29 for 1.29.x")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in listing(args.version) if r["type"] == "1on1" and r["filetype"] == "w3g"]
    (args.out / "index.json").write_text(json.dumps(rows), encoding="utf-8")
    for n, row in enumerate(rows, 1):
        target = args.out / f"{row['id']}.w3g"
        if not target.exists():
            target.write_bytes(fetch(f"{API}/{row['id']}/download"))
            time.sleep(0.25)
        if n % 50 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    print(f"{len(rows)} replays in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
