"""The games' link if they ran on GCP VMs and the GPU container stayed on Modal: an echo VM per region (a public
address, the tunnel bench's listener replying REPLY bytes to every command, as a game its observation), for
tunnel_latency.py's "ext" arm to play 128 games' loops against from a Modal container in the matching region.

    python -m training.bench.gcp_link up        # the firewall rule and the two VMs; prints each one's host:port
    BENCH_REGION=us-east python -m modal run training/bench/tunnel_latency.py --arms ext,tls-out --external <us-east4 host:port>
    BENCH_REGION=us-west python -m modal run training/bench/tunnel_latency.py --arms ext,tls-out --external <us-west1 host:port>
    python -m training.bench.gcp_link down      # both VMs and the rule

The VMs are spot e2-small (about $0.01/hour each) and delete themselves after HOURS at the latest."""

from __future__ import annotations

import sys
import time

from training.bench.tunnel_latency import BOX_PORT, CLIENT, REPLY
from training.cloud.vm import ROOT, gcloud

PROJECT = "wc3env-pw"
ZONES = {"us-east": "us-east4-a", "us-west": "us-west1-a"}  # Modal's region -> the GCP zone beside it
RULE, TAG, LABEL, HOURS = "wc3-bench-echo", "wc3-bench-echo", "wc3-run=bench-link", 2


def up() -> None:
    if not gcloud("compute", "firewall-rules", "list", f"--project={PROJECT}", f"--filter=name={RULE}",
                  "--format=value(name)", capture=True):  # fmt: skip
        gcloud("compute", "firewall-rules", "create", RULE, f"--project={PROJECT}", f"--allow=tcp:{BOX_PORT}",
               "--source-ranges=0.0.0.0/0", f"--target-tags={TAG}")  # fmt: skip
    script = ROOT / "build" / "gcp-bench-echo.sh"
    script.parent.mkdir(exist_ok=True)
    script.write_text(f"#!/bin/bash\ncat > /opt/echo.py <<'EOF'\n{CLIENT}\nEOF\n"
                      f"nohup python3 /opt/echo.py listen {REPLY} {BOX_PORT} > /var/log/echo.log 2>&1 &\n", newline="\n")  # fmt: skip
    for region, zone in ZONES.items():
        gcloud("compute", "instances", "create", f"bench-echo-{region}", f"--project={PROJECT}", f"--zone={zone}",
               "--machine-type=e2-small", "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
               f"--max-run-duration={HOURS * 3600}s", "--image-family=ubuntu-2404-lts-amd64",
               "--image-project=ubuntu-os-cloud", f"--tags={TAG}", f"--labels={LABEL}",
               f"--metadata-from-file=startup-script={script}")  # fmt: skip
    time.sleep(60)  # the listener starts with the VM
    for region, zone in ZONES.items():
        ip = gcloud("compute", "instances", "describe", f"bench-echo-{region}", f"--project={PROJECT}", f"--zone={zone}",
                    "--format=value(networkInterfaces[0].accessConfigs[0].natIP)", capture=True)  # fmt: skip
        print(f"{region}: {ip}:{BOX_PORT}")


def down() -> None:
    for region, zone in ZONES.items():
        gcloud("compute", "instances", "delete", f"bench-echo-{region}", f"--project={PROJECT}", f"--zone={zone}",
               "--quiet", check=False)  # fmt: skip
    gcloud("compute", "firewall-rules", "delete", RULE, f"--project={PROJECT}", "--quiet", check=False)
    left = gcloud("compute", "instances", "list", f"--project={PROJECT}", f"--filter=labels.{LABEL}",
                  "--format=value(name)", capture=True)  # fmt: skip
    print("left:", left or "nothing")


if __name__ == "__main__":
    {"up": up, "down": down}[sys.argv[1]]()
