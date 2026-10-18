"""A cloning run's charts on a local page: losses, speed, gradient norm and held-out accuracy (with the plan and
with the empty plan), from the run's container log on Modal.

    python -m training.cloud.dashboard --app ap-LbQYdE7kDCzUuQpj8EVaDc --name bc11 --steps 70000
    then open http://localhost:8766 (it redraws every 30 s; refresh any time)

`modal app logs` gives only the latest lines, so every 30 s the new ones are added to build/dash/<name>.log, which
keeps the whole run (the Volume's log.txt only changes at checkpoints).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP = re.compile(r"(\d\d:\d\d:\d\d) step (\d+) lr ([0-9.e+-]+) (\d+) steps/s \| (.*)")
HELD = re.compile(r"(\d\d:\d\d:\d\d) held-out at step (\d+)(, empty plan)?: (.*)")
PAIRS = re.compile(r"(\w+) ([0-9.e+-]+)")


def collect(app: str, log: Path, every: float = 30) -> None:
    """Forever: the app's latest log lines, the new ones added to `log`."""
    known = set(log.read_text(encoding="utf-8", errors="replace").splitlines()) if log.exists() else set()
    while True:
        try:
            out = subprocess.run([sys.executable, "-m", "modal", "app", "logs", app], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=90).stdout  # fmt: skip
            fresh = [l for l in out.splitlines() if l.strip() and l not in known and not re.search(r"deprecat|set_event", l, re.I)]
            if fresh:
                known.update(fresh)
                with log.open("a", encoding="utf-8") as f:
                    f.write("\n".join(fresh) + "\n")
        except Exception as e:  # a failed fetch: try again next time
            print(f"log fetch failed: {e}", file=sys.stderr)
        time.sleep(every)


def parse(log: Path) -> dict:
    train, held = {}, []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []:
        if m := STEP.search(line):
            row = {"time": m[1], "step": int(m[2]), "lr": float(m[3]), "steps_per_s": int(m[4])}
            row.update({k: float(v) for k, v in PAIRS.findall(m[5])})
            train[row["step"]] = row  # a line fetched twice once
        elif m := HELD.search(line):
            row = {"time": m[1], "step": int(m[2]), "plan": "empty" if m[3] else "plan"}
            row.update({k: float(v) for k, v in PAIRS.findall(m[4])})
            held.append(row)
    return {"train": [train[k] for k in sorted(train)], "held": held}


PAGE = """<!doctype html><meta charset="utf-8"><title>%(name)s</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#f6f7f9;color:#1d2330}
h1{font-size:20px;margin:0 0 4px} #top{color:#555;margin-bottom:12px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,520px),1fr));gap:12px}
.card{background:#fff;border:1px solid #dde1e7;border-radius:8px;padding:12px} .card h2{font-size:14px;margin:0 0 6px}
.note{color:#666;font-size:12px;margin:0 0 6px} canvas{max-height:300px}
</style>
<h1>%(name)s</h1><div id="top">loading...</div>
<div class="grid">
<div class="card"><h2>Training loss per head</h2><p class="note">cross-entropy on the training batches (lower is better)</p><canvas id="loss"></canvas></div>
<div class="card"><h2>Held-out accuracy: plan vs empty plan</h2><p class="note">solid: each game's own plan; dashed: the empty plan. The gap is how much the plan is used</p><canvas id="held"></canvas></div>
<div class="card"><h2>Speed</h2><p class="note">game steps learned per second; loader wait per update (s, right)</p><canvas id="speed"></canvas></div>
<div class="card"><h2>Gradient norm and learning rate</h2><p class="note">before clipping; learning rate on the right</p><canvas id="grad"></canvas></div>
</div>
<script>
const TOTAL = %(steps)d, BATCH = 2048, charts = {};
const C = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#db2777","#4b5563"];
function draw(id, datasets, y1) {
  const opts = {animation:false, parsing:false, normalized:true, interaction:{mode:"nearest",axis:"x",intersect:false},
    elements:{point:{radius:0}, line:{borderWidth:1.5}}, plugins:{legend:{labels:{boxWidth:12}}},
    scales:{x:{type:"linear", title:{display:true, text:"update"}}, y:{beginAtZero:false}}};
  if (y1) opts.scales.y1 = {position:"right", grid:{drawOnChartArea:false}};
  if (charts[id]) { charts[id].data.datasets = datasets; charts[id].update(); return; }
  charts[id] = new Chart(document.getElementById(id), {type:"line", data:{datasets}, options:opts});
}
const pts = (rows, k) => rows.filter(r => r[k] !== undefined).map(r => ({x:r.step, y:r[k]}));
async function refresh() {
  const d = await (await fetch("data.json")).json(), t = d.train, h = d.held;
  if (!t.length) { document.getElementById("top").textContent = "no training lines yet"; return; }
  const last = t[t.length-1], recent = t.slice(-12), rate = recent.reduce((a,r)=>a+r.steps_per_s,0)/recent.length;
  const left = (TOTAL-last.step)*BATCH/rate/60;
  document.getElementById("top").textContent = `update ${last.step.toLocaleString()} of ${TOTAL.toLocaleString()} (${(100*last.step/TOTAL).toFixed(1)}%%) · ${Math.round(rate).toLocaleString()} steps/s · ~${Math.floor(left/60)} h ${Math.round(left%%60)} min left · last line ${last.time} UTC`;
  draw("loss", ["type","argument","actor","target","region","size","cell"].map((k,i)=>({label:k, data:pts(t,k), borderColor:C[i]})));
  const ds = [];
  ["type","argument","actor","target"].forEach((k,i)=>{ for (const p of ["plan","empty"]) ds.push({label:`${k} ${p==="plan"?"":"(empty)"}`,
    data:pts(h.filter(r=>r.plan===p),k), borderColor:C[i], borderDash:p==="empty"?[5,4]:[], pointRadius:3}); });
  draw("held", ds);
  draw("speed", [{label:"steps/s", data:pts(t,"steps_per_s"), borderColor:C[0]}, {label:"loader wait (s)", data:pts(t,"wait_s"), borderColor:C[3], yAxisID:"y1"}], true);
  draw("grad", [{label:"grad norm", data:pts(t,"grad_norm"), borderColor:C[1]}, {label:"lr", data:pts(t,"lr"), borderColor:C[4], yAxisID:"y1"}], true);
}
refresh(); setInterval(refresh, 30000);
</script>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", required=True, help="the run's Modal app id (ap-...)")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=70000, help="the run's updates in all, for progress and time left")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    log = ROOT / "build" / "dash" / f"{a.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=collect, args=(a.app, log), daemon=True).start()
    page = (PAGE % {"name": a.name, "steps": a.steps}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body, kind = (json.dumps(parse(log)).encode(), "application/json") if self.path.startswith("/data.json") else (page, "text/html")
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    print(f"http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
