"""A self-play run's few charts that matter, on a local page, from its W&B history (the full run, every update).

    python -m training.cloud.selfplay_dashboard --name sp35 [--compare sp36] [--port 8767]
    then open http://localhost:8767 (it fetches again on every page load while the run is live)

Each chart says what it shows and what good looks like. Needs WANDB_API_KEY (in .env).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import modal
import numpy as np

import wandb

PROJECT = "off-policy-inc/wc3"
HEADS = ("type", "actor", "target", "region", "cell", "argument", "size", "same_type", "queued", "aim")  # model.HEADS
# combat's order types but noop (modes/combat.py FIGHT_TYPES)
ORDERS = ("move", "attack_move", "smart", "attack", "stop", "hold", "cast", "cast_point", "cast_unit", "use_item")
CRITIC_CHANNELS = ("win", "build", "units", "upgrades", "dealt", "taken")  # model.CRITIC
KEYS = [
    *(f"play/entropy_{h}" for h in HEADS),
    *(f"play/kl_{h}" for h in HEADS),
    "games/ladder_won",
    "games/ladder_lost",
    "games/ladder_time",
    "games/self_pool_won",
    "games/self_pool_lost",
    "games/self_pool_time",
    "play/handicap",
    "play/minutes",
    "play/entropy",
    "play/kl_to_clone",
    "play/reward_win",
    "play/reward_dealt",
    "play/reward_taken",
    "play/reward_build",
    "play/reward_units",
    "play/reward_upgrades",
    "play/reward_scout",
    "play/units_mean",
    "play/units_max",
    "play/gold_min",
    "play/value_explained",
    "play/explained_weighted",
    *(f"play/explained_{c}" for c in CRITIC_CHANNELS),
    "play/clip_fraction",
    "play/grad_norm",
    "play/steps_per_s",
    "play/games",
    "play/ladder",
    *(f"play/share_{o}" for o in ORDERS),
    "play/share_noop",
]

HEAD_NAMES = {
    "type": "type: kind of order",
    "actor": "actor: which own unit",
    "target": "target: which unit aimed at",
    "region": "region: where on the map",
    "cell": "cell: spot within the region",
    "argument": "argument: what to build / train / cast",
    "size": "size: how many units",
    "same_type": "same_type: units of one kind",
    "queued": "queued",
    "aim": "aim",
}
# (chart, its note, [(series, its name in the legend)]): one chart a row, the legend down its right side
SECTIONS = [
    (
        "Ladder: against the opponent the league sets",
        "Ladder games (Warcraft's AI in whole games, the scripted attack-move in combat; games.jsonl) in the order "
        "they started, as the league counts them, in blocks of 200: each point one block's share won, lost and run to "
        "the time limit; x: games played (all of them, ladder and self-play) when the block's last one started. Only games old enough that every game started before them has ended. Bold: the share over the last 1,000 games; faint: each block.",
        [("ladder_blocks/won", "won"), ("ladder_blocks/lost", "lost"), ("ladder_blocks/time limit", "time limit")],
    ),
    (
        "Self-play: against itself and its snapshots",
        "Self-play games in blocks of 200 as they finish: each point one block's player-0 share won, lost and run to "
        "the time limit (a loss for both); x: games played by the block's last. Runs before cb6 (no self-play games in "
        "games.jsonl): of the last 200 at each update.",
        [("self_blocks/won", "won"), ("self_blocks/lost", "lost"), ("self_blocks/time limit", "time limit"),
         ("games/self_pool_won", "won"), ("games/self_pool_lost", "lost"), ("games/self_pool_time", "time limit")],
    ),
    (
        "Ladder results by start time",
        "The ladder's shares, with games in the order they started (runs after sp39: the learner's "
        "games.jsonl): each point, the last 480 games to start; x: the update the last of them started at. "
        "of them. Only games old enough that every game started then has finished.",
        [("by_start/won", "won"), ("by_start/lost", "lost"), ("by_start/time limit", "time limit")],
    ),
    (
        "Entropy (nats)",
        "How spread its choices are: H = -sum p ln p over a head's options; e^H is the effective number "
        "of equally likely options (0.7: ~2, 2.0: ~7). total: the heads one decision used, added, averaged over all "
        "decisions. A head: averaged over only the decisions that use it (so the heads do not add up to the total).",
        [("play/entropy", "TOTAL"), *((f"play/entropy_{h}", HEAD_NAMES[h]) for h in HEADS)],
    ),
    (
        "KL to the clone (nats)",
        "How far the policy has moved from the checkpoint it started from: sum over options of "
        "p_clone ln(p_clone / p_now); e^-KL is about how much less likely the clone's own choice has become (0.3: ~25% "
        "less). By head only for runs after sp35.",
        [("play/kl_to_clone", "TOTAL"), *((f"play/kl_{h}", HEAD_NAMES[h]) for h in HEADS)],
    ),
    (
        "Rewards per update",
        "What the policy is paid for, per channel. dealt: damage to the enemy; taken: damage "
        "received (negative); win: +1 / -1 at a game's end. Good: dealt and win up, taken toward 0.",
        [
            ("play/reward_dealt", "dealt"),
            ("play/reward_taken", "taken"),
            ("play/reward_win", "win"),
            ("play/reward_build", "build"),
            ("play/reward_units", "units"),
            ("play/reward_upgrades", "upgrades"),
            ("play/reward_scout", "scout"),
        ],
    ),
    ("Army", "Units it has over its games.", [("play/units_mean", "units, mean"), ("play/units_max", "units, max")]),
    ("Economy", "Gold mined per minute over its games.", [("play/gold_min", "gold per minute")]),
    (
        "Critic",
        "How well the critic predicts each channel's returns (1 perfect, 0 useless); TOTAL: the weighted mix the "
        "advantage uses; value explained: the win channel (runs before sp37).",
        [
            ("play/explained_weighted", "TOTAL"),
            *((f"play/explained_{c}", c) for c in CRITIC_CHANNELS),
            ("play/value_explained", "value explained"),
        ],
    ),
    (
        "Update size",
        "Share of the update PPO clipped (high: steps too big).",
        [("play/clip_fraction", "clip fraction")],
    ),
    ("Speed", "Game steps learned per second.", [("play/steps_per_s", "steps per second")]),
]
COMBAT_ONLY = [
    (
        "Orders: what it does with a step",
        "Each order type's share of the learner's steps in the update (no order: the next chart). --explore-floor holds "
        "each at least that chance where legal until it has been chosen --explore-tries times (the log's floored: how "
        "many types are still held).",
        [*((f"play/share_{o}", o) for o in ORDERS)],
    ),
    (
        "No order",
        "The share of steps it gives no order (its units keep their last one, and fight what comes in range).",
        [("play/share_noop", "no order")],
    ),
    (
        "Fight length",
        "Game minutes per fight, over the last 200 (ladder and self-play): a fight is decided once one "
        "side's army value stays below 30% of the other's for 2 s, or reaches the 150 s time limit.",
        [("play/minutes", "minutes")],
    ),
]
# whole games' charts; and the start-order one: it trails by the longest recent game, there for whole games (a wave's
# wins end first, its time limits last), while a fight lasts a minute or two
NOT_COMBAT = ("Army", "Economy", "Ladder results by start time")
COMBAT_CHANNELS = ("win", "dealt", "taken", "weighted")  # combat's reward channels (modes/combat.py), and the mix


# the spells, items and skills charts (combat): each argument's share of the steps, the most used ones
CHOSEN = (
    ("Spells: which it casts", "spell_", "Each spell's share of the learner's steps in the update (cast, cast_point and cast_unit), the 11 most cast."),
    ("Items: which it uses", "item_", "Each item's share of the learner's steps in the update (use_item), the 11 most used."),
    ("Skills: which it learns", "learn_", "Each skill's share of the learner's steps in the update (a hero levelling up mid-fight), the 11 most learned."),
)


def chosen_sections(data: dict) -> list:
    """CHOSEN's charts, each with its run's most chosen arguments (empty before cb9)."""
    out = []
    for title, prefix, note in CHOSEN:
        totals = {k: sum(v for _, v in p) for k, p in data.items() if k.startswith("play/" + prefix) and isinstance(p, list)}
        top = sorted(totals, key=totals.get, reverse=True)[: len(COLORS)]
        out.append((title, note, [(k, k[len("play/" + prefix):].replace("_", " ")) for k in top] or [("play/" + prefix, "none")]))
    return out


def sections(mode: str, data: dict | None = None) -> list:
    """The charts for a run of this mode (training/selfplay/modes): combat's without the economy and the whole games'
    reward channels, with the fight length and which spells, items and skills it chooses."""
    if mode != "combat":
        return SECTIONS
    out = []
    for title, note, series in SECTIONS:
        if title in NOT_COMBAT:
            continue
        if title in ("Rewards per update", "Critic"):
            series = [(k, n) for k, n in series if k.rsplit("_", 1)[-1] in COMBAT_CHANNELS]
        out.append((title, note, series))
    at = next(i for i, (t, _, _) in enumerate(out) if t.startswith("Self-play")) + 1
    return out[:at] + COMBAT_ONLY + chosen_sections(data or {}) + out[at:]


# the dataviz palette's eight categorical slots in order, then three more drawn dashed; TOTAL is near-black and thick
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
]


ROOT = Path(__file__).resolve().parents[2]
CACHE: dict = {}  # run -> (when fetched, its series): a page load within a minute reuses them


def history(name: str) -> dict:
    if name in CACHE and time.time() - CACHE[name][0] < 60:
        return CACHE[name][1]
    if "WANDB_API_KEY" not in os.environ and (ROOT / ".env").exists():  # the key, as .env keeps it
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("WANDB_API_KEY="):
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    run = wandb.Api().run(f"{PROJECT}/{name}")
    rows = list(run.scan_history(keys=None))
    for r in rows:  # the ladder's shares, whatever its rung is called (ai0..ai2, attack-move: "move" before track.py
        # took hyphens), as games/ladder_*
        for k in list(r):
            m = re.fullmatch(r"games/(.+)_(won|lost|time)", k)
            if m and m[1] != "self_pool" and r[k] is not None:
                r[f"games/ladder_{m[2]}"] = r[k]
                r["_rung"] = "attack-move" if m[1] == "move" else m[1]
    # and each spell, item and skill the learner logged (play/spell_*, play/item_*, play/learn_*: from cb9)
    keys = [*KEYS, *sorted({k for r in rows for k in r if re.match(r"play/(spell|item|learn)_", k)})]
    out = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            if r.get(k) is not None:
                out[k].append([r["_step"], r[k]])
    out["stages"] = stages(rows)
    # [update] each time the run was resumed: a gap of over 10 minutes between two logged updates
    timed = sorted((r["_step"], r["_timestamp"]) for r in rows if r.get("_timestamp") is not None)
    out["resumes"] = [[u] for (_, a), (u, b) in zip(timed, timed[1:]) if b - a > 600]
    now = [r for r in rows if r.get("_rung")]  # what the ladder is now, for the page's first line
    out["mode"] = run.config.get("mode", "game")  # the learner's --mode (training/selfplay/modes)
    out["status"] = (
        {"rung": now[-1]["_rung"], "handicap": now[-1].get("play/handicap"), "update": now[-1]["_step"]} if now else {}
    )
    out.update(by_start(name))
    # x: games played (every game, ladder and self-play: the learner's play/games, logged with each update), when the run
    # logged it; each series' points and the ladder's moves are on the update until here
    done = sorted((r["_step"], r["play/games"]) for r in rows if r.get("play/games") is not None)
    # from 0 games: a run resumed from another's checkpoint (cb9 from cb8's update 2200) counts on from that run's
    # games, but its charts start at its own first
    done = [(0, 0)] + [(u, g - done[0][1]) for u, g in done] if done and done[0][0] > 50 else [(0, 0), *done]
    out["x"] = "games" if len(done) > 1 else "update"
    if len(done) > 1:
        played = lambda u: round(float(np.interp(u, *zip(*done))))  # noqa: E731
        for k, v in out.items():
            if isinstance(v, list):
                out[k] = [[played(p[0]), *p[1:]] for p in v]
    CACHE[name] = (time.time(), out)
    return out


BLOCK = 200  # games in each point of the ladder and self-play charts
AVERAGE = 1000  # and the faint line beside them: the share over the last this many games, at each block's end
EVERY = 20  # games between the points of the start-order chart
SPAN = 480  # games each of its points covers: a wave (16 actors x 32 games, less the lost ones)
OUTCOMES = {"victory": "won", "defeat": "lost"}  # games.jsonl's outcomes as the charts name them


def stages(rows: list) -> list:
    """[update, from, to] each time the ladder moved on (its rung or its stage's handicap), in update order."""
    out, last = [], None
    for r in sorted(rows, key=lambda r: r["_step"]):
        if r.get("play/handicap") is None:
            continue
        now = (r.get("_rung") or (last or ("", 0))[0], r["play/handicap"])
        if last is not None and now != last:
            show = (lambda v: f"{v[0]} {v[1]:g}") if now[0] != last[0] else (lambda v: f"{v[1]:g}")
            out.append([r["_step"], show(last), show(now)])
        last = now
    return out


def blocks(games: list, prefix: str, x) -> dict:
    """Won / lost / time-limit shares of each BLOCK games in order, x(the block's last game) its update; and
    (prefix_avg) the shares over the last AVERAGE games at each block's end, once there are that many."""
    out = {}
    for k in ("won", "lost", "time limit"):
        hit = np.cumsum([0] + [OUTCOMES.get(g["outcome"], g["outcome"]) == k for g in games])
        ends = range(BLOCK, len(games) + 1, BLOCK)
        out[f"{prefix}/{k}"] = [[x(games[i - 1]), (hit[i] - hit[i - BLOCK]) / BLOCK] for i in ends]
        out[f"{prefix}_avg/{k}"] = [[x(games[i - 1]), (hit[i] - hit[i - AVERAGE]) / AVERAGE] for i in ends if i >= AVERAGE]
    return out


def by_start(name: str) -> dict:
    """From games.jsonl: the ladder in blocks of BLOCK games in start order, self-play
    in blocks as they finish, and the start-order shares over each SPAN games, a point every EVERY games, x = the
    update the last of them started at. Only starts before (latest end - longest game): any game started later may not
    have ended yet."""
    try:
        raw = b"".join(modal.Volume.from_name("wc3-dataset").read_file(f"runs/{name}/games.jsonl")).decode()
    except Exception:  # runs before the log, or no volume access
        return {}
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [g for g in rows if "eval" not in g]  # evaluation games (--eval-games): neither ladder nor self-play
    games = [g for g in rows if g.get("began") is not None]  # the ladder's (self-play games have no start, from cb6)
    out = blocks(sorted((g for g in rows if g.get("began") is None), key=lambda g: g["ended"]), "self_blocks", lambda g: g["update"])
    if not games:
        return out
    # nearly the longest of the recent games (the 98th percentile): one outlier (cb1: a first fight of 12 minutes while
    # the run warmed up, a fight of 815 s later) would hide every game after it
    recent = sorted(g["ended"] - g["began"] for g in sorted(games, key=lambda g: g["ended"])[-SPAN:])
    cutoff = max(g["ended"] for g in games) - recent[int(0.98 * (len(recent) - 1))]
    started = sorted(games, key=lambda g: g["began"])
    games = [g for g in started if g["began"] <= cutoff]
    ends = sorted((g["ended"], g["update"]) for g in games)
    at = lambda t: round(float(np.interp(t, [e for e, _ in ends], [u for _, u in ends])))  # noqa: E731  time -> update
    out.update(blocks(games, "ladder_blocks", lambda g: at(g["began"])))  # in start order: x when its last one started
    for k in ("by_start/won", "by_start/lost", "by_start/time limit"):
        out[k] = []
    for i in range(SPAN, len(games) + 1, EVERY):
        span, x = games[i - SPAN : i], at(games[i - 1]["began"])
        for outcome in ("won", "lost", "time limit"):
            out[f"by_start/{outcome}"].append(
                [x, sum(OUTCOMES.get(g["outcome"], g["outcome"]) == outcome for g in span) / SPAN]
            )
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Self-play: %(names)s</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>body{font:14px system-ui,sans-serif;margin:16px 24px;background:#fcfcfb;color:#1a1a19}
h1{font-size:20px;margin:0 0 4px}.now{font-size:16px;margin:0 0 4px}.sub{color:#5f5e57;margin:0 0 20px}
.card{background:#fff;border:1px solid #e1e0d9;border-radius:8px;padding:12px 14px;margin:0 0 18px;max-width:1500px}
.card h2{font-size:17px;margin:0 0 2px}.card p{margin:0 0 8px;color:#5f5e57}.row{display:flex;gap:12px;align-items:flex-start}
.box{position:relative;height:380px;flex:1;min-width:0}.legend{width:250px;flex:none;font-size:13px;line-height:1.3}
.legend div{display:flex;align-items:center;gap:8px;margin:0 0 6px}.legend svg{flex:none}</style>
</head><body><h1>Self-play: %(names)s</h1><p class="now">%(status)s</p><p class="sub">x: %(x)s on every chart (not smoothed; the per-update charts a point every 10 updates). Dashed vertical lines: the ladder moving on (from &rarr; to); dotted pink: the run resumed.%(compare)s</p><div id="g"></div>
<script>
const DATA=%(data)s, SECTIONS=%(sections)s, COLORS=%(colors)s, RUNS=Object.keys(DATA);
Chart.defaults.font.size=13; Chart.defaults.color='#3d3c36';
const STAGES=DATA[RUNS[0]].stages||[], RESUMES=DATA[RUNS[0]].resumes||[], XNAME=DATA[RUNS[0]].x=='games'?'games played':'update';
const XMAX=Math.max(...RUNS.flatMap(r=>Object.values(DATA[r]).filter(Array.isArray).flatMap(p=>p.length&&Array.isArray(p[0])?[p[p.length-1][0]]:[])));
const stageLines={id:'stages',afterDatasetsDraw(ch){const {ctx,chartArea:a,scales:{x}}=ch;ctx.save();
  STAGES.forEach(([u,from,to])=>{const px=x.getPixelForValue(u);if(px<a.left||px>a.right)return;
    ctx.strokeStyle='#5f5e57';ctx.setLineDash([5,4]);ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(px,a.top);ctx.lineTo(px,a.bottom);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle='#3d3c36';ctx.font='12px system-ui';ctx.textAlign='left';ctx.fillText(from+' → '+to,px+4,a.top+12);});
  RESUMES.forEach(([u])=>{const px=x.getPixelForValue(u);if(px<a.left||px>a.right)return;  // the run stopped and resumed here
    ctx.strokeStyle='#b8336a';ctx.setLineDash([2,3]);ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(px,a.top);ctx.lineTo(px,a.bottom);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle='#b8336a';ctx.font='12px system-ui';ctx.textAlign='right';ctx.fillText('resumed',px-4,a.top+12);});
  ctx.restore();}};
SECTIONS.forEach(([title,note,series])=>{
  const share=/^(by_start|games|ladder_blocks|self_blocks)[/]/.test(series[0][0]);
  // self-play in blocks of games when the run logged them (from cb6), else its last 200 at each update
  const blocked=series.some(([k])=>k.startsWith('self_blocks/')&&(DATA[RUNS[0]][k]||[]).length);
  if(series.some(([k])=>k.startsWith('self_blocks/')))series=series.filter(([k])=>k.startsWith('self_blocks/')==blocked);
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<h2>${title}</h2><p>${note}</p><div class="row"><div class="box"><canvas></canvas></div><div class="legend"></div></div>`;
  document.getElementById('g').appendChild(c);
  const ds=[];RUNS.forEach((run,ri)=>series.forEach(([k,name],si)=>{const p=DATA[run][k]||[];if(!p.length)return;
    const total=name=='TOTAL', col=total?'#1a1a19':COLORS[si-(series[0][1]=='TOTAL'?1:0)];
    const dash=(ri?[2,3]:[]).concat(!total&&si-(series[0][1]=='TOTAL'?1:0)>=8?[8,4]:[]);
    const label=(RUNS.length>1?run+': ':'')+name;
    // games in blocks: each block faint, the last 1,000 games' share bold and on top (Chart.js draws lower order last)
    const avg=/^(ladder|self)_blocks[/]/.test(k)?DATA[run][k.replace('/','_avg/')]||[]:[];
    ds.push({label:label+(avg.length?' (blocks of 200)':''),data:p.map(q=>({x:q[0],y:q[1]})),borderColor:avg.length?col+'4d':col,
      backgroundColor:avg.length?col+'4d':col,borderDash:dash,pointRadius:0,borderWidth:total?3.5:avg.length?1.5:2,order:1});
    if(avg.length)ds.push({label:label+' (last 1,000)',data:avg.map(q=>({x:q[0],y:q[1]})),borderColor:col,backgroundColor:col,
      borderDash:dash,pointRadius:0,borderWidth:2.5,order:0});
}));
  if(!ds.length){c.querySelector('.row').outerHTML='<p>not logged for this run</p>';return;}
  c.querySelector('.legend').innerHTML=ds.map(d=>`<div><svg width="28" height="6"><line x1="0" y1="3" x2="28" y2="3" stroke="${d.borderColor}" stroke-width="${Math.min(d.borderWidth,3.5)}" stroke-dasharray="${(d.borderDash||[]).join(' ')}"/></svg>${d.label}</div>`).join('');
  new Chart(c.querySelector('canvas'),{type:'line',data:{datasets:ds},plugins:[stageLines],options:{animation:false,parsing:false,maintainAspectRatio:false,
    interaction:{mode:'nearest',axis:'x',intersect:false},
    plugins:{legend:{display:false},
      tooltip:{callbacks:{title:t=>XNAME+' '+t[0].parsed.x}}},
    scales:{x:{type:'linear',min:0,max:XMAX,title:{display:true,text:XNAME},grid:{color:'#efeee9'}},
            y:Object.assign({grid:{color:'#efeee9'},afterFit:a=>{a.width=64}},share?{min:0,max:1}:{})}}});
});
// live: the page reloads every minute (the server's W&B cache is 60 s), back where it was scrolled to
try { const y = sessionStorage.getItem("y"); if (y) window.scrollTo(0, +y); } catch (e) {}
setTimeout(() => { try { sessionStorage.setItem("y", window.scrollY); } catch (e) {} location.reload(); }, 60000);
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--compare", nargs="*", default=[], help="other runs drawn dashed on the same charts")
    p.add_argument("--port", type=int, default=8767)
    a = p.parse_args()
    names = [a.name, *a.compare]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = {n: history(n) for n in names}
            now = data[a.name].get("status") or {}
            h = now.get("handicap")
            stage = (
                ""
                if h is None
                else f" at {h:g} (its hit points x{h:g})"
                if h < 1
                else " at even strength"
                if h == 1
                else f" outnumbering the learner {h:g}x"
            )
            status = (
                f"Ladder now (update {now['update']}): <b>{now['rung']}</b>{stage}" if now else "Ladder: not logged yet"
            )
            body = (
                PAGE
                % {
                    "names": " vs ".join(names),
                    "data": json.dumps(data),
                    "sections": json.dumps(sections(data[a.name].get("mode", "game"), data[a.name])),
                    "status": status,
                    "x": "games played (all of them, ladder and self-play)" if data[a.name].get("x") == "games" else "the learner's update",
                    "compare": " Dotted: " + ", ".join(names[1:]) + "." if names[1:] else "",
                    "colors": json.dumps(COLORS),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    print(f"http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
