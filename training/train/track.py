"""Charts on wandb (wandb.ai), from a run's own log lines: cloning's ("step ...", "held-out at step ...") and the
self-play learner's ("update ..."). Only when WANDB_API_KEY is set (on Modal: the secret "wandb", which
training/cloud/train.py and selfplay.py attach when it exists) and wandb is installed; otherwise nothing.

    python -m training.train.track runs/bc11/log.txt --name bc11     # a finished run's log, uploaded afterwards

Keys: train/<head> (training losses, lr, steps/s, wait_s, grad_norm), held/<head> and held_empty/<head> (held-out,
with each game's plan and with the empty plan); self-play: play/<field> and games/<opponent>_won|lost|time.
"""

from __future__ import annotations

import argparse
import os
import re

PROJECT = "wc3"
STEP = re.compile(r"step (\d+) lr ([0-9.e+-]+) (\d+) steps/s \| (.*)")
HELD = re.compile(r"held-out at step (\d+)(, empty plan)?: (.*)")
UPDATE = re.compile(r"update (\d+) (\d+) steps/s games (\d+) ladder \S+ \((\d+)/\d+\) \| (.*)")
OUTCOME = re.compile(r"([\w/-]+) won ([0-9.]+) lost ([0-9.]+) time ([0-9.]+)")  # (attack-move: a hyphen)
PAIRS = re.compile(r"([\w/]+) (-?[0-9.]+(?:e[+-]?\d+)?)\b")


def parse(line: str) -> tuple[int, dict] | None:
    """A log line's step and charted values, or None for a line with none."""
    if m := STEP.search(line):
        values = {"train/lr": float(m[2]), "train/steps_per_s": float(m[3])}
        values.update({f"train/{k}": float(v) for k, v in PAIRS.findall(m[4])})
        return int(m[1]), values
    if m := HELD.search(line):
        prefix = "held_empty" if m[2] else "held"
        return int(m[1]), {f"{prefix}/{k}": float(v) for k, v in PAIRS.findall(m[3])}
    if m := UPDATE.search(line):
        values = {"play/steps_per_s": float(m[2]), "play/games": float(m[3]), "play/ladder": float(m[4])}
        parts = m[5].split(" | ")
        for who, won, lost, time in OUTCOME.findall(parts[0]):
            who = who.replace("/", "_")
            values.update({f"games/{who}_won": float(won), f"games/{who}_lost": float(lost), f"games/{who}_time": float(time)})
        values.update({f"play/{k.replace('/', '_')}": float(v) for part in parts[1:] for k, v in PAIRS.findall(part)})
        return int(m[1]), values
    return None


def modal_secrets() -> list:
    """The Modal secret "wandb" (WANDB_API_KEY) to attach to a run's function, when it exists: [] otherwise."""
    import modal

    try:
        secret = modal.Secret.from_name("wandb")
        secret.hydrate()
        return [secret]
    except Exception:  # not created: the run goes on without charts
        return []


class Track:
    """A run on wandb, fed the run's log lines (line()); does nothing without WANDB_API_KEY or wandb."""

    def __init__(self, name: str, config: dict | None = None, group: str = ""):
        self.run = None
        if not os.environ.get("WANDB_API_KEY"):
            return
        try:
            import wandb
        except ImportError:
            return
        # the run's name is its id too: a run restarted on Modal (retries) goes on in the same charts
        self.run = wandb.init(project=PROJECT, name=name, id=name, resume="allow", group=group or None, config=config)
        try:  # its lines colored in the charts view
            workspace()
        except Exception as e:
            print(f"wandb view: {e}", flush=True)

    def line(self, line: str) -> None:
        if self.run is None:
            return
        parsed = parse(line)
        if parsed:
            step, values = parsed
            try:
                self.run.log(values, step=step)
            except Exception as e:  # a chart is never worth the run
                print(f"wandb: {e}", flush=True)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


VIEW = "https://wandb.ai/off-policy-inc/wc3?nw=zxejrmng5r7"  # the project's charts view (workspace() remakes it)
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563")  # the dashboard's


def workspace() -> str:
    """The project's charts as the local dashboard draws them (training/cloud/dashboard.py): a few panels of several
    lines each, a color per head, held-out with the plan solid and with the empty plan dashed. wandb keys a line's
    color, dash and title by run and metric, so they are set for every run in the project: a new run calls this
    (Track) to get its own. Saved into VIEW; returns its URL."""
    import wandb
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    runs = [r.id for r in wandb.Api().runs(f"off-policy-inc/{PROJECT}")]

    def plot(title, keys, dashed=(), colors=None, **kw):
        colors = colors or {k: COLORS[i % len(COLORS)] for i, k in enumerate(keys)}
        name = lambda k: k.split("/", 1)[1] + (" (empty plan)" if k in dashed else "")  # noqa: E731
        series = [(f"{r}:{k}", k) for r in runs for k in keys]
        return wr.LinePlot(title=title, x="Step", y=list(keys), legend_position="south",
                           line_titles={s: (name(k) if len(runs) == 1 else f"{s.split(':')[0]} {name(k)}") for s, k in series},
                           line_colors={s: colors[k] for s, k in series},
                           line_marks={s: "dashed" if k in dashed else "solid" for s, k in series}, **kw)  # fmt: skip

    smooth = {"smoothing_type": "exponential", "smoothing_factor": 0.6}
    heads = ("type", "argument", "actor", "target")
    held = [f"{p}/{k}" for k in heads for p in ("held", "held_empty")]
    by_head = {k: COLORS[heads.index(k.split("/")[1])] for k in held}  # a head's color, with the plan or without
    cloning = ws.Section(name="Cloning", is_open=True, layout_settings=ws.SectionLayoutSettings(columns=2, rows=4), panels=[
        plot("Training loss per head (lower is better)", [f"train/{k}" for k in ("type", "argument", "actor", "target", "region", "size", "cell")], **smooth),
        plot("Held-out accuracy: plan (solid) vs empty plan (dashed)", held, dashed=[k for k in held if k.startswith("held_empty")], colors=by_head),
        plot("Held-out loss: plan (solid) vs empty plan (dashed)", ["held/loss", "held_empty/loss"], dashed=["held_empty/loss"],
             colors={"held/loss": COLORS[0], "held_empty/loss": COLORS[0]}),
        plot("Held-out: type given an action", ["held/type_given_act"]),
        plot("Speed: game steps learned per second", ["train/steps_per_s"], **smooth),
        plot("Loader wait per update (s)", ["train/wait_s"], colors={"train/wait_s": COLORS[3]}, **smooth),
        plot("Gradient norm (before clipping)", ["train/grad_norm"], colors={"train/grad_norm": COLORS[1]}, **smooth),
        plot("Learning rate", ["train/lr"], colors={"train/lr": COLORS[4]}),
    ])  # fmt: skip
    selfplay = ws.Section(name="Self-play", is_open=True, layout_settings=ws.SectionLayoutSettings(columns=2, rows=4), panels=[
        wr.LinePlot(title="Win share per opponent", x="Step", metric_regex="games/.*_won", legend_position="south"),
        wr.LinePlot(title="Loss share per opponent", x="Step", metric_regex="games/.*_lost", legend_position="south"),
        plot("Ladder rung", ["play/ladder"]),
        plot("Game length (min) and its limit", ["play/minutes", "play/limit"], dashed=()),
        plot("Gold per minute", ["play/gold_min"], colors={"play/gold_min": COLORS[3]}),
        plot("Speed: steps per second", ["play/steps_per_s"]),
        wr.LinePlot(title="Learner stats", x="Step", metric_regex="play/(kl|entropy|loss.*|grad.*|value.*)", legend_position="south"),
    ])  # fmt: skip
    view = ws.Workspace.from_url(VIEW)
    view.sections = [cloning, selfplay]
    view.settings = ws.WorkspaceSettings(max_runs=10)
    view.save()
    return VIEW


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", help="a run's log.txt")
    p.add_argument("--name")
    p.add_argument("--group", default="")
    p.add_argument("--workspace", action="store_true", help="(re)make the project's charts view instead (pip install wandb-workspaces)")
    a = p.parse_args()
    if a.workspace:
        print(workspace())
        return
    if not (a.log and a.name):
        p.error("a log and --name")
    track = Track(a.name, group=a.group)
    if track.run is None:
        raise SystemExit("set WANDB_API_KEY (and pip install wandb) first")
    with open(a.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            track.line(line)
    track.close()


if __name__ == "__main__":
    main()
