"""Does the clone (bc10) know to send its orders into the enemy's half, or only never get there? The data it learned
from sends about a third of its point orders there (training/bench/data_attacks.py); played, bc10 sent 0.5%.

    python -m modal run training/bench/clone_check.py          # on the data: what bc10 predicts at the players' orders
    python -m training.bench.clone_check play                  # in its own games against the easy AI (locally)

On the data (teacher forcing: every game from its start with memory, the players' own earlier choices fed in, as in
cloning's held-out accuracy): at each point order, the share of the players' targets in the enemy's half, against
the probability bc10's region head puts on the enemy's half, and its region accuracy, by game minute; on the held-out
games (never trained on) and on a sample it trained on. In its own games: the share of its point orders into the
enemy's half by game minute, the same measure as the players'.
"""

from __future__ import annotations

import sys

import modal

from training.bench.common import FLASH_ATTN_WHEEL, image

app = modal.App("wc3-clone-check", image=image(f"pip install {FLASH_ATTN_WHEEL}"))
VOLUME = modal.Volume.from_name("wc3-dataset")
BUCKETS = (0, 3, 6, 10, 15, 30)  # game minutes
TRAINED_SAMPLE = 32
GROUP = 32


def bucket(minute: float) -> str:
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        if minute < hi:
            return f"{lo}-{hi}"
    return f"{BUCKETS[-1]}+"


def region_halves(d):
    """Which of the 64 regions (8 x 8 patches of the 128 x 128 map) have their middle in the enemy's half."""
    import numpy as np

    from training.bench.data_attacks import geometry, in_enemy_half
    from training.policy.features import MAP, PATCH

    places = geometry(d)
    if places is None:
        return None
    side = MAP // PATCH
    middles = [(r // side * PATCH + PATCH // 2) * MAP + (r % side * PATCH + PATCH // 2) for r in range(side * side)]
    return in_enemy_half(np.array(middles), *places)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3600, volumes={"/dataset": VOLUME})
def teacher_forced(folder: str) -> dict:
    import random
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch

    from training.policy.features import STEP_MS
    from training.policy.model import load
    from training.train.cloning import split
    from training.train.loader import ReplayData

    device = torch.device("cuda")
    model = load("/dataset/runs/bc10/step8k.pt", device)
    trained, held = split(Path(f"/dataset/{folder}"))
    random.Random(0).shuffle(trained)
    out = {}
    for name, files in (("held-out", held), ("trained on", trained[:TRAINED_SAMPLE])):
        sums = defaultdict(
            lambda: np.zeros(6)
        )  # bucket -> [orders, label enemy, mass enemy, right, right on enemy-half labels, enemy labels]
        played = 0
        for at in range(0, len(files), GROUP):  # GROUP games at once (all of pro2's held-out games ran out of memory)
            games = [ReplayData(f) for f in files[at : at + GROUP]]
            halves = [region_halves(g.d) for g in games]
            keep = [i for i, h in enumerate(halves) if h is not None]
            if not keep:
                continue
            games, halves = [games[i] for i in keep], torch.tensor(np.stack([halves[i] for i in keep]), device=device)
            played += len(games)
            state = None
            length = 64
            for start in range(0, max(g.steps for g in games), length):
                chunks = [g.chunk(start, length) for g in games]
                batch = {k: torch.from_numpy(np.stack([c[k] for c in chunks])).to(device) for k in chunks[0]}
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    out_ = model(batch, state)
                    h = model.heads(out_, batch)
                state = out_["state"]
                used = model.uses(batch)["region"]  # point orders: steps whose label uses the region head
                probs = h["region"].float().softmax(-1)  # [B, T, 64]
                mass = (probs * halves[:, None, :].float()).sum(-1)
                label = batch["region"].long().clamp(min=0)
                label_enemy = halves.long().gather(1, label.reshape(len(games), -1)).reshape(label.shape)
                right = probs.argmax(-1) == label
                for b, t in zip(*torch.nonzero(used, as_tuple=True)):
                    b, t = int(b), int(t)
                    s = sums[bucket((start + t) * STEP_MS / 60000)]
                    enemy = bool(label_enemy[b, t])
                    s += [1, enemy, float(mass[b, t]), bool(right[b, t]), enemy and bool(right[b, t]), enemy]
        out[name] = {
            "games": played,
            "buckets": {
                k: v.tolist() for k, v in sorted(sums.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+")))
            },
        }
    return out


MAKES_GAMES = 100
MAKE_GROUPS = {"altar": ("oalt", "halt", "eate", "uaod"), "barracks": ("obar", "hbar", "eaom", "usep")}


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3600, volumes={"/dataset": VOLUME})
def makes(folder: str, checkpoint: str) -> dict:
    """At the players' build / train / buy orders (teacher forced, held-out games), with their own plan and with the
    empty plan: the clone's probability of the player's type, and of its argument given the type; top-1 of each."""
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch

    from training.policy.actions import TYPES
    from training.policy.masks import _id, argument_ids
    from training.policy.model import load
    from training.policy.plan import hero_arguments
    from training.train.cloning import split
    from training.train.loader import ReplayData

    device = torch.device("cuda")
    model = load(f"/dataset/runs/{checkpoint}", device)
    ids = argument_ids()
    group_of = {ids[_id(c)]: g for g, codes in MAKE_GROUPS.items() for c in codes}
    group_of.update({a: "hero" for a in hero_arguments()})
    kinds = torch.tensor([TYPES.index(t) for t in ("build", "train", "buy")], device=device)
    _, held = split(Path(f"/dataset/{folder}"))
    held = held[:MAKES_GAMES]
    out = {}
    for with_plan in (True, False):
        sums = defaultdict(lambda: np.zeros(5))  # group -> [orders, p type, top-1 type, p argument, top-1 argument]
        for at in range(0, len(held), GROUP):
            games = [ReplayData(f) for f in held[at : at + GROUP]]
            state = None
            for start in range(0, max(g.steps for g in games), 64):
                chunks = [g.chunk(start, 64, with_plan) for g in games]
                batch = {k: torch.from_numpy(np.stack([c[k] for c in chunks])).to(device) for k in chunks[0]}
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    o = model(batch, state)
                    terms = model.terms(batch, o, model.heads(o, batch))
                state = o["state"]
                tl, tk, _ = terms["type"]
                al, ak, _ = terms["argument"]
                tp, ap = tl.float().softmax(-1), al.float().softmax(-1)
                tk, ak = tk.long().clamp(min=0), ak.long().clamp(min=0)
                p_type = tp.gather(-1, tk[..., None])[..., 0]
                p_arg = ap.gather(-1, ak[..., None])[..., 0]
                top_type, top_arg = tp.argmax(-1) == tk, ap.argmax(-1) == ak
                sel = torch.isin(tk, kinds) & (ak > 0) & model.uses(batch)["argument"]
                for b, t in zip(*torch.nonzero(sel, as_tuple=True)):
                    row = [1, float(p_type[b, t]), bool(top_type[b, t]), float(p_arg[b, t]), bool(top_arg[b, t])]
                    sums["all"] += row
                    g = group_of.get(int(ak[b, t]))
                    if g:
                        sums[g] += row
            print(
                f"{folder} {'plan' if with_plan else 'empty'}: {min(at + GROUP, len(held))}/{len(held)} games",
                flush=True,
            )
        out["with plan" if with_plan else "empty plan"] = {k: v.tolist() for k, v in sums.items()}
    return out


def table(name: str, result: dict) -> None:
    for part, r in result.items():
        print(
            f"{name}, {part} ({r['games']} player-games): minutes | point orders | players' targets in enemy half | "
            "bc10's probability on enemy half | region top-1 (all / on enemy-half targets)"
        )
        for b, (n, enemy, mass, right, right_enemy, enemies) in r["buckets"].items():
            print(
                f"  {b:6s} {int(n):7d}   {enemy / n:6.1%}   {mass / n:6.1%}   {right / n:6.1%} / {right_enemy / max(enemies, 1):6.1%}"
            )


def play(games: int = 8, minutes: float = 15.0) -> None:
    """bc10's own games against the easy AI (locally): its point orders into the enemy's half, by game minute."""
    from collections import defaultdict

    import numpy as np

    from training.bench.data_attacks import geometry, in_enemy_half
    from training.play.evaluate import evaluate
    from training.policy.features import STEP_MS

    sums = defaultdict(lambda: np.zeros(2))
    for g in evaluate("runs/bc10/step8k.pt", games, 0, minutes, 1.0, False, "runs/plans.npz",
                      "Maps/FrozenThrone/(2)EchoIsles.w3x", record=True):  # fmt: skip
        d = g.get("data")
        places = geometry(d) if d is not None else None
        if places is None:
            continue
        point, kind = np.asarray(d["point"]), np.asarray(d["type"])
        for t in np.flatnonzero((kind > 0) & (point >= 0)):
            sums[bucket(t * STEP_MS / 60000)] += [1, bool(in_enemy_half([point[t]], *places)[0])]
        print(f"game {g.get('race')} vs {g.get('opponent')}: {g.get('result')}", flush=True)
    print("bc10 against the easy AI: minutes | point orders | into the enemy's half")
    for b, (n, enemy) in sorted(sums.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+"))):
        print(f"  {b:6s} {int(n):7d}   {enemy / n:6.1%}")


@app.local_entrypoint()
def make_orders(checkpoint: str = "bc10/last.pt"):
    folders = ("pro2", "bots3k")
    for folder, result in zip(folders, makes.starmap([(f, checkpoint) for f in folders])):
        for part, groups in result.items():
            print(f"{folder}, {part}: order | count | p(type) top-1 | p(argument | type) top-1")
            for g, (n, pt, tt, pa, ta) in groups.items():
                print(f"  {g:9s} {int(n):7d}   {pt / n:5.1%} {tt / n:5.1%}   {pa / n:5.1%} {ta / n:5.1%}")


@app.local_entrypoint()
def main():
    for folder in ("pro2", "bots3k"):
        table(folder, teacher_forced.remote(folder))


if __name__ == "__main__" and sys.argv[1:] == ["play"]:
    sys.path[:0] = ["src", "."]
    play()
