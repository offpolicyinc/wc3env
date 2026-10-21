"""Behavioral cloning on the replay data set (docs/training/5-trainer.md).

    python -m training.train.cloning runs/dataset --out runs/cloning/try1 --steps 200     # locally, a trial
    python -m training.cloud.train --name bc1                                             # on a Modal H100

Lanes: the batch has one row per lane, and each lane reads one game from start to end, `length` steps per
batch, carrying the memory it ended the last chunk with (detached: gradients stop at the chunk's start). A lane
at a game's end starts another random game, with its memory reset. Loader processes each own some lanes, cut
their chunks and pass them over shared memory while the GPU trains on the previous batch.

5% of the replays (both players' files, chosen by replay id) are held out. Every `eval_every` steps: each
head's accuracy on held-out games, played from their start with memory; then a checkpoint (`last.pt`: model,
optimizer, step), from which a rerun resumes. The log is plain lines in `log.txt` next to it.
"""

from __future__ import annotations

import argparse
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from ..policy.model import CRITIC, FRESH, PATCHES, Policy, flash_attn_varlen_func, pack, row_size, upgrade
from .loader import ReplayData
from .track import Track

HELD_OUT_PERCENT = 5


def split(folder: Path) -> tuple[list[Path], list[Path]]:
    """(training files, held-out files), by replay id so both players of a replay are on the same side."""
    train, held = [], []
    for f in sorted(folder.glob("*.npz")):
        replay = f.stem.split("_")[0]
        (held if zlib.crc32(replay.encode()) % 100 < HELD_OUT_PERCENT else train).append(f)
    return train, held


# ---- loader processes


def _lanes(worker, files, rows, length, seed, keep, buffers, free, done, stop):
    """Fill rows [lo, hi) of each batch buffer the main process hands over (`free`: buffer numbers), then say so
    (`done`). Each row is a lane: one game read from start to end, `length` steps per batch."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed + 1000 * worker)
    lo, hi = rows
    state = [None] * (hi - lo)  # per lane: (data, next step, which parts of its plan it is given)
    views = [{k: v.numpy() for k, v in b.items()} for b in buffers]
    while not stop.is_set():
        try:
            slot = free.get(timeout=1)
        except Exception:  # queue.Empty: look at stop again
            continue
        for k in range(hi - lo):
            reset = state[k] is None or state[k][1] >= state[k][0].steps
            if reset:
                state[k] = (ReplayData(files[rng.integers(len(files))]), 0, tuple(bool(rng.random() < p) for p in keep))
            data, start, with_plan = state[k]
            chunk = data.chunk(start, length, with_plan)
            chunk["reset"] = np.bool_(reset)
            state[k] = (data, start + length, with_plan)
            for key, value in chunk.items():
                views[slot][key][lo + k] = value
        done.put(slot)


class Lanes:
    """`lanes` rows from `workers` processes, written straight into `slots` batch buffers in shared memory: a
    process only says which buffer it finished, so nothing is pickled or copied between processes (shipping
    each process's arrays made the main process the bottleneck, slower the more processes there were)."""

    def __init__(
        self, files: list[Path], lanes: int, length: int, workers: int, seed: int, keep: tuple, slots: int = 3
    ):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.stop = ctx.Event()
        sample = ReplayData(files[0]).chunk(0, length)
        sample["reset"] = np.bool_(False)
        self.buffers = [
            {
                k: torch.from_numpy(np.zeros((lanes, *np.shape(v)), np.asarray(v).dtype)).share_memory_()
                for k, v in sample.items()
            }
            for _ in range(slots)
        ]
        self.frees = [ctx.Queue() for _ in range(workers)]
        self.done = ctx.Queue()
        self.finished = [0] * slots
        self.turn, self.workers = 0, workers
        per = [lanes // workers + (w < lanes % workers) for w in range(workers)]
        bounds = np.concatenate([[0], np.cumsum(per)])
        self.procs = []
        for w in range(workers):
            p = ctx.Process(
                target=_lanes,
                args=(
                    w,
                    [str(f) for f in files],
                    (int(bounds[w]), int(bounds[w + 1])),
                    length,
                    seed,
                    keep,
                    self.buffers,
                    self.frees[w],
                    self.done,
                    self.stop,
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        for slot in range(slots):
            self.release(slot)

    def _wait(self) -> int:
        """The next buffer in turn, once every process has filled its rows of it."""
        slot = self.turn
        while self.finished[slot] < self.workers:
            self.finished[self.done.get()] += 1
        self.finished[slot] = 0
        self.turn = (slot + 1) % len(self.buffers)
        return slot

    def release(self, slot: int) -> None:
        for free in self.frees:
            free.put(slot)

    def next(self) -> dict:
        slot = self._wait()
        batch = {k: v.clone() for k, v in self.buffers[slot].items()}
        self.release(slot)
        batch["unit_count_host"] = batch["unit_count"]
        return batch

    def on(self, device):
        """Batches already on `device`: a thread takes each finished buffer, copies it into a page-locked one,
        hands the shared buffer back, and sends the copy to the GPU on a side stream while it trains. Returns
        the function that gives the next batch (with "unit_count_host", its unit counts on the CPU: the packing is
        made there, without waiting for the GPU)."""
        import queue
        import threading

        if device.type != "cuda":
            return self.next
        ready = queue.Queue(maxsize=2)
        stream = torch.cuda.Stream()
        pinned = [{k: torch.empty_like(v).pin_memory() for k, v in self.buffers[0].items()} for _ in range(2)]
        copied = [None, None]

        def fill():
            turn = 0
            while not self.stop.is_set():
                slot = self._wait()
                if copied[turn] is not None:
                    copied[turn].synchronize()  # that page-locked buffer's last copy to the GPU has finished
                host = pinned[turn]
                for k, v in self.buffers[slot].items():
                    host[k].copy_(v)
                self.release(slot)
                with torch.cuda.stream(stream):
                    b = {k: v.to(device, non_blocking=True) for k, v in host.items()}
                    event = torch.cuda.Event()
                    event.record(stream)
                b["unit_count_host"] = host["unit_count"].clone()
                copied[turn] = event
                turn = 1 - turn
                while not self.stop.is_set():
                    try:
                        ready.put((b, event), timeout=1)
                        break
                    except queue.Full:
                        continue

        self.thread = threading.Thread(target=fill, daemon=True)
        self.thread.start()

        def get():
            b, event = ready.get()
            torch.cuda.current_stream().wait_event(event)
            for v in b.values():
                if v.is_cuda:
                    v.record_stream(torch.cuda.current_stream())
            return b

        return get

    def close(self):
        self.stop.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=5)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ---- held-out accuracy


@torch.no_grad()
@torch.no_grad()
def evaluate(model: Policy, files: list[Path], length: int, device, with_plan: bool = True) -> dict:
    """Each head's accuracy on these games, every game played from its start with memory, with its plan (or the
    empty plan)."""
    model.eval()
    games = [ReplayData(f) for f in files]
    total: dict[str, list[int]] = {}
    losses: dict[str, float] = {}
    batches = 0
    state = None
    for start in range(0, max(g.steps for g in games), length):
        chunks = [g.chunk(start, length, with_plan) for g in games]  # a finished game gives padding
        batch = {k: torch.from_numpy(np.stack([c[k] for c in chunks])).to(device) for k in chunks[0]}
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch, state)
            _, parts, h = model.cloning_loss(batch, out)
            counts = model.counts(batch, out, h)
        state = out["state"]
        for k, (right, used) in counts.items():
            r, u = total.setdefault(k, [0, 0])
            total[k] = [r + right, u + used]
        for k, v in parts.items():
            losses[k] = losses.get(k, 0.0) + v
        batches += 1
    model.train()
    result = {k: r / max(u, 1) for k, (r, u) in total.items()}
    result["loss"] = sum(v for k, v in losses.items() if k != "value") / batches
    result["value_loss"] = losses["value"] / batches
    return result


# ---- the loop


def train(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.txt").open("a", buffering=1)
    track = Track(
        out.name, {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}, "cloning"
    )

    def say(line: str):
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {line}", flush=True)
        log.write(f"{stamp} {line}\n")
        track.line(line)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_files, held = [], []
    for folder in args.data:
        t, h = split(Path(folder))
        train_files, held = train_files + t, held + h
    held.sort(key=lambda f: zlib.crc32(f.name.encode()))  # every folder in the evaluation, not the first one only
    held = held[: args.eval_games]
    if args.only_wins:
        train_files = [f for f in train_files if int(np.load(f)["result"]) == 1]
        say(f"only won games: {len(train_files)} files")
    say(f"{len(train_files)} training files, {len(held)} held-out files for accuracy; device {device}")

    model = Policy().to(device)
    # the learner's two speed-ups: the step transformer on real tokens only (packed, FlashAttention-2's variable-length
    # kernel; without it every step is padded to the fullest one's row size) and its blocks compiled (small kernels fused)
    varlen = args.pack and device.type == "cuda" and flash_attn_varlen_func is not None
    if args.compile and device.type == "cuda":
        model.compile_blocks(None if varlen else False)  # packed: the token count changes with every piece
    say(
        f"packed attention {'on' if varlen else 'off'}, compiled blocks {'on' if args.compile and device.type == 'cuda' else 'off'}"
    )
    lanes = Lanes(train_files, args.batch, args.length, args.workers, args.seed, (args.keep_order, args.keep_made))
    next_batch = lanes.on(device)
    batch = next_batch()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n and "choice" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n and "choice" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95)
    )
    step = 0
    checkpoint = out / "last.pt"
    if args.init and not checkpoint.exists():  # start from another run's weights (a fine-tune), fresh optimizer
        missing, unexpected = model.load_state_dict(
            upgrade(torch.load(args.init, map_location=device)["model"]), strict=False
        )
        if unexpected or any(not k.startswith(FRESH) for k in missing):
            raise ValueError(f"{args.init} does not match the model: missing {missing}, unexpected {unexpected}")
        say(f"weights from {args.init}")
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        step = saved["step"]
        say(f"resumed at step {step}")
    say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters; batch {args.batch} x {args.length} steps")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        progress = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    state = None
    sums: dict[str, float] = {}
    counted, t0, last_eval = 0, time.time(), time.time()
    try:
        while step < args.steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            # --micro lanes at a time through the GPU, their gradients added up: the same step on a smaller GPU
            micro = args.micro or args.batch
            units = batch.pop("unit_count_host").numpy()
            counts = {k: v.sum() for k, v in model.uses(batch).items()}  # the whole batch's: the pieces add up to it
            parts, states = {}, []
            for lo in range(0, args.batch, micro):
                piece = {k: v[lo : lo + micro] for k, v in batch.items()}
                count = units[lo : lo + micro].reshape(-1)
                size = row_size(int(count.max()))
                packing = pack(count, size, device) if varlen else None
                # the share of the padded tokens that are real (packed: the only ones computed)
                parts["tokens_kept"] = parts.get("tokens_kept", 0.0) + (np.minimum(count, size) + PATCHES + 1).sum() / (
                    (size + PATCHES + 1) * units.size
                )
                with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    out_ = model(
                        piece, None if state is None else [h[lo : lo + micro] for h in state], rows=size, pack=packing
                    )
                    loss, piece_parts, _ = model.cloning_loss(piece, out_, args.value_weight, counts)
                loss.backward()
                states.append([h.detach() for h in out_["state"]])
                for k, v in piece_parts.items():
                    parts[k] = parts.get(k, 0.0) + v
            state = [torch.cat(hs) for hs in zip(*states)]
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            waited = time.time()
            batch = next_batch()
            sums["wait_s"] = sums.get("wait_s", 0.0) + time.time() - waited  # time the loop sat waiting for data
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            sums["grad_norm"] = sums.get("grad_norm", 0.0) + float(norm)
            counted += 1
            if step % args.log_every == 0:
                rate = counted * args.batch * args.length / (time.time() - t0)
                heads = " ".join(f"{k} {v / counted:.3f}" for k, v in sums.items())
                say(f"step {step} lr {lr_at(step):.2e} {rate:.0f} steps/s | {heads}")
                sums, counted, t0 = {}, 0, time.time()
            if step % args.eval_every == 0 or step == args.steps or time.time() - last_eval > args.eval_minutes * 60:
                if held:
                    acc = evaluate(model, held, args.length, device)
                    say(f"held-out at step {step}: " + " ".join(f"{k} {v:.3f}" for k, v in acc.items()))
                    empty = evaluate(model, held, args.length, device, with_plan=False)  # how much the plan is used
                    say(
                        f"held-out at step {step}, empty plan: "
                        + " ".join(f"{k} {empty[k]:.3f}" for k in ("type", "argument", "loss"))
                    )
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "critic": CRITIC},
                    checkpoint,
                )
                if args.snapshot_every and step % args.snapshot_every == 0:  # kept, to pick the best by games later
                    torch.save({"model": model.state_dict(), "step": step, "critic": CRITIC}, out / f"step{step}.pt")
                if args.on_save:
                    args.on_save()
                say(f"saved {checkpoint}")
                last_eval, t0, sums, counted = time.time(), time.time(), {}, 0
    finally:
        lanes.close()
        log.close()
        track.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="+", help="folders of data set files (pro replays, recorded bot games), used together")
    p.add_argument("--out", required=True, help="the run's folder: log.txt and last.pt")
    p.add_argument("--steps", type=int, default=10000, help="optimizer steps (one pass is ~2,400 at 32 x 64)")
    p.add_argument("--batch", type=int, default=32, help="lanes")
    p.add_argument(
        "--micro", type=int, default=0, help="lanes through the GPU at once, gradients added up (an L4: 4 or 8)"
    )
    p.add_argument("--length", type=int, default=64, help="steps per chunk")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--value-weight", type=float, default=0.1)
    # each game's plan in two parts, each kept or left out on its own (AlphaStar's 0.8 and 0.5), so the policy plays
    # either alone or none: the build order, and what was made over the game (heroes, unit types, research levels)
    p.add_argument("--keep-order", type=float, default=0.8, help="the share of games given their build order")
    p.add_argument("--keep-made", type=float, default=0.5, help="the share given what was made over the game")
    p.add_argument("--only-wins", action="store_true", help="train on the winners' files only (the final fine-tune)")
    p.add_argument("--init", help="a checkpoint to start from (its weights only)")
    p.add_argument("--workers", type=int, default=6, help="loader processes")
    p.add_argument("--eval-games", type=int, default=32, help="held-out files for accuracy")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-minutes", type=float, default=15)
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="also keep the weights as step<N>.pt every so many steps (0: only last.pt)",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="packed attention (needs flash-attn or PyTorch's varlen_attn)",
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the transformer blocks and SSM layers compiled",
    )
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    a.on_save = None
    train(a)
