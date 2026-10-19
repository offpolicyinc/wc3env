"""Every head of every game sampled on the GPU in one pass, from masks prepared ahead as tables.

The heads are sampled in order and each one's mask can depend on the choices before it (the argument mask on the
type, the actor mask on the type and argument). Asking the CPU for each head's mask cost a GPU -> CPU round trip
per head; with many actor processes on one GPU each of those waited for the others' turns
(docs/training/9-benchmarks.md). So the masks of every choice that can come up arrive with the observation as
tables (wc3hook/masks.c: the type mask, each allowed type's argument mask, each (type, argument)'s actor mask), and
the GPU looks them up:

    pack(tables, seats, keys)   the batch's tables in fixed shapes, on the CPU (policy/native.tables makes a block's)
    unpack(packed)              them on the device as the heads read them
    sample(model, out, tables, unit_count, temperature)
                                all games' heads in order with those masks, on the device (no copy to the CPU,
                                no shape that depends on the data: policy/agent.py replays it as a CUDA graph)

Which heads a type uses comes from actions.head_rules; a head with no legal choice makes the decision the "failed"
noop it became.
"""

from __future__ import annotations

import numpy as np
import torch

from .actions import AIMS, ANY_ARGUMENT, TYPES, head_rules
from .features import MAX_UNITS
from .masks import argument_ids

POINT, UNIT = AIMS.index("point"), AIMS.index("unit")
N_ARGS = len(argument_ids()) + 1
HEADS = ("type", "argument", "aim", "queued", "actor", "same_type", "size", "target", "region", "cell")
UNUSED = {h: -1 if h in ("actor", "target", "region", "cell") else 0 for h in HEADS}  # a head a type does not use
NO_KEY = np.iinfo(np.int64).max  # an actor-mask key slot with no mask (after every real key: they stay sorted)


def actor_keys(seat_tables: list[dict]) -> int:
    """How many actor masks the tables hold."""
    return sum(len(t["actor_keys"]) for t in seat_tables)


def pack(seat_tables: list[dict], seats: int, keys: int) -> dict[str, np.ndarray]:
    """The tables of a batch's games in fixed shapes, on the CPU: `seats` rows (those past the tables' have no legal
    choice) and `keys` actor-mask slots (at least actor_keys(seat_tables)). type [S, T] bool; argument [S, T, A/8]
    each allowed type's argument mask, packed bits; actor_keys [K] (s * T * N_ARGS + type * N_ARGS + argument,
    sorted, NO_KEY past the real ones) and actor_rows [K, MAX_UNITS/8] packed bits."""
    t_count, width = len(TYPES), (N_ARGS + 7) // 8
    types = np.zeros((seats, t_count), bool)
    argument = np.zeros((seats, t_count, width), np.uint8)
    for b, t in enumerate(seat_tables):
        types[b] = t["type"]
        if len(t["argument_types"]):
            argument[b, t["argument_types"].astype(np.int64)] = t["argument"]
    out_keys = np.full(keys, NO_KEY, np.int64)
    out_rows = np.zeros((keys, MAX_UNITS // 8), np.uint8)
    if seat_tables:
        real = np.concatenate([t["actor_keys"] + b * t_count * N_ARGS for b, t in enumerate(seat_tables)])
        order = np.argsort(real, kind="stable")
        out_keys[: len(real)] = real[order]
        out_rows[: len(real)] = np.concatenate([t["actor_rows"] for t in seat_tables])[order]
    return {"type": types, "argument": argument, "actor_keys": out_keys, "actor_rows": out_rows}


def unpack(packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """pack()'s tables, on the device, as sample() reads them: the bits as bools."""
    return {"type": packed["type"], "argument": _unpack(packed["argument"])[..., :N_ARGS],
            "actor_keys": packed["actor_keys"], "actor_rows": _unpack(packed["actor_rows"])}  # fmt: skip


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """np.packbits' bytes (most significant bit first) back to bools: [..., n] uint8 -> [..., 8n] bool."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed[..., None] >> shifts) & 1).bool().flatten(-2)


_DEVICE_RULES: dict = {}


def _device_rules(device) -> dict:
    if device not in _DEVICE_RULES:
        r = head_rules()
        _DEVICE_RULES[device] = {
            k: torch.as_tensor(r[k], device=device)
            for k in ("has_argument", "has_actor", "grouped", "queueable", "aim")
        } | {"any_argument": torch.tensor([t in ANY_ARGUMENT for t in TYPES], device=device)}
    return _DEVICE_RULES[device]


@torch.no_grad()
def sample(model, out, batch: dict, unit_count: torch.Tensor, temperature: float, generator=None) -> dict:
    """Each row's choices, log-probability and failed head, with masks from `batch` (unpack()). Returns tensors on
    the device: {head: [B]} choices as the data set stores them, `log_prob` [B], `failed` [B] (the index into
    HEADS of the head that had no legal choice, -1 for none). `generator`: None for the device's default (the one a
    CUDA graph replays)."""
    dev = unit_count.device
    rules = _device_rules(dev)
    n = unit_count.shape[0]
    rows = torch.arange(n, device=dev)
    log_prob = torch.zeros(n, device=dev)
    failed = torch.full((n,), -1, dtype=torch.long, device=dev)
    c: dict[str, torch.Tensor] = {}
    kind = torch.zeros(n, dtype=torch.long, device=dev)
    type_logits = None

    def draw(logits, mask, use, temp):
        """Picks where `use`, from logits masked by `mask` ([B, K] bool, or None): picks, their log-probs, and
        which used rows had no legal choice."""
        logits = logits.float()
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        empty = use & ~mask.any(-1)
        live = use & ~empty
        safe = mask | ~live[:, None]  # rows not sampled keep a harmless mask (never all -inf)
        masked = logits.masked_fill(~safe, float("-inf"))
        probs = (masked / temp).softmax(-1)
        # a draw from probs: the largest p_i / E_i with E_i ~ Exp(1) (the Gumbel-max trick, without its logarithms);
        # no data-dependent shapes or host syncs, unlike multinomial
        noise = torch.empty_like(probs).exponential_(generator=generator).clamp_(min=1e-30)
        pick = (probs / noise).argmax(-1)
        lp = torch.log_softmax(masked, -1).gather(1, pick[:, None])[:, 0]
        return pick, torch.where(live, lp, torch.zeros_like(lp)), live, empty

    flow = model._flow(out)
    name, logits = next(flow)
    while True:
        logits = logits[:, 0]
        h = HEADS.index(name)
        ok = (kind != 0) & (failed < 0)
        pick = torch.full((n,), UNUSED[name], dtype=torch.long, device=dev)
        mask, use, temp = None, ok, temperature
        if name == "type":
            mask, use, temp = batch["type"], torch.ones(n, dtype=torch.bool, device=dev), 1.0
            type_logits = logits.float()
        elif name == "argument":
            use = ok & rules["has_argument"][kind]
            pick = torch.where(ok, torch.zeros_like(pick), pick)  # a type without an argument: 0
            mask = batch["argument"][rows, kind]
        elif name == "aim":
            mask = rules["aim"][kind]
        elif name == "queued":
            use = ok & rules["has_actor"][kind] & rules["queueable"][kind]
            pick = torch.where(ok, torch.zeros_like(pick), pick)
        elif name == "actor":
            use = ok & rules["has_actor"][kind]
            mask = _actor_masks(batch, kind, c["argument"], rules, n)
        elif name in ("same_type", "size"):
            use = ok & rules["has_actor"][kind] & rules["grouped"][kind]
            pick = torch.where(ok, torch.zeros_like(pick), pick)
        elif name == "target":
            use = ok & (c["aim"] == UNIT)
            mask = torch.arange(logits.shape[-1], device=dev)[None] < unit_count[:, None]
        elif name in ("region", "cell"):
            use = ok & (c["aim"] == POINT)
        drawn, lp, live, empty = draw(logits, mask, use, temp)
        pick = torch.where(live, drawn, pick)
        if name != "type":
            failed = torch.where(empty, torch.full_like(failed, h), failed)
            pick = torch.where(empty, torch.full_like(pick, UNUSED[name]), pick)
        log_prob = log_prob + lp
        c[name] = pick
        if name == "type":
            kind = pick
        try:
            name, logits = flow.send(pick[:, None])
        except StopIteration:
            break
    # a decision whose action could not be finished is the noop it became, with the noop's probability
    noop = torch.log_softmax(type_logits.masked_fill(~batch["type"], float("-inf")), -1)[:, 0]
    log_prob = torch.where(failed >= 0, noop, log_prob)
    return {**c, "log_prob": log_prob, "failed": failed}


def _actor_masks(batch, kind, argument, rules, n) -> torch.Tensor:
    """[B, MAX_UNITS]: each row's actor mask for its type and argument (all false when the tables have none)."""
    dev = kind.device
    arg = torch.where(rules["any_argument"][kind], torch.zeros_like(argument), argument.clamp(min=0))
    want = (torch.arange(n, device=dev) * len(TYPES) + kind) * N_ARGS + arg
    keys = batch["actor_keys"]
    at = torch.searchsorted(keys, want).clamp(max=keys.numel() - 1)
    found = keys[at] == want
    return batch["actor_rows"][at] & found[:, None]
