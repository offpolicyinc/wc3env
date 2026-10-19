"""The policy network (docs/training/3-model.md): a step transformer, memory over time, the action heads.

    model = Policy()
    out = model(batch, state)               # batch from training/train/loader.py: [B, T, ...] arrays as tensors
    loss, parts, logits = model.cloning_loss(batch, out)
    state = out["state"]                    # the memory at the chunk's end, for the game's next chunk

Each step: unit rows (<= MAX_UNITS), 64 map patches (16 x 16 cells each) and one scalar token go through a
transformer; the scalar token's output is the step vector. The scalar token also carries the game's plan
(training/policy/plan.py: heroes, build order, unit mix), the strategy the policy is asked to play. Over time: SSM layers and one attention layer over a
window of recent steps. Heads, each seeing the choices before it (during training the label's):

    type      what kind of action, or noop                len(TYPES)
    argument  unit/structure/item/upgrade type, ability, order     argument vocabulary
    aim       none, a point or a unit                     3
    queued    Shift: after the units' current orders      2
    actor     pointer over the unit rows                  MAX_UNITS
    same_type, size                                       2, len(SIZES)
    target    pointer over the unit rows                  MAX_UNITS
    region    pointer over the 64 patches                 64
    cell      a cell inside the region                    256
    value     how the game will end                       1

Timing, as Pluto: a decision every step, noop among the types. (Cloned per-step noops can collapse to always
noop, VPT's case; ours did not: the model's chance of acting matches the players'.)

Masks from the data set each head's illegal choices to -inf. The SSM is a plain PyTorch diagonal, input-gated
recurrence (Mamba-like, without the fused kernel), fine for development; a CUDA kernel can replace it later with
the same weights.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

try:  # FlashAttention-2's variable-length kernel: the learner's step transformer on real tokens only (packed)
    from flash_attn import flash_attn_varlen_func
except ImportError:  # no flash-attn here (a CPU, the inference image): the padded path
    flash_attn_varlen_func = None

from wc3env.binary import fourcc

from .actions import AIMS, SIZES, TYPES, head_rules, reference
from .features import (
    COLUMN,
    GOLD_MINE,
    MAP,
    MAP_CHANNELS,
    MAX_UNITS,
    OWN_BASE,
    OWNERS,
    PATCH,
    UNIT_FLOATS,
    ability_ids,
    buff_ids,
    scalar_size,
    type_ids,
)
from .masks import argument_ids
from .plan import PLAN_HEROES, PLAN_ORDER

PRESENCE = 32  # each presence vector's embedding (AlphaStar's 32)

PATCHES = (MAP // PATCH) ** 2  # 64
ROW_SIZES = (64, 96, 128, MAX_UNITS)  # unit rows through the transformer: the fullest step's count rounded up to one
PER_GAME = ("static_map", "plan_heroes", "plan_order", "plan_mix")  # batch entries [B, ...], not [B, T, ...]
HEADS = ("type", "argument", "aim", "queued", "actor", "same_type", "size", "target", "region", "cell")
# the critic's outputs: one per reward channel (training/selfplay/reward.py), so the learner can weigh the channels
# (fade the shaping) without the critic relearning anything
CRITIC = ("win", "build", "units", "upgrades", "dealt", "taken")


def draw_map(s, static_map):
    """features.build_map for a batch of steps on the GPU: s holds [N, ...] unit rows (`unit_count`, `unit_cell`,
    `unit_owner`, `unit_type`, `units`), `trees` [N, MAP, MAP] and `recent` [N, HISTORY] (-1 for none);
    static_map [N, 4, MAP, MAP] is walkable, buildable, starts and mines. Returns [N, MAP_CHANNELS, MAP, MAP]."""
    n, cells = s["trees"].shape[0], MAP * MAP
    m = torch.zeros(n, MAP_CHANNELS, cells, device=s["trees"].device)
    m[:, 0] = s["trees"].reshape(n, cells).float()
    m[:, 1] = static_map[:, 3].reshape(n, cells).float()
    owner, cell = s["unit_owner"].long(), s["unit_cell"].long()
    structure = s["units"][..., COLUMN["structure"]] > 0
    channel = torch.full_like(owner, -1)
    channel = torch.where(owner == OWNERS.index("me"), 2 + structure.long(), channel)
    channel = torch.where(owner == OWNERS.index("enemy"), 4 + structure.long(), channel)
    channel = torch.where(owner == OWNERS.index("creep"), torch.full_like(channel, 6), channel)
    mine = type_ids().get(GOLD_MINE, -1)
    channel = torch.where(s["unit_type"].long() == mine, torch.full_like(channel, -1), channel)
    real = torch.arange(owner.shape[1], device=owner.device)[None] < s["unit_count"][:, None]
    keep = (channel >= 0) & real
    flat = m.view(n, -1)
    flat.scatter_add_(1, (channel.clamp(min=0) * cells + cell).masked_fill(~keep, 0), keep.float())
    recent = s["recent"].long()
    flat.scatter_reduce_(1, 7 * cells + recent.clamp(min=0), (recent >= 0).float(), "amax")
    m[:, :8] = m[:, :8].clamp(max=4.0) / 4.0
    m[:, 8:11] = static_map[:, :3].reshape(n, 3, cells).float().clamp(max=1.0)
    grid = m.view(n, MAP_CHANNELS, MAP, MAP)
    # features.enemy_starts: the starts with none of the player's buildings within OWN_BASE cells (a max-pool of
    # the own-structure channel is "a building within OWN_BASE cells either way")
    # (a k x k max is a 1 x k max of a k x 1 max: the same result, 2k comparisons a cell instead of k^2)
    k = 2 * OWN_BASE + 1
    near = F.max_pool2d((grid[:, 3:4] > 0).float(), (k, 1), stride=1, padding=(OWN_BASE, 0))
    near = F.max_pool2d(near, (1, k), stride=1, padding=(0, OWN_BASE))[:, 0] > 0
    has_base = (grid[:, 3] > 0).flatten(1).any(1)[:, None, None]
    grid[:, 11] = grid[:, 10] * ~(near & has_base)
    return grid


class _Embed(torch.autograd.Function):
    """F.embedding, or F.embedding_bag (a sum, index 0 padding), with the table's gradient computed as the
    one-hot rows (a bag's: counts of each index) transposed times the output's gradient. PyTorch's own backward
    sums every row of one index on one thread, and most rows here are padding (index 0): it was a quarter of a
    learner update; 72 ms against 21 for a minibatch's tables (training/bench/embedding_grad.py)."""

    @staticmethod
    def forward(ctx, weight, ids, bag):
        ctx.save_for_backward(ids)
        ctx.size, ctx.bag = weight.shape[0], bag
        return F.embedding_bag(ids, weight, mode="sum", padding_idx=0) if bag else F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        grad = grad.float().reshape(-1, grad.shape[-1])
        if ctx.bag:
            counts = torch.zeros(ids.shape[0], ctx.size, device=ids.device)
            counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
            counts[:, 0] = 0  # the padding index learns nothing
        else:
            counts = F.one_hot(ids.reshape(-1), ctx.size).float()
        return counts.T @ grad, None, None


def embed(table: nn.Embedding | nn.EmbeddingBag, ids: torch.Tensor) -> torch.Tensor:
    """table(ids), with _Embed's backward when it will be trained (no gradient: the module itself)."""
    if not (torch.is_grad_enabled() and table.weight.requires_grad):
        return table(ids)
    return _Embed.apply(table.weight, ids, isinstance(table, nn.EmbeddingBag))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x[mask].mean() without waiting for the GPU to count the mask (0 where it is empty)."""
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1)


# parts a checkpoint may predate, started fresh on load: the critic of a cloning run, PopArt's statistics, and the
# plan input as AlphaStar's (a checkpoint from before it: its plan then adds nothing)
FRESH = ("value_net.", "opponent_in.", "value_mean", "value_std", "plan_start",
         "plan_order.", "plan_units.", "plan_upgrades.", "plan_made.")  # fmt: skip


def upgrade(state: dict, critic: tuple | None = None) -> dict:
    """A state dict from before the map's twelfth channel (the enemy's start locations): its patch input gets zero
    weights for the new channel's cells, so the model starts out exactly as it was and learns to use them."""
    w = state.get("patch_in.weight")
    if w is not None and w.shape[1] == (MAP_CHANNELS - 1) * PATCH * PATCH:
        state = dict(state)
        state["patch_in.weight"] = torch.cat([w, w.new_zeros(w.shape[0], PATCH * PATCH)], 1)
    if (
        "plan_mix.weight" in state
    ):  # the plan input before AlphaStar's (summed into one vector): it starts as the empty plan
        state = {k: v for k, v in state.items() if k != "plan_mix.weight"}
        state["plan_out.3.weight"], state["plan_out.3.bias"] = (
            state["plan_out.3.weight"] * 0,
            state["plan_out.3.bias"] * 0,
        )
    # the critic's checkpoint is kept only if it has today's channels (CRITIC, saved as "critic"); otherwise it starts
    # fresh with its statistics: the one-layer critic before value_net (2026-10-11), or other channels
    stale = (
        "critic.3.weight" in state
        or (critic is not None and tuple(critic) != CRITIC)
        or ("value_net.out" in state and state["value_net.out"].shape[0] != len(CRITIC))
    )
    if stale:
        state = {
            k: v for k, v in state.items() if not k.startswith(("critic.", "value_net.", "value_mean", "value_std"))
        }
    # the auxiliary heads (opponent scalars, the core 1 and 4 s ahead), dropped 2026-10-12: sp40's froze the core
    # (its cosine to 4 s later 0.82 -> 0.97 in 150 updates, training/bench/rl_drift.py)
    return {k: v for k, v in state.items() if not k.startswith(("opponent_head.", "future_heads."))}


def row_size(fullest: int) -> int:
    """The unit rows a batch whose fullest step has `fullest` units goes through the transformer with (ROW_SIZES)."""
    return next(r for r in ROW_SIZES if r >= max(1, fullest))


def pack(counts: np.ndarray, rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy.encode_step's `pack` for steps with these unit counts at this row size: the places of each step's real
    tokens (its first min(count, rows) unit rows, the patches, the step token) among the steps' rows + PATCHES + 1
    tokens each, step after step, and where each step begins (cu_seqlens), on the device. Made on the CPU: the
    GPU would have to say how many tokens there are."""
    columns = np.arange(rows + PATCHES + 1)
    keep = (columns[None] < np.minimum(counts, rows)[:, None]) | (columns[None] >= rows)
    cu = np.zeros(len(counts) + 1, np.int32)
    cu[1:] = np.cumsum(keep.sum(1))
    pinned = device.type == "cuda"
    move = lambda a: torch.from_numpy(a).pin_memory().to(device, non_blocking=True) if pinned else torch.from_numpy(a)  # noqa: E731
    return move(np.flatnonzero(keep)), move(cu)


class Critic(nn.Module):
    """A value network per reward channel (AlphaStar's baselines, DI-star's ValueBaseline: 16 residual blocks of 256
    each): a linear to `width`, `blocks` residual blocks (RMS norm, linear, SiLU, linear) and a linear to the value,
    the channels' networks run side by side as batched matrix products. Its own depth, so what it learns lives
    here and not in the trunk the policy reshapes (sp37: a one-layer critic's fit fell from ~0.4 to ~0.08 once the
    policy learned). x [..., inputs] -> [..., channels]."""

    def __init__(self, inputs: int, channels: int, width: int = 256, blocks: int = 16):
        super().__init__()
        self.inp = nn.Parameter(torch.randn(channels, inputs, width) / inputs**0.5)
        self.inp_bias = nn.Parameter(torch.zeros(channels, width))
        self.norm = nn.Parameter(torch.ones(blocks, channels, width))
        self.w1 = nn.Parameter(torch.randn(blocks, channels, width, width) / width**0.5)
        self.b1 = nn.Parameter(torch.zeros(blocks, channels, width))
        self.w2 = nn.Parameter(torch.zeros(blocks, channels, width, width))  # each block starts as the identity
        self.b2 = nn.Parameter(torch.zeros(blocks, channels, width))
        self.out = nn.Parameter(torch.zeros(channels, width))  # the values start at 0 (PopArt's mean)
        self.out_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        h = torch.einsum("...i,cio->...co", x, self.inp) + self.inp_bias
        for k in range(self.w1.shape[0]):
            y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm[k]
            y = F.silu(torch.einsum("...ci,cio->...co", y, self.w1[k]) + self.b1[k])
            h = h + torch.einsum("...ci,cio->...co", y, self.w2[k]) + self.b2[k]
        return (h * self.out).sum(-1) + self.out_bias


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


ATTENTION = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class Block(nn.Module):
    """Pre-norm transformer block: attention with QK-norm, then a SwiGLU feed-forward."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d // heads), RMSNorm(d // heads)
        self.out = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def forward(self, x, key_mask=None, causal=False, packed=None):
        """packed: (cu_seqlens, longest) when x is [1, T, d], several sequences' tokens end to end (sequence i is
        tokens cu_seqlens[i] .. cu_seqlens[i + 1] - 1, at most `longest` long), each attending only to itself."""
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if packed is not None:  # each sequence's real tokens only: FlashAttention-2's variable-length kernel
            cu, longest = packed
            a = flash_attn_varlen_func(q[0].to(v.dtype), k[0].to(v.dtype), v[0], cu, cu, longest, longest)
        else:
            mask = None if key_mask is None else key_mask[:, None, None, :]
            # cuDNN's kernel first: with the key mask PyTorch otherwise takes its memory-efficient one, twice as slow
            # at this shape on an H100 (training/bench/attention_varlen.py); flash for the unmasked causal time attention
            with sdpa_kernel(ATTENTION, set_priority=True):
                a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                                   is_causal=causal and mask is None).transpose(1, 2)  # fmt: skip
        x = x + self.out(a.reshape(b, n, d))
        gate, up = self.up(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class SSM(nn.Module):
    """Diagonal, input-gated linear recurrence over time with a residual: h_t = a_t * h_{t-1} + (1 - a_t) * u_t."""

    def __init__(self, d, state=2):
        super().__init__()
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, 2 * d * state, bias=False)
        self.gate = nn.Linear(d, d * state)
        self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.5), d * state))
        self.out = nn.Linear(d * state, d, bias=False)
        self.width = d * state

    def forward(self, x, h0=None):
        b, t, _ = x.shape
        u, z = self.inp(self.norm(x)).chunk(2, -1)
        log_a = -F.softplus(self.gate(self.norm(x)).float()) * self.log_rate.exp()  # log of the decay in (0, 1)
        u = u.float()  # the recurrence in float32: a memory kept over thousands of steps loses bfloat16's bits
        h = x.new_zeros(b, self.width, dtype=torch.float32) if h0 is None else h0.float()
        if t == 1:  # one step (playing)
            a = log_a[:, 0].exp()
            hs = (a * h + (1 - a) * u[:, 0])[:, None]
        else:  # a chunk (training): every step at once, h_t = sum_s<=t exp(L_t - L_s) (1 - a_s) u_s + exp(L_t) h0
            hs = _scan(log_a, (1 - log_a.exp()) * u, h)
        y = hs.to(z.dtype) * F.silu(z)
        return x + self.out(y), hs[:, -1]


def _scan(log_a, b, h0):
    """h_t = a_t h_{t-1} + b_t for every t of [B, T, W] inputs at once, from h0 [B, W]: with L the running sum of
    log a, h_t = sum over s <= t of exp(L_t - L_s) b_s, plus exp(L_t) h0. The weights exp(L_t - L_s) are at most 1
    (a decays), so nothing overflows; one [B, T, T, W] product instead of T small steps (the step-by-step loop
    was thousands of tiny GPU kernels an update, forward and backward)."""
    t = log_a.shape[1]
    cum = log_a.cumsum(1)
    later = torch.ones(t, t, dtype=torch.bool, device=log_a.device).tril()  # [t, s]: s <= t
    gap = (cum[:, :, None] - cum[:, None]).masked_fill(~later[None, :, :, None], float("-inf"))
    return torch.einsum("btsw,bsw->btw", gap.exp(), b) + cum.exp() * h0[:, None]


class Policy(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, memory=3):
        super().__init__()
        self.d = d
        n_args = len(argument_ids()) + 1
        self.unit_in = nn.Linear(UNIT_FLOATS, d)
        self.type_emb = nn.Embedding(len(type_ids()) + 1, d)
        self.owner_emb = nn.Embedding(len(OWNERS), d)
        self.order_emb = nn.Embedding(16, d)
        self.item_emb = nn.EmbeddingBag(len(type_ids()) + 1, d, mode="sum", padding_idx=0)
        self.buff_emb = nn.EmbeddingBag(len(buff_ids()) + 1, d, mode="sum", padding_idx=0)
        self.ability_emb = nn.EmbeddingBag(len(ability_ids()) + 1, d, mode="sum", padding_idx=0)
        self.patch_in = nn.Linear(MAP_CHANNELS * PATCH * PATCH, d)
        self.patch_pos = nn.Parameter(torch.zeros(PATCHES, d))
        self.scalar_in = nn.Linear(scalar_size(), d)
        self.previous_emb = nn.Embedding(len(TYPES), d)
        self.plan_choice = nn.Embedding(n_args, d, padding_idx=0)  # heroes and build order: which
        self.plan_hero_pos = nn.Parameter(torch.zeros(PLAN_HEROES, d))  # and in what place
        self.plan_order_pos = nn.Parameter(torch.zeros(PLAN_ORDER, d))
        self.plan_start = nn.Parameter(torch.zeros(d))  # the build order's own token, always there to attend to
        self.plan_order = Block(
            d, heads
        )  # the build order through its own attention layer (AlphaStar's small transformer)
        self.plan_units = nn.Sequential(nn.Linear(n_args, PRESENCE), nn.ReLU())  # unit, building, hero types made
        self.plan_upgrades = nn.Sequential(nn.Linear(n_args, PRESENCE), nn.ReLU())  # research levels
        self.plan_made = nn.Linear(2 * PRESENCE, d)  # the two presence vectors into the plan's vector
        self.plan_out = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.step_blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.memory = nn.ModuleList(SSM(d) for _ in range(memory))
        self.time_attention = Block(d, heads)
        self.core_norm = RMSNorm(d)
        self.step_norm = RMSNorm(d)  # units, patches and the step vector leave the step transformer normalised
        self.query_norm = RMSNorm(d)  # each head's query: the core plus the embeddings of the choices before it
        # heads
        self.type_head = nn.Linear(d, len(TYPES))
        self.type_choice = nn.Embedding(len(TYPES), d)
        self.argument_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_args))
        self.argument_choice = nn.Embedding(n_args, d)
        self.aim_head = nn.Linear(d, len(AIMS))
        self.aim_choice = nn.Embedding(len(AIMS), d)
        self.queued_head = nn.Linear(d, 2)
        self.queued_choice = nn.Embedding(2, d)
        self.actor_query = nn.Linear(d, d)
        self.unit_key = nn.Linear(d, d)
        self.group_head = nn.Linear(2 * d, 2 + len(SIZES))  # same_type, then size
        self.target_query = nn.Linear(2 * d, d)
        self.region_query = nn.Linear(2 * d, d)
        self.patch_key = nn.Linear(d, d)
        self.cell_head = nn.Linear(3 * d, PATCH * PATCH)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        # RL's critic: the core and the opponent's scalars (its gold, food, research and unit counts), which a
        # player cannot see but the critic may, as AlphaStar's did: judging a position is much easier knowing the
        # other side's army. Only training uses it; the policy never sees the opponent.
        self.opponent_in = nn.Linear(scalar_size(), d)
        self.value_net = Critic(2 * d, len(CRITIC))
        # PopArt (IMPALA): the critic predicts returns in normalised units; value_mean and value_std (moving
        # statistics of the returns, kept in the checkpoint so actors use them) turn them back into returns
        self.register_buffer("value_mean", torch.zeros(len(CRITIC)))
        self.register_buffer("value_std", torch.ones(len(CRITIC)))
        research = torch.zeros(n_args, dtype=torch.bool)
        research[[i for code, i in argument_ids().items() if fourcc(code) in reference()["upgrades"]]] = True
        self.register_buffer("research", research, persistent=False)
        for name, rule in head_rules().items():
            self.register_buffer(f"rule_{name}", torch.as_tensor(rule), persistent=False)
        # every head starts out uniform (and the value at 0.5): no large first gradients from confident guesses
        for layer in (
            self.type_head,
            self.argument_head[-1],
            self.aim_head,
            self.queued_head,
            self.group_head,
            self.cell_head,
            self.value_head[-1],
            self.actor_query,
            self.target_query,
            self.region_query,
        ):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # ---- encoding
    def encode_step(self, s, rows: int | None = None, pack=None):
        """s: one batch of steps flattened to [N, ...]. Returns unit vectors, patch vectors, step vector, and which
        unit rows are real. Only as many unit rows as the fullest step has, rounded up to one of ROW_SIZES, go
        through the transformer (the rest are padding, which attention ignores anyway; a few fixed sizes let
        torch.compile make one program per size); the unit vectors come back padded to MAX_UNITS with zeros.
        `rows`: that size when the caller knows it (reading the fullest count from the GPU waits for it).
        `pack`: (index, cu_seqlens) to run the transformer on the real tokens only (Block's packed form): index [T]
        the real tokens' places in the steps' [N * (rows + PATCHES + 1)] tokens, step after step, cu_seqlens [N + 1]
        where each step's begin (learner.pack makes them; needs flash-attn). The padding rows come out zero; every
        use of them is masked."""
        n = s["units"].shape[0]
        if rows is None:
            fullest = max(1, int(s["unit_count"].max()))
            rows = next(r for r in ROW_SIZES if r >= fullest)

        def cut(v):  # the first `rows` unit rows, zeros added where the inputs were cut shorter
            v = v[:, :rows]
            return v if v.shape[1] == rows else F.pad(v, (0, 0) * (v.dim() - 2) + (0, rows - v.shape[1]))

        units = self.unit_in(cut(s["units"]).float())
        units = (
            units
            + embed(self.type_emb, cut(s["unit_type"]).long())
            + embed(self.owner_emb, cut(s["unit_owner"]).long())
        )
        units = units + embed(self.order_emb, cut(s["unit_order"]).long().clamp(max=15))
        units = units + embed(self.type_emb, cut(s["unit_queue"]).long())
        for bag, field in (
            (self.item_emb, "unit_inventory"),
            (self.buff_emb, "unit_buffs"),
            (self.ability_emb, "unit_abilities"),
        ):
            ids = cut(s[field]).long().reshape(n * rows, -1)
            units = units + embed(bag, ids).view(n, rows, self.d)
        grid = s["map"] if "map" in s else draw_map(s, s["static_map"])
        patches = grid.float().unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)  # [N, C, 8, 8, 16, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, PATCHES, -1)
        patches = self.patch_in(patches) + self.patch_pos
        scalar = self.scalar_in(s["scalars"].float()) + self.previous_emb(s["previous_type"].long()).sum(1)
        if "plan" in s:
            scalar = scalar + s["plan"]
        x = torch.cat([units, patches, scalar[:, None]], 1)
        valid = torch.arange(MAX_UNITS, device=x.device)[None] < s["unit_count"][:, None]
        key_mask = torch.cat([valid[:, :rows], torch.ones(n, PATCHES + 1, dtype=torch.bool, device=x.device)], 1)
        if pack is None:
            for block in self.step_blocks:
                x = block(x, key_mask)
            x = self.step_norm(x)
        else:
            index, cu = pack
            tokens = x.shape[1]
            xp = x.reshape(n * tokens, self.d)[index][None]
            for block in self.step_blocks:
                xp = block(xp, packed=(cu, tokens))
            xp = self.step_norm(xp)[0]
            x = xp.new_zeros(n * tokens, self.d).index_copy(0, index, xp).view(n, tokens, self.d)
        unit_out = F.pad(x[:, :rows], (0, 0, 0, MAX_UNITS - rows))
        return unit_out, x[:, rows:-1], x[:, -1], valid

    def compile_blocks(self, dynamic: bool | None = False) -> None:
        """torch.compile the transformer blocks and the SSM layers, in place (the state dict is unchanged): their
        small elementwise kernels fused. With unit rows in ROW_SIZES, a few programs cover every batch. The whole
        forward pass does not compile (Inductor mixes float32 and bfloat16 in one of its matrix products).
        `dynamic`: torch.compile's (None, as the learner's packed tokens need: a program for any token count)."""
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        for block in [*self.step_blocks, self.time_attention, *self.memory]:
            block.compile(dynamic=dynamic)

    def encode_plan(self, batch):
        """Each row's plan [B, d], added to the scalar token (0 for the empty plan), as AlphaStar's z: the build order
        through its own attention layer, and what was made over the game as two presence vectors, unit types and
        research levels (docs/training/3-model.md, the plan input)."""
        heroes, order = batch["plan_heroes"].long(), batch["plan_order"].long()
        mix = batch["plan_mix"].float()
        h = (self.plan_choice(heroes) + self.plan_hero_pos * (heroes > 0)[..., None]).sum(1)
        o = self.plan_choice(order) + self.plan_order_pos
        o = torch.cat([self.plan_start.expand(len(order), 1, -1).to(o.dtype), o], 1)
        o = self.plan_order(o, key_mask=F.pad(order > 0, (1, 0), value=True))[:, 0]
        units = self.plan_units((mix > 0).to(mix.dtype) * ~self.research)
        levels = (16 * mix**2).round().clamp(max=3) / 3 * self.research  # mix is sqrt(count) / 4
        upgrades = self.plan_upgrades(levels)
        ordered, made = (order > 0).any(1)[:, None], ((heroes > 0).any(1) | (mix > 0).any(1))[:, None]
        x = o * ordered + (h + self.plan_made(torch.cat([units, upgrades], -1))) * made
        return self.plan_out(x) * (ordered | made)

    def forward(self, batch, state=None, window=None, window_mask=None, rows: int | None = None, pack=None):
        """batch: dict of [B, T, ...] tensors. Returns per-step encodings and the core output [B, T, d].

        `state` is the memory the previous chunk of each row's game ended with (None: every game starts here);
        a row whose `reset` [B] is true starts a new game, so its memory starts from zero. The attention layer sees
        the chunk's steps so far; when playing one step at a time, `window` is the earlier steps of the current
        chunk (out["window"] of the last call; None at a chunk's start, every CHUNK steps from the game's start);
        rows whose games are at different points of their chunks pad their windows on the left, `window_mask` [B, L]
        marking the real steps (one step at a time only). `rows`, `pack`: encode_step's, when the caller has them."""
        b, t = batch["unit_count"].shape
        flat = {
            k: v.reshape(b * t, *v.shape[2:])
            for k, v in batch.items()
            if k not in PER_GAME and v.dim() >= 2 and v.shape[:2] == (b, t)
        }
        if "static_map" in batch:  # once per row: the same map data for each of its steps
            flat["static_map"] = batch["static_map"][:, None].expand(-1, t, -1, -1, -1).reshape(b * t, 4, MAP, MAP)
        if "plan_heroes" in batch:
            flat["plan"] = self.encode_plan(batch)[:, None].expand(-1, t, -1).reshape(b * t, self.d)
        units, patches, step, valid = self.encode_step(flat, rows, pack)
        x = step.view(b, t, self.d)
        if state is not None and "reset" in batch:
            keep = (~batch["reset"].bool()).to(x.dtype)[:, None]
            state = [h * keep for h in state]
        states = []
        for k, layer in enumerate(self.memory):
            x, h = layer(x, None if state is None else state[k])
            states.append(h)
        seq = x if window is None else torch.cat([window, x], 1)
        if window_mask is None:
            x = self.time_attention(seq, causal=True)[:, -t:]  # within the chunk, as in training
        else:  # one new step: it may see every real step before it
            keys = torch.cat([window_mask.bool(), torch.ones(b, t, dtype=torch.bool, device=x.device)], 1)
            x = self.time_attention(seq, key_mask=keys)[:, -t:]
        core = self.core_norm(x)
        return {
            "units": units.view(b, t, MAX_UNITS, self.d),
            "patches": patches.view(b, t, PATCHES, self.d),
            "valid": valid.view(b, t, MAX_UNITS),
            "core": core,
            "state": states,
            "window": seq,
        }

    # ---- heads, given the choices before them (teacher forcing in training)
    @staticmethod
    def _masked(logits, mask):
        return logits.masked_fill(~mask, -1e9)

    def _flow(self, out):
        """The heads in HEADS order, as a generator: it yields (head, logits) and is sent each head's choice (a
        long tensor shaped like the logits without their last axis; -1 where unused) before the next. Training
        sends the labels (heads()), playing its samples (decide.sample): one computation for both."""
        core, units, patches = out["core"], out["units"], out["patches"]
        scale = math.sqrt(self.d)

        def pick(table, index):  # rows of table [..., N, d] at index [...]
            index = index.clamp(min=0).long()
            return torch.gather(table, -2, index[..., None, None].expand(*index.shape, 1, self.d)).squeeze(-2)

        kind = yield "type", self.type_head(core)
        q = core + self.type_choice(kind.clamp(min=0))
        argument = yield "argument", self.argument_head(self.query_norm(q))
        q = q + self.argument_choice(argument.clamp(min=0))
        aim = yield "aim", self.aim_head(self.query_norm(q))
        q = q + self.aim_choice(aim.clamp(min=0))
        queued = yield "queued", self.queued_head(self.query_norm(q))
        q = self.query_norm(q + self.queued_choice(queued.clamp(min=0)))
        keys = self.unit_key(units)
        actor = yield "actor", torch.einsum("...d,...nd->...n", self.actor_query(q), keys) / scale
        qa = torch.cat([q, pick(units, actor)], -1)
        group = self.group_head(qa)
        yield "same_type", group[..., :2]
        yield "size", group[..., 2:]
        yield "target", torch.einsum("...d,...nd->...n", self.target_query(qa), keys) / scale
        region = yield (
            "region",
            torch.einsum("...d,...pd->...p", self.region_query(qa), self.patch_key(patches)) / scale,
        )
        yield "cell", self.cell_head(torch.cat([qa, pick(patches, region)], -1))

    def heads(self, out, labels):
        """Every head's logits given the labels' earlier choices (teacher forcing), and the cloning value."""
        flow = self._flow(out)
        h = {}
        name, logits = next(flow)
        while True:
            h[name] = logits
            try:
                name, logits = flow.send(labels[name].long())
            except StopIteration:
                break
        # the value reads the core but does not train it: with ~1,300 game results it would teach the trunk to
        # recognise each game and recall its winner
        h["value"] = self.value_head(out["core"].detach()).squeeze(-1)
        return h

    def normalized_value(self, out, opponent_scalars=None):
        """The critic's estimates in normalised units [B, T, CRITIC]: from the core (detached: the critic does not train the
        trunk) and the opponent's scalars [B, T, S] (zeros when there is no opponent observation)."""
        core = out["core"].detach()
        if opponent_scalars is None:
            opponent = torch.zeros_like(core)
        else:
            opponent = self.opponent_in(opponent_scalars.float())
        return self.value_net(torch.cat([core, opponent], -1))

    def value(self, out, opponent_scalars=None):
        """The critic's estimates of each channel's return [B, T, CRITIC]."""
        return self.normalized_value(out, opponent_scalars) * self.value_std + self.value_mean

    @torch.no_grad()
    def update_value_statistics(self, returns: torch.Tensor, rate: float) -> None:
        """PopArt, per channel: move value_mean and value_std towards these returns' [N, CRITIC] and rescale the
        critic's last layer so its estimates in return units stay exactly where they were."""
        old_mean, old_std = self.value_mean.clone(), self.value_std.clone()
        second = old_std**2 + old_mean**2
        mean = (1 - rate) * old_mean + rate * returns.float().mean(0)
        second = (1 - rate) * second + rate * (returns.float() ** 2).mean(0)
        std = (second - mean**2).clamp(min=1e-8).sqrt().clamp(min=1e-4)
        self.value_net.out.mul_((old_std / std)[:, None])
        self.value_net.out_bias.mul_(old_std).add_(old_mean - mean).div_(std)
        self.value_mean.copy_(mean)
        self.value_std.copy_(std)

    @torch.no_grad()
    def uses(self, batch) -> dict:
        """Each head's steps where the label uses it, from the labels alone: the type on every real step; the others
        on acting steps whose type uses them (head_rules); the value on real steps of games with a known result.
        Padding after a game's end never counts."""
        real = batch["step_valid"].bool() if "step_valid" in batch else torch.ones_like(batch["type"], dtype=torch.bool)
        kind, aim = batch["type"].long(), batch["aim"].long()
        acting = real & (kind > 0)
        uses_actor = acting & self.rule_has_actor[kind]
        grouped = uses_actor & self.rule_grouped[kind]
        targeted, pointed = acting & (aim == AIMS.index("unit")), acting & (aim == AIMS.index("point"))
        return {"type": real, "argument": acting & self.rule_has_argument[kind], "aim": acting,
                "queued": uses_actor & self.rule_queueable[kind], "actor": uses_actor, "same_type": grouped,
                "size": grouped, "target": targeted, "region": pointed, "cell": pointed,
                "value": (batch["result"] >= 0) & real if "result" in batch else real}  # fmt: skip

    def terms(self, batch, out, h):
        """Each head's (masked logits, choice, steps where it is used: uses()), with the stored masks (the data
        set's, or those an actor sampled with) and the aim each type allows."""
        labels, u = batch, self.uses(batch)
        kind = labels["type"].long()
        argument_mask = batch["argument_mask"].bool().clone()
        argument_mask[..., 0] = False
        actor_mask = batch["actor_mask"].bool() & out["valid"]
        target_mask = batch["target_mask"].bool() & out["valid"]

        def masked(logits, mask, use):  # unused rows keep any mask: they never count, and must not be all -inf
            return self._masked(logits, mask | ~use[..., None])

        return {
            "type": (self._masked(h["type"], batch["type_mask"].bool()), labels["type"], u["type"]),
            "argument": (masked(h["argument"], argument_mask, u["argument"]), labels["argument"], u["argument"]),
            "aim": (masked(h["aim"], self.rule_aim[kind], u["aim"]), labels["aim"], u["aim"]),
            "queued": (h["queued"], labels["queued"], u["queued"]),
            "actor": (masked(h["actor"], actor_mask, u["actor"]), labels["actor"], u["actor"]),
            "same_type": (h["same_type"], labels["same_type"], u["same_type"]),
            "size": (h["size"], labels["size"], u["size"]),
            "target": (masked(h["target"], target_mask, u["target"]), labels["target"], u["target"]),
            "region": (h["region"], labels["region"], u["region"]),
            "cell": (h["cell"], labels["cell"], u["cell"]),
        }

    def log_probs(self, batch, out):
        """For RL: each step's log-probability of its action [B, T] (the sum over the heads it used), and per
        head (log-probabilities [B, T, K], choice, used) for entropies and the KL to another policy."""
        h = self.heads(out, batch)
        per = {}
        total = torch.zeros_like(batch["type"], dtype=torch.float)
        for name, (logits, choice, use) in self.terms(batch, out, h).items():
            lp = F.log_softmax(logits.float(), -1)
            chosen = lp.gather(-1, choice.long().clamp(min=0)[..., None]).squeeze(-1)
            total = total + torch.where(use, chosen, torch.zeros_like(chosen))
            per[name] = (lp, choice, use)
        return total, per

    def cloning_loss(self, batch, out, value_weight=0.1, counts: dict | None = None):
        """Cross-entropy of each head on the label's choice over the steps whose label uses the head, plus the value
        head's cross-entropy on the game's result over every real step of games with a known result. Every step
        counts the same: weighting rare actions up taught the clone their odds wrong (bc9 cancelled its own
        production 44x as often as pros). Each head's sum is divided by its step count in `counts` ({head: count}
        over a whole batch, when this one is a piece of it: the pieces then add up to the whole's loss; default: this
        batch's own). Returns (loss, {head: loss}, logits)."""
        h = self.heads(out, batch)
        uses = self.uses(batch)
        counts = counts or {k: v.sum() for k, v in uses.items()}
        parts = {}
        for name, (logits, target, use) in self.terms(batch, out, h).items():
            ce = F.cross_entropy(logits[use].float(), target[use].long(), reduction="sum")
            parts[name] = ce / counts[name].clamp(min=1)
        known = uses["value"]
        value = F.binary_cross_entropy_with_logits(
            h["value"][known].float(), batch["result"][known].float(), reduction="sum"
        )
        parts["value"] = value / counts["value"].clamp(min=1)
        loss = sum(v * (value_weight if k == "value" else 1) for k, v in parts.items())
        return loss, {k: float(v.detach()) for k, v in parts.items()}, h

    @torch.no_grad()
    def counts(self, batch, out, h) -> dict:
        """{head: (correct, used)} of the most likely choice against the label (unweighted), plus: `p_act`, the
        model's chance of acting (to compare with `label_acts`, how often the player did; argmax alone would say
        noop whenever it is the likeliest single type); `type_given_act`, the type it would pick given that it acts,
        on steps where the player acted; and `value`, the result's side of 0.5 over real steps of known games."""
        terms, uses = self.terms(batch, out, h), self.uses(batch)
        c = {k: (int((lg.argmax(-1) == t)[u].sum()), int(u.sum())) for k, (lg, t, u) in terms.items()}
        lg, t, real = terms["type"]
        acting = real & (t > 0)
        c["p_act"] = (float((1 - lg.float().softmax(-1)[..., 0])[real].sum()), int(real.sum()))
        c["label_acts"] = (int(acting.sum()), int(real.sum()))
        c["type_given_act"] = (int((lg[..., 1:].argmax(-1) + 1 == t)[acting].sum()), int(acting.sum()))
        known = uses["value"]
        c["value"] = (int(((h["value"] > 0).long() == batch["result"])[known].sum()), int(known.sum()))
        return c


def float32(out: dict) -> dict:
    """A forward pass's outputs in float32 (the trunk runs in bfloat16 on a GPU; the heads, whose log-probabilities
    the learner and the actors compare, in float32 on both)."""
    return {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in out.items()}


def load(checkpoint, device="cpu", eval: bool = True) -> Policy:
    """A Policy from a checkpoint ({"model": state dict}, or such a dict itself). Parts a checkpoint predates
    (FRESH) start fresh: the critic at 0, the statistics at mean 0 and spread 1."""
    # our own checkpoints: a self-play one also keeps its league (a plain object, not only tensors)
    saved = (
        torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict)
        else checkpoint
    )
    state = saved.get("model", saved)
    model = Policy().to(device)
    missing, unexpected = model.load_state_dict(upgrade(state, saved.get("critic")), strict=False)
    if unexpected or any(not k.startswith(FRESH) for k in missing):
        raise ValueError(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    return model.eval() if eval else model
