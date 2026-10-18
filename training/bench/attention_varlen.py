"""The step transformer with its padding left out, on an H100. Today each step's unit rows are padded to the
batch's row size (96 for mid-game batches: 161 tokens a step where ~118 are real) and a key mask hides the padding.
Packed, the steps' real tokens are laid end to end and attention runs per step ("variable length"). One
minibatch (2048 steps, 53 units a step on average, 92 at most, + 64 patches + the step token; d 256, 8 heads of 32),
forward + backward in bfloat16, attention alone and a whole policy.model.Block; the largest difference of each
packed way's outputs on the real tokens from today's:

    padded/cudnn     today: F.scaled_dot_product_attention with the key mask, cuDNN's kernel
    packed/flash2    flash_attn_varlen_func (FlashAttention-2, the flash-attn package)
    packed/nested    PyTorch's own: a jagged nested tensor through F.scaled_dot_product_attention
    packed/flex      flex_attention over the packed tokens, a block mask keeping each step to itself

    python -m modal run training/bench/attention_varlen.py
"""

from __future__ import annotations

import modal

from training.bench.common import FLASH_ATTN_WHEEL, image

app = modal.App("wc3-bench-attention-varlen", image=image(f"pip install {FLASH_ATTN_WHEEL}"))


@app.function(gpu="H100", cpu=4, memory=65536, timeout=1800)
def bench() -> str:
    import time

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from training.policy.model import Block

    lines = []

    def say(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    torch.manual_seed(0)
    b, heads, d, rows, rest = 2048, 8, 256, 96, 65
    dim = d // heads
    count = torch.clamp(torch.normal(53.0, 15.0, (b,)), 1, 92).long()
    count[0] = 92
    lengths = (count + rest).cuda()  # each step's real tokens: its units, the patches, the step token
    n = rows + rest
    key_mask = torch.cat([torch.arange(rows)[None] < count[:, None], torch.ones(b, rest, dtype=torch.bool)], 1).cuda()
    total = int(lengths.sum())
    cu = torch.zeros(b + 1, dtype=torch.int32, device="cuda")
    cu[1:] = lengths.cumsum(0)
    say(f"{b} steps: {b * n} padded tokens, {total} real ({total / (b * n):.0%})")

    def timed(fn, *tensors, repeat=10):
        grads = [torch.randn_like(fn(*tensors))]
        for _ in range(3):
            for t in tensors:
                t.grad = None
            fn(*tensors).backward(grads[0])
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeat):
            for t in tensors:
                t.grad = None
            out = fn(*tensors)
            out.backward(grads[0])
        torch.cuda.synchronize()
        return (time.perf_counter() - began) / repeat * 1000, out.detach()

    def pack(x):  # [b, n, ...] -> [total, ...]: the real tokens, step after step
        return x[key_mask]

    # ---- attention alone
    q, k, v = (torch.randn(b, n, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    qp, kp, vp = (pack(t).detach().requires_grad_(True) for t in (q, k, v))
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))

    def padded(q, k, v):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION], set_priority=True):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                  attn_mask=key_mask[:, None, None, :]).transpose(1, 2)  # fmt: skip

    ms, reference = timed(padded, q, k, v)
    reference = pack(reference)
    say(f"attention  padded/cudnn  {ms:6.2f} ms")
    ways = {}
    try:
        from flash_attn import flash_attn_varlen_func

        m = int(lengths.max())
        ways["packed/flash2"] = lambda q, k, v: flash_attn_varlen_func(q, k, v, cu, cu, m, m)
    except Exception as error:  # noqa: BLE001
        say(f"attention  packed/flash2 unavailable: {type(error).__name__}: {str(error)[:150]}")

    offsets = cu.long()

    def nested(q, k, v):
        def jag(t):
            return torch.nested.nested_tensor_from_jagged(t, offsets).transpose(1, 2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(jag(q), jag(k), jag(v))
        return out.transpose(1, 2).values()

    ways["packed/nested"] = nested
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        step = torch.repeat_interleave(torch.arange(b, device="cuda"), lengths)

        def same_step(batch, head, i, j):
            return step[i] == step[j]

        block = create_block_mask(same_step, None, None, total, total, device="cuda")
        flex = torch.compile(flex_attention)
        ways["packed/flex"] = lambda q, k, v: flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                                   block_mask=block)[0].transpose(0, 1)  # fmt: skip
    except Exception as error:  # noqa: BLE001
        say(f"attention  packed/flex unavailable: {type(error).__name__}: {str(error)[:150]}")
    attention = {}
    for name, fn in ways.items():
        try:
            ms, out = timed(fn, qp, kp, vp)
            attention[name] = fn
            say(f"attention  {name:13s} {ms:6.2f} ms   max diff {float((out.float() - reference.float()).abs().max()):.1e}")
        except Exception as error:  # noqa: BLE001
            say(f"attention  {name:13s} failed: {type(error).__name__}: {str(error)[:200]}")

    # ---- a whole Block: today's (padded, compiled as in the learner) and packed with each way
    block = Block(d, heads).cuda()
    compiled = torch.compile(block, dynamic=False)
    x = torch.randn(b, n, d, device="cuda", requires_grad=True)

    def block_padded(x):
        with torch.autocast("cuda", torch.bfloat16):
            return compiled(x, key_mask)

    ms, ref_out = timed(block_padded, x, repeat=5)
    ref_out = pack(ref_out).float()
    say(f"block      padded/cudnn  {ms:6.2f} ms (compiled)")

    def packed_math(fn):
        def run(xp):  # Block.forward's math on [1, total, d] (3-D, as the padded block), attention per step
            xp = xp[None]
            h = block.qkv(block.norm1(xp)).view(1, -1, 3, heads, dim).unbind(2)
            qh, kh, vh = block.q_norm(h[0])[0], block.k_norm(h[1])[0], h[2][0]
            a = fn(qh.to(torch.bfloat16), kh.to(torch.bfloat16), vh.to(torch.bfloat16))
            y = xp + block.out(a.reshape(1, -1, d))
            gate, up = block.up(block.norm2(y)).chunk(2, -1)
            return (y + block.down(F.silu(gate) * up))[0]

        return run

    def autocast(fn):
        def run(*a):
            with torch.autocast("cuda", torch.bfloat16):
                return fn(*a)

        return run

    def packed_block(fn):
        return autocast(packed_math(fn))

    xp = pack(x).detach().requires_grad_(True)
    for name, fn in attention.items():
        try:
            ms, out = timed(packed_block(fn), xp, repeat=5)
            say(f"block      {name:13s} {ms:6.2f} ms (eager)   max diff {float((out.float() - ref_out).abs().max()):.1e}")
        except Exception as error:  # noqa: BLE001
            say(f"block      {name:13s} failed: {type(error).__name__}: {str(error)[:200]}")
    if "packed/flash2" in attention:
        flash = attention["packed/flash2"]
        try:  # the whole packed block compiled, flash-attn's op inside it
            whole = autocast(torch.compile(packed_math(flash), dynamic=False))
            ms, out = timed(whole, xp, repeat=5)
            say(f"block      packed/flash2 {ms:6.2f} ms (compiled whole)   max diff {float((out.float() - ref_out).abs().max()):.1e}")
        except Exception as error:  # noqa: BLE001
            say(f"block      packed/flash2 compiled whole failed: {type(error).__name__}: {str(error)[:200]}")

        @torch.compile(dynamic=False)
        def before(xp):
            h = block.qkv(block.norm1(xp)).view(-1, 3, heads, dim).unbind(1)
            return block.q_norm(h[0]).to(torch.bfloat16), block.k_norm(h[1]).to(torch.bfloat16), h[2].to(torch.bfloat16)

        @torch.compile(dynamic=False)
        def after(xp, a):
            y = xp + block.out(a.reshape(-1, d))
            gate, up = block.up(block.norm2(y)).chunk(2, -1)
            return y + block.down(F.silu(gate) * up)

        try:  # compiled around an eager flash call
            ms, out = timed(autocast(lambda xp: after(xp, flash(*before(xp)))), xp, repeat=5)
            say(f"block      packed/flash2 {ms:6.2f} ms (compiled around)   max diff {float((out.float() - ref_out).abs().max()):.1e}")
        except Exception as error:  # noqa: BLE001
            say(f"block      packed/flash2 compiled around failed: {type(error).__name__}: {str(error)[:200]}")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    from pathlib import Path

    Path("build/attention_varlen.txt").write_text(bench.remote(), encoding="utf-8")
