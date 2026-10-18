"""Two SVG plots of steps/s against model size, from the estimates in docs/hardware/README.md."""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OURS = 11.9  # M parameters
# GPU microseconds per step for our 11.9M model: (learner, inference)
CARDS = {"RTX 3090": (326, 124), "RTX 4090": (266, 106), "RTX 5090": (157, 62)}
CPU_ONE, CPU_TWO = (1000, 1300), (2300, 2900)

W, H, L, R, T, B = 760, 440, 70, 190, 30, 55
XMIN, XMAX, YMAX = 4, 40, 8000


def X(p): return L + (p - XMIN) / (XMAX - XMIN) * (W - L - R)
def Y(s): return T + (1 - min(s, YMAX) / YMAX) * (H - T - B)


def frame(title):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{L}" y="18" font-size="14">{title}</text>']
    for s in range(0, YMAX + 1, 1000 if YMAX > 4000 else 500):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(s):.1f}" y2="{Y(s):.1f}" stroke="#e1e0d9"/>')
        out.append(f'<text x="{L - 6}" y="{Y(s) + 4:.1f}" text-anchor="end" fill="#666">{s:,}</text>')
    for p in range(5, XMAX + 1, 5):
        out.append(f'<text x="{X(p):.1f}" y="{H - B + 18}" text-anchor="middle" fill="#666">{p}M</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle" fill="#444">model size (millions of parameters)</text>')
    out.append(f'<text x="16" y="{(T + H - B) / 2}" text-anchor="middle" fill="#444" transform="rotate(-90 16 {(T + H - B) / 2})">steps per second</text>')
    out.append(f'<line x1="{X(OURS):.1f}" x2="{X(OURS):.1f}" y1="{T}" y2="{H - B}" stroke="#999" stroke-dasharray="2 3"/>')
    out.append(f'<text x="{X(OURS) + 4:.1f}" y="{T + 12}" fill="#666">our model, 11.9M</text>')
    return out


def curve(f, color, dash=""):
    pts = " ".join(f"{X(p / 10):.1f},{Y(f(p / 10)):.1f}" for p in range(XMIN * 10, XMAX * 10 + 1, 2))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'


def band(lo, hi, color, label, row):
    return [f'<rect x="{L}" y="{Y(hi):.1f}" width="{W - L - R}" height="{Y(lo) - Y(hi):.1f}" fill="{color}" opacity="0.15"/>',
            f'<text x="{W - R + 8}" y="{Y((lo + hi) / 2) + 4:.1f}" fill="#444">{label}</text>' if row is None else ""]


def legend(items, top=40):
    out = []
    for i, (label, color, dash) in enumerate(items):
        y = top + i * 20
        out.append(f'<line x1="{W - R + 10}" x2="{W - R + 34}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        out.append(f'<text x="{W - R + 40}" y="{y + 4}" fill="#222">{label}</text>')
    return out


def gpu(card): return lambda p: 1e6 / sum(CARDS[card]) * OURS / p


# 1: what each card can feed, against the CPU limits of one and two boxes
svg = frame("GPU limit by card, against the CPU limit of one box and of two boxes")
svg += band(*CPU_ONE, "#888", "CPU: one box", None) + band(*CPU_TWO, "#888", "CPU: two boxes", None)
two_3090 = lambda p: 1e6 / CARDS["RTX 3090"][0] * OURS / p  # learner on one card, inference on the other
cols = {"RTX 3090": (gpu("RTX 3090"), "#2a78d6", ""), "RTX 4090": (gpu("RTX 4090"), "#eb6834", "6 4"),
        "RTX 5090": (gpu("RTX 5090"), "#1baf7a", "2 3"), "2x RTX 3090": (two_3090, "#eda100", "9 3 2 3")}
for f, c, d in cols.values():
    svg.append(curve(f, c, d))
svg += legend([(k, c, d) for k, (_, c, d) in cols.items()], top=140)
(OUT / "steps_vs_model_by_gpu.svg").write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")

YMAX = 4000
# 2: expected speed of the setups (the lower of the CPU and GPU limits; CPU at its band's middle)
one_3090 = gpu("RTX 3090")
one_5090 = gpu("RTX 5090")
setups = {
    "1 box, 1x 3090": (lambda p: min(sum(CPU_ONE) / 2, one_3090(p)), "#2a78d6", ""),
    "2 boxes, 1x 3090": (lambda p: min(sum(CPU_TWO) / 2, one_3090(p)), "#eb6834", "6 4"),
    "2 boxes, 2x 3090": (lambda p: min(sum(CPU_TWO) / 2, two_3090(p)), "#1baf7a", "2 3"),
    "1 box, 1x 5090": (lambda p: min(sum(CPU_ONE) / 2, one_5090(p)), "#eda100", "9 3 2 3"),
    "2 boxes, 1x 5090": (lambda p: min(sum(CPU_TWO) / 2, one_5090(p)), "#e87ba4", "12 4"),
}
svg = frame("Expected speed of each setup (the lower of the CPU and GPU limits)")
for label, (f, c, d) in setups.items():
    svg.append(curve(f, c, d))
svg += legend([(k, c, d) for k, (_, c, d) in setups.items()])
(OUT / "steps_vs_model_by_setup.svg").write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")

for card in CARDS:
    print(card, round(1e6 / sum(CARDS[card])), "steps/s at 11.9M;", "max model at 2,600:", round(gpu(card)(OURS) * OURS / 2600, 1))
print("2x3090", round(two_3090(OURS)), "max model at 2,600:", round(two_3090(OURS) * OURS / 2600, 1))
print("1 box 1x3090 max model at 1,150:", round(one_3090(OURS) * OURS / 1150, 1))
