"""Figure for the EC1 V4 diagnosis: where the gap between 0.679 and 0.778 goes, and how large each lever was.

Numbers are the registered ones (docs/EXPERIMENT_REGISTRY.md): EXP-066 ceiling, EXP-063/064/067/061 paired gains on
image_disjoint, EXP-071 Gaussian neighbourhood (sigma 500 m) and the estimate for the wealth-confusion pairs.
Palette: dataviz reference slots 1 (blue) and 2 (orange) + neutral grey.
Usage: python src/make_fig_exp071_gap.py   Output: outputs/exp071/figures/exp071_gap.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "exp071" / "figures"
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#bdbcb6", "#0b0b0b", "#52514e"

fig, (a, b) = plt.subplots(2, 1, figsize=(7.4, 4.9), gridspec_kw={"height_ratios": [1, 1.9]})

# (a) decomposition of the range above chance on new images
segs = [(0.50, 0.679, BLUE, "reached by DINOv2 + linear head"),
        (0.679, 0.699, ORANGE, "room: 2.0 pts"),
        (0.699, 0.778, GREY, "voter disagreement: 7.9 pts")]
for lo, hi, c, lab in segs:
    a.barh(0, hi - lo, left=lo, height=0.5, color=c, edgecolor="white", linewidth=2)
    a.text((lo + hi) / 2, 0.75 if c == ORANGE else 0.36, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
    if c == ORANGE:
        a.plot([(lo + hi) / 2] * 2, [0.28, 0.72], color=INK2, linewidth=0.6)
for x, t, ha in [(0.50, "chance\n0.50", "center"), (0.677, "model\n0.679", "right"),
                 (0.701, "ceiling\n0.699", "left"), (0.778, "in-sample\noracle 0.778", "center")]:
    a.text(x, -0.36, t, ha=ha, va="top", fontsize=7.5, color=INK2)
a.set_xlim(0.49, 0.80)
a.set_ylim(-1.1, 1.1)
a.axis("off")
a.set_title("(a) Pairwise accuracy on new images", fontsize=9, loc="left", color=INK)

# (b) measured gain of each lever against the room
levers = [("Best of 14 heads (EXP-063)", 0.24, False), ("DINOv3 (EXP-064)", 0.17, False),
          ("DINOv2 ViT-L, 3x larger (EXP-067)", 0.24, False), ("Fine-tuning 2 blocks (EXP-061)", 0.28, False),
          ("Gaussian neighbourhood, 500 m (EXP-071)", 0.01, False),
          ("Fixing all wealth-confusion pairs (estimate)", 0.40, True)]
y = list(range(len(levers)))[::-1]
for yi, (name, g, est) in zip(y, levers):
    b.barh(yi, g, height=0.55, color="white" if est else BLUE, edgecolor=BLUE, hatch="////" if est else None,
           linewidth=1.2)
    b.text(g + 0.04, yi, f"+{g:.2f}", va="center", fontsize=7.5, color=INK)
b.axvline(2.0, color=ORANGE, linewidth=2)
b.text(1.97, y[0] + 0.45, "room left: 2.0 pts", ha="right", fontsize=7.5, color=INK)
b.set_yticks(y, [l[0] for l in levers], fontsize=7.5)
b.set_xlim(0, 2.2)
b.set_xlabel("gain in pairwise accuracy (percentage points)", fontsize=8)
b.grid(axis="x", color="#e6e5e0", linewidth=0.6)
b.set_axisbelow(True)
b.spines[["top", "right"]].set_visible(False)
b.tick_params(axis="x", labelsize=7.5)
b.set_title("(b) What each lever won, against that room (hatched: estimate)", fontsize=9, loc="left", color=INK)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "exp071_gap.pdf")
fig.savefig(OUT / "exp071_gap.png", dpi=110)
print("ok")
