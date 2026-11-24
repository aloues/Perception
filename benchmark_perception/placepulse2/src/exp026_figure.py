"""EXP-026 — figura resumen: Δ pairwise accuracy vs ego-0 (image_disjoint) con IC 95 % pareado.

Uso:  python exp026_figure.py      Salida: outputs/exp026/exp026_delta_accuracy.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils import OUTPUTS  # noqa: E402

OUT = OUTPUTS / "exp026"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"sustitutivo": "#eb6834", "aditivo": "#2a78d6"}

ROWS = [  # (comparación en paired_comparisons.csv, etiqueta, grupo)
    ("sust250 - ego0", "Vecinas ≤250 m reemplazan a la imagen (EXP-021)", "sustitutivo"),
    ("sust_rand250 - ego0", "Aleatorias de la ciudad reemplazan a la imagen", "sustitutivo"),
    ("add250 - ego0", "Imagen + media de vecinas ≤250 m", "aditivo"),
    ("add400 - ego0", "Imagen + media de vecinas ≤400 m", "aditivo"),
    ("add_rand250 - ego0", "Imagen + media de aleatorias (control)", "aditivo"),
    ("add_city - ego0", "Imagen + media de la ciudad", "aditivo"),
]

p = pd.read_csv(OUT / "paired_comparisons.csv")
p = p[(p["protocol"] == "image_disjoint")].set_index("comparison")

fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for k, (comp, label, grp) in enumerate(ROWS):
    y = len(ROWS) - 1 - k
    r = p.loc[comp]
    d, lo, hi = 100 * r["delta_acc"], 100 * r["delta_acc_ci_low"], 100 * r["delta_acc_ci_high"]
    ax.plot([lo, hi], [y, y], color=COL[grp], lw=2, solid_capstyle="round")
    ax.plot(d, y, "o", ms=8, color=COL[grp], mec=SURF, mew=2)
    ax.text(hi + 0.12, y, f"{d:+.2f}", va="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[1] for r in ROWS][::-1], fontsize=8.5, color=INK)
ax.set_xlabel("Δ pairwise accuracy vs. ego-0 (puntos, IC 95 % bootstrap pareado, n = 34,607 pares)",
              fontsize=8, color=INK2)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlim(-4.4, 1.0)
handles = [plt.Line2D([], [], color=COL[g], marker="o", lw=2, label=f"Contexto {g}") for g in COL]
ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
ax.set_title("EXP-026 · Reemplazar la imagen por su vecindario vs. añadirlo",
             fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(OUT / "exp026_delta_accuracy.png", facecolor=SURF)
print(OUT / "exp026_delta_accuracy.png")
