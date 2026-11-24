"""EXP-028 — figura: fidelidad modelo–humano según la escala espacial, frente al techo de ruido humano.

Uso:  python exp028_figure.py      Salida: outputs/exp028/exp028_scale.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import OUTPUTS  # noqa: E402

OUT = OUTPUTS / "exp028"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

r = pd.read_csv(OUT / "neighborhood_scale.csv")
r = r[r["protocol"] == "city_holdout"]
labels = {"imagen": "Imagen", "celda_250m": "250 m", "celda_500m": "500 m", "celda_1000m": "1 km", "celda_2000m": "2 km"}
r = r.set_index("level").loc[list(labels)]
x = np.arange(len(r))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, (col, lo, hi, rel, title) in zip(axes, [
        ("rho", "rho_ci_low", "rho_ci_high", "reliability", "Todas las celdas"),
        ("rho_within_city", "rho_within_ci_low", "rho_within_ci_high", "reliability_within_city",
         "Dentro de cada ciudad (sin efecto ciudad)")]):
    ax.set_facecolor(SURF)
    ceil = np.sqrt(r[rel].to_numpy())
    ax.plot(x, ceil, color=ORANGE, lw=2, marker="o", ms=7, mec=SURF, mew=2, label="Techo: acuerdo humano consigo mismo")
    ax.plot(x, r[col], color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=2, label="Modelo vs. juicio humano")
    ax.vlines(x, r[lo], r[hi], color=BLUE, lw=1.2)
    for xi, v in zip(x, r[col]):
        ax.text(xi, v - 0.075, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, list(labels.values()), fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    ax.set_ylim(0.3, 1.0)
axes[0].set_ylabel("ρ de Spearman", fontsize=8, color=INK2)
axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
fig.suptitle("EXP-028 · Ciudades NO vistas: la predicción se acerca al techo humano al pasar de imagen a barrio",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "exp028_scale.png", facecolor=SURF)
print(OUT / "exp028_scale.png")
