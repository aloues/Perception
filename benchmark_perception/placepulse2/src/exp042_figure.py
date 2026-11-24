"""
EXP-042 — figura: Δ pairwise accuracy de cada modelo de grafo frente a la imagen sola (E0), con IC 95 %
agrupado por imagen piloto. Lee outputs/exp042/paired_vs_baselines.csv (cifras reales).

Uso:
    python exp042_figure.py
Salida: outputs/exp042/exp042_delta_vs_E0.png
"""

from __future__ import annotations

import json

import pandas as pd

import figstyle
from utils import OUTPUTS

OUT = OUTPUTS / "exp042"
LABELS = {"G0-ego0": "G0 · barrio 0", "G0-ego1": "G0 · barrio 1", "G0-ego2": "G0 · barrio 2",
          "G0-rand": "G0 · calles al azar", "G0-radius": "G0 · radio (r en val)", "G0-bovw": "G0 · BoVW barrio 1",
          "G1-gcn": "G1 · GCN", "G1-noedge": "G1 · GCN sin aristas", "G2-crf": "G2 · CRF (λ en val)"}


def main() -> None:
    import matplotlib.pyplot as plt
    figstyle.setup()
    p = pd.read_csv(OUT / "paired_vs_baselines.csv")
    p = p[p["comparison"].str.endswith(" - E0")].copy()
    p["model"] = p["comparison"].str.replace(" - E0", "", regex=False)
    p = p[p["model"].isin(LABELS)].set_index("model").loc[list(LABELS)]
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = range(len(p))[::-1]
    colors = [figstyle.SERIES[0] if m.startswith("G0") else figstyle.SERIES[1] if m.startswith("G1")
              else figstyle.SERIES[2] for m in p.index]
    for yi, (m, r), c in zip(y, p.iterrows(), colors):
        ax.plot([100 * r["ci_low_grouped"], 100 * r["ci_high_grouped"]], [yi, yi], color=c, lw=2)
        ax.plot(100 * r["delta_acc"], yi, "o", color=c, markersize=8)
    ax.axvline(0, color=figstyle.INK2, lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABELS[m] for m in p.index])
    ax.set_xlabel("Δ precisión por par vs imagen sola (puntos porcentuales, IC 95 %)")
    ax.set_title(f"Place Pulse, {summ['n_test_touch']:,} pares de test en Atlanta, Berlin y Santiago")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "exp042_delta_vs_E0.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
