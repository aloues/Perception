"""
Figura reutilizable (tesis / PPT / póster): cómo el modelo convierte una foto de calle en una decisión de seguridad
PERCIBIDA, y qué partes están congeladas, cuáles se entrenan y qué se usa solo en entrenamiento vs evaluación.

Corresponde al código real: encoders.py (E_θ congelado, L2), ranking_head.py (s = wᵀz sin bias, logit = s_i − s_j),
train_probe.py (BCEWithLogits, AdamW), evaluate.py (exactitud por pares). El contexto de grafo es OPCIONAL
(EXP-026/042/048: aditivo, [z_i, h_i]); no forma parte del modelo principal de foto.

Uso:  python src/make_fig_pipeline_decision.py
Salida: deliverables/figures/pipeline_decision.{pdf,svg,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "deliverables" / "figures"

# paleta del proyecto (figstyle.py), segura para daltónicos en este orden
BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#9b9a95", "#fcfcfb"
FROZEN_FC, TRAIN_FC, HUMAN_FC, OUT_FC = "#eeeeec", "#e3eefb", "#fdeee6", "#e5f6ef"


def box(ax, x, y, w, h, title, sub=None, fc="white", ec=INK2, ls="-", lw=1.6, tsize=13, ssize=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=ssize, color=INK2,
                zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=3)


def arrow(ax, p0, p1, color=INK2, ls="-", lw=1.6, rad=0.0, head=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.35, 8.55, "From a street image to a perceived-safety decision", fontsize=18, weight="bold",
            color=INK, va="center")
    ax.text(0.35, 8.08, "Pairwise task (Place Pulse 2.0): a person chose which of two images looks safer. "
            "The model must reproduce that choice.", fontsize=11.5, color=INK2, va="center")

    yi, yj, hh = 5.75, 3.55, 1.25          # filas de la imagen i y de la imagen j
    # --- imágenes
    for y, lab in ((yi, "i"), (yj, "j")):
        box(ax, 0.35, y, 1.75, hh, f"Image {lab}", "street view\n224×224 px", fc="white")
    # --- encoder compartido y congelado
    box(ax, 2.6, yj, 2.3, yi + hh - yj, "Encoder  $E_\\theta$",
        "DINOv2 ViT-B/14\n(DINOv3 / I-JEPA\nin EXP-064)\n\nFROZEN\nshared by i and j", fc=FROZEN_FC, ec=GRID)
    # --- embeddings
    for y, lab in ((yi, "i"), (yj, "j")):
        box(ax, 5.4, y, 2.0, hh, f"$z_{lab}\\in\\mathbb{{R}}^{{768}}$", "L2-normalised\nembedding", fc="white",
            tsize=14)
        arrow(ax, (2.1, y + hh / 2), (2.6, y + hh / 2))
        arrow(ax, (4.9, y + hh / 2), (5.4, y + hh / 2))
    # --- contexto de grafo opcional
    box(ax, 7.85, yj, 2.15, yi + hh - yj, "Graph context",
        "(optional)\n$h_i=\\mathrm{AGG}(z_i,\\{z_k\\}_{k\\in N(i)})$\nego-k mean / GCN\n\ninput $=[z_i, h_i]$\n"
        "helps PLACE target,\nnot the photo vote", fc="white", ec=GRID, ls="--", tsize=12, ssize=9.8)
    # --- cabeza entrenada
    box(ax, 10.45, yj, 2.0, yi + hh - yj, "Head  $g_\\phi$",
        "linear: $s=w^{\\top}z$\n(768 weights, no bias)\n\nEXP-063 also tests\nMLP, kNN, SVM,\nRF, XGBoost\n\nTRAINED",
        fc=TRAIN_FC, ec=BLUE, lw=2.0, ssize=9.8)
    for y in (yi, yj):
        arrow(ax, (7.4, y + hh / 2), (10.45, y + hh / 2))
    # --- puntajes
    for y, lab in ((yi, "i"), (yj, "j")):
        box(ax, 12.9, y + 0.15, 1.15, hh - 0.3, f"$s_{lab}$", None, fc="white", tsize=16)
        arrow(ax, (12.45, y + hh / 2), (12.9, y + hh / 2))
    ax.text(13.475, yi - 0.12, "looks-safe\nscore\n(not crime)", ha="center", va="top", fontsize=9.5,
            color=INK2)
    # --- probabilidad del par
    box(ax, 14.35, 4.15, 1.45, 2.25, "$\\sigma(s_i-s_j)$", "$=P(i \\succ j)$\n\nchoose i\nif $s_i>s_j$",
        fc=OUT_FC, ec=GREEN, lw=2.0, tsize=13.5, ssize=10.5)
    arrow(ax, (14.05, yi + hh / 2), (14.35, 5.6))
    arrow(ax, (14.05, yj + hh / 2), (14.35, 4.95))

    # --- fila inferior: entrenamiento vs evaluación
    yb, hb = 0.45, 2.05
    box(ax, 0.35, yb, 4.3, hb, "Human vote  $y_{ij}\\in\\{0,1\\}$",
        "1 if the volunteer chose i as safer\n(ties removed from training)\nno 0–1 score per image is given",
        fc=HUMAN_FC, ec=ORANGE, lw=2.0, ssize=10.5)
    box(ax, 5.15, yb, 5.0, hb, "TRAINING  (train pairs)",
        "$\\mathcal{L}=-y\\log p-(1-y)\\log(1-p)$,  $p=\\sigma(s_i-s_j)$\nAdamW on $\\phi$ only; early stop on val loss\n"
        "encoder never updated", fc=TRAIN_FC, ec=BLUE, lw=2.0, ssize=10.5)
    box(ax, 10.65, yb, 5.15, hb, "EVALUATION  (test pairs, once)",
        "pairwise accuracy $=\\frac{1}{N}\\sum 1[(s_i>s_j)=y_{ij}]$\nchance 0.50 · DINOv2+linear 0.679\n"
        "(image_disjoint, N = 34,607 pairs)", fc=OUT_FC, ec=GREEN, lw=2.0, ssize=10.5)
    arrow(ax, (4.65, yb + hb / 2), (5.15, yb + hb / 2), color=ORANGE)
    arrow(ax, (15.3, 4.15), (14.6, yb + hb), color=GREEN)
    # retropropagación: el error del par (BCE) solo actualiza la cabeza
    arrow(ax, (9.3, yb + hb), (10.95, yj), color=BLUE, ls=(0, (4, 3)), lw=2.0)
    ax.text(9.05, 3.0, "backprop updates $\phi$ only", fontsize=10.5, color=BLUE, ha="right", style="italic")

    # --- leyenda
    lx, ly = 0.35, 7.25
    for k, (fc, ec, ls, lab) in enumerate(((FROZEN_FC, GRID, "-", "frozen (pre-trained, SSL)"),
                                           (TRAIN_FC, BLUE, "-", "trained on Place Pulse votes"),
                                           ("white", GRID, "--", "optional / ablation only"),
                                           (HUMAN_FC, ORANGE, "-", "human supervision"),
                                           (OUT_FC, GREEN, "-", "prediction & metric"))):
        x = lx + k * 3.1
        ax.add_patch(FancyBboxPatch((x, ly), 0.42, 0.3, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=fc, ec=ec, ls=ls, lw=1.6))
        ax.text(x + 0.55, ly + 0.15, lab, va="center", fontsize=10.5, color=INK2)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"pipeline_decision.{ext}", dpi=300 if ext == "png" else None)
    print("ok", OUT)


if __name__ == "__main__":
    main()
