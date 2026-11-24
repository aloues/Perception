"""
Figura EXP-066 (re-ejecutable; solo lee outputs):
  (a) escalera de exactitud por pares en fotos nuevas (`image_disjoint`): azar, encoders con la misma cabeza lineal
      (incluye los de EXP-067 si ya existen), ajuste fino (EXP-061), techo del modelo latente con IC (EXP-066), techo
      sabiendo solo la ciudad y oráculo en muestra (EXP-055).
  (b) cómo se estimó τ: estadístico retenido simulado S(τ) vs el observado (método de momentos simulados).
→ deliverables/figures/exp066_noise_ceiling.{png,pdf}            (inglés, paneles a+b, EC1/póster)
→ deliverables/figures/es/techo_voto.png                          (español, solo panel a, PPT)
Paleta y ejes: src/figstyle.py (paleta validada, orden fijo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from utils import OUTPUTS, REPO_ROOT, RUNS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG = REPO_ROOT / "deliverables" / "figures"
ENCS = [  # (encoder, etiqueta en, etiqueta es)
    ("ours_simclr_vit_b16", "SimCLR (inherited)", "SimCLR (heredado)"),
    ("ijepa_vith14_meanpatch", "I-JEPA ViT-H IN-1k", "I-JEPA ViT-H IN-1k"),
    ("ijepa_vith14_in22k_meanpatch", "I-JEPA ViT-H IN-22k", "I-JEPA ViT-H IN-22k"),
    ("vjepa2_vitl16_meanpatch", "V-JEPA 2 ViT-L", "V-JEPA 2 ViT-L"),
    ("imagenet_vit_b16", "ImageNet ViT-B", "ImageNet ViT-B"),
    ("dinov2_vit_b14", "DINOv2 ViT-B (thesis)", "DINOv2 ViT-B (tesis)"),
    ("dinov2_vitl14_cls", "DINOv2 ViT-L", "DINOv2 ViT-L"),
    ("dinov3_vitb16_cls", "DINOv3 ViT-B", "DINOv3 ViT-B"),
]
TXT = {
    "en": {"chance": "Chance", "ft": "DINOv2 fine-tuned (K=2)", "ceil": "Latent ceiling (EXP-066)",
           "city": "Ceiling knowing only the city", "oracle": "In-sample oracle (sees test votes)",
           "x": "Pairwise accuracy on new images (34,607 test pairs)",
           "title": "(a) DINOv2 reaches {e:.0f}% of the range the votes allow"},
    "es": {"chance": "Azar", "ft": "DINOv2 con ajuste fino (K=2)", "ceil": "Techo latente del voto (EXP-066)",
           "city": "Techo sabiendo solo la ciudad", "oracle": "Oráculo en muestra (ve los votos de test)",
           "x": "Exactitud por pares en fotos nuevas (34,607 pares de test)",
           "title": "DINOv2 alcanza el {e:.0f} % del rango que permiten los votos"},
}


def acc(enc: str, prot: str = "image_disjoint", key: str = "test") -> float | None:
    m = RUNS / f"{enc}__{prot}__linear__seed42" / "metrics.json"
    return json.loads(m.read_text())["test"][key]["pairwise_accuracy"] if m.exists() else None


def ladder(ax, r: dict, lang: str) -> None:
    t = TXT[lang]
    rows = [(t["chance"], 0.5, figstyle.NEUTRAL)]
    for enc, en, es in ENCS:
        a = acc(enc)
        if a is not None:
            rows.append((en if lang == "en" else es, a, figstyle.SERIES[0]))
    ft = OUTPUTS / "exp061" / "paired_vs_frozen.csv"
    if ft.exists():
        f = pd.read_csv(ft)
        f = f[f.protocol == "image_disjoint"]
        if len(f):
            rows.append((t["ft"], float(f.acc_finetune.iloc[0]), figstyle.SERIES[0]))
    rows.append((t["ceil"], r["ceiling"], figstyle.SERIES[1]))
    rows.append((t["city"], r["ceiling_city_only"], figstyle.NEUTRAL))
    rows.append((t["oracle"], r["oracle_in_sample_observed"], figstyle.NEUTRAL))
    rows.sort(key=lambda x: x[1])
    lo, hi = r["ceiling_ci95"]
    ax.axvspan(lo, hi, color=figstyle.SERIES[1], alpha=0.15, lw=0)
    for i, (lab, a, col) in enumerate(rows):
        ax.plot([0.5, a], [i, i], color=figstyle.GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([a], [i], s=60, color=col, zorder=3, edgecolor=figstyle.SURFACE, linewidth=2)
        ax.text(a + 0.004, i, f"{a:.3f}", va="center", fontsize=8, color=figstyle.INK2)
    ax.set_yticks(np.arange(len(rows)), [x[0] for x in rows])
    ax.set_xlim(0.49, 0.80)
    ax.set_xlabel(t["x"])
    ax.grid(axis="y", visible=False)
    ax.set_title(t["title"].format(e=100 * r["efficiency_above_chance"]), loc="left")


def main() -> None:
    figstyle.setup()
    j = json.loads((OUTPUTS / "exp066" / "summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(OUTPUTS / "exp066" / "grid.csv")
    r = j["links"]["logit_gauss"]["protocols"]["image_disjoint"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ladder(ax, r, "en")
    g = grid[grid.variant == "logit_gauss"].groupby("tau")["S"].mean().reset_index()
    bx.plot(g["tau"], g["S"], color=figstyle.SERIES[0], marker="o", markersize=5)
    tau = j["links"]["logit_gauss"]["tau_hat"]
    bx.axhline(j["S_obs"], color=figstyle.SERIES[1], lw=1.5)
    bx.axvline(tau, color=figstyle.INK2, lw=1, ls=(0, (3, 3)))
    bx.text(g["tau"].min(), j["S_obs"] + 0.004, f"observed S = {j['S_obs']:.3f}", fontsize=8, color=figstyle.INK2)
    bx.text(tau + 0.02, g["S"].min() + 0.005, f"$\\hat\\tau$ = {tau:.2f}", fontsize=8, color=figstyle.INK2)
    bx.set_xlabel(r"spread of image effects $\tau$ (logit units)")
    bx.set_ylabel("held-out vote accuracy of a BT fit")
    bx.set_title("(b) Estimating $\\tau$ by simulated moments", loc="left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"exp066_noise_ceiling.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ladder(ax, r, "es")
    fig.tight_layout()
    (FIG / "es").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "es" / "techo_voto.png", dpi=200)
    print("ok", FIG / "exp066_noise_ceiling.png", FIG / "es" / "techo_voto.png")


if __name__ == "__main__":
    main()
