"""
Figuras para el PPT post-asesoría (re-ejecutable; nunca inventa datos).

1. EXP-064 forest plot: Δ de exactitud por pares vs DINOv2 (misma cabeza lineal-BT, mismos pares) por encoder ×
   protocolo, IC 95 % bootstrap de imágenes. Lee outputs/exp064/paired_vs_dinov2.csv; si no existe, avisa y sale sin
   escribir la figura (el PPT muestra entonces "IN PROGRESS").
   → deliverables/figures/exp064_encoders_delta.{png,pdf}
2. Fotos del ejemplo trabajado (pares reales de test, outputs/exp063/worked_example.json) extraídas del zip de
   Place Pulse 2.0 para componer la diapositiva con texto nativo.
   → deliverables/presentations/fig/we_<caso>_<left|right>.jpg

Paleta: la misma validada en src/exp063_analyze.py (dataviz validate_palette, modo claro).
Uso: python src/make_fig_exp064.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, REPO_ROOT, ZIP_PATH  # noqa: E402

FIG = REPO_ROOT / "deliverables" / "figures"
PFIG = REPO_ROOT / "deliverables" / "presentations" / "fig"
PROT_COLOR = {"image_disjoint": "#2a78d6", "city_holdout": "#eb6834", "paper_protocol": "#1baf7a"}
PROT_MARK = {"image_disjoint": "o", "city_holdout": "s", "paper_protocol": "^"}
PROT_LABEL = {"image_disjoint": "image_disjoint (new photos)", "city_holdout": "city_holdout (new cities)",
              "paper_protocol": "paper_protocol (historical)"}
ENC = {"dinov3_vitb16_cls": "DINOv3 ViT-B/16 · CLS", "dinov3_vitb16_meanpatch": "DINOv3 ViT-B/16 · mean patch",
       "ijepa_vith14_meanpatch": "I-JEPA ViT-H/14 · mean patch", "imagenet_vit_b16": "ImageNet ViT-B/16 (prior)",
       "ours_simclr_vit_b16": "SimCLR ViT-B/16 (inherited)", "lejepa_resnet18_pilot": "LeJEPA ResNet-18 pilot",
       "resnet18_random_init": "ResNet-18 random init"}


def fig_exp064() -> None:
    f = OUTPUTS / "exp064" / "paired_vs_dinov2.csv"
    if not f.exists():
        print(f"[make_fig_exp064] falta {f}: no se dibuja (el PPT mostrará IN PROGRESS)")
        return
    d = pd.read_csv(f)
    d = d[d.encoder != "dinov2_vit_b14"].dropna(subset=["delta_vs_dinov2"])
    if d.empty:
        print("[make_fig_exp064] CSV sin comparaciones todavía")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    encs = [e for e in ENC if e in set(d.encoder)]
    prots = [p for p in PROT_COLOR if p in set(d.protocol)]
    off = {p: (i - (len(prots) - 1) / 2) * 0.22 for i, p in enumerate(prots)}
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(encs) + 2.2))
    for i, e in enumerate(encs[::-1]):
        for p in prots:
            r = d[(d.encoder == e) & (d.protocol == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + off[p]
            ax.plot([100 * r.delta_img_lo, 100 * r.delta_img_hi], [y, y], color=PROT_COLOR[p], lw=2,
                    solid_capstyle="round")
            ax.plot(100 * r.delta_vs_dinov2, y, PROT_MARK[p], color=PROT_COLOR[p], ms=8, mec="white", mew=1.5)
            ax.text(100 * r.delta_img_hi + 0.08, y, f"{100 * r.delta_vs_dinov2:+.2f}", va="center", fontsize=8.5,
                    color="#3b3b38")
    ax.axvline(0, color="#8a8a85", lw=1, ls="--")
    for v in (-0.5, 0.5):
        ax.axvline(v, color="#c9c9c4", lw=0.8, ls=":")
    ax.set_yticks(range(len(encs)))
    ax.set_yticklabels([ENC[e] for e in encs[::-1]], fontsize=10, color="#1b1b19")
    ax.set_xlabel("Δ pairwise accuracy vs frozen DINOv2 ViT-B/14 (points; 95% CI, image-level bootstrap)", fontsize=10)
    ax.set_title("EXP-064 · same linear Bradley–Terry head, same test pairs, seed 42", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(encs) - 0.3)
    handles = [Line2D([], [], color=PROT_COLOR[p], marker=PROT_MARK[p], lw=2, ms=7, label=PROT_LABEL[p])
               for p in prots]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=len(prots), fontsize=8.5,
              frameon=False)
    fig.text(0.01, 0.005, "Dashed: no difference. Dotted: ±0.5 pt pre-registered minimum effect. "
             "Source: outputs/exp064/paired_vs_dinov2.csv", fontsize=7.5, color="#5e5e59")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"exp064_encoders_delta.{ext}", dpi=220)
    plt.close(fig)
    print("[make_fig_exp064] escrito", FIG / "exp064_encoders_delta.png")


def worked_example_assets() -> None:
    f = OUTPUTS / "exp063" / "worked_example.json"
    if not f.exists() or not ZIP_PATH.exists():
        print("[make_fig_exp064] sin worked_example.json o zip: no se exportan fotos")
        return
    we = json.loads(f.read_text(encoding="utf-8"))
    PFIG.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for case, e in we["pairs"].items():
            for side in ("left", "right"):
                im = Image.open(io.BytesIO(zf.read(e[f"{side}_image"]))).convert("RGB")
                im.save(PFIG / f"we_{case}_{side}.jpg", quality=90)
    print("[make_fig_exp064] fotos del ejemplo en", PFIG)


if __name__ == "__main__":
    fig_exp064()
    worked_example_assets()
