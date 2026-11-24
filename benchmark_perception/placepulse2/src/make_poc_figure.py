"""
FASE D — Figura IMAGEN -> CONCEPTO -> DETECCION -> REPRESENTACION SEMANTICA.

Usa SOLO detecciones guardadas (poc_detections.csv, gdino_raw_detections.json);
no vuelve a ejecutar ningun modelo. Las imagenes se eligen de forma determinista:
para cada ciudad distinta, la imagen con mas conceptos detectados por Grounding DINO
(score >= umbral), hasta N_EXAMPLES. Es una seleccion ILUSTRATIVA, no una muestra
aleatoria: se declara en el pie de figura. Las metricas del PoC salen de las 72
imagenes completas (poc_metrics.json).

Salidas: outputs/figures/fig_poc_imagen_concepto_deteccion.png
         outputs/poc/poc_examples.csv  (imagen, concepto, herramienta, score, bbox)
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_poc_concepts import DETECT_PROMPTS
from utils import OUTPUTS, log

POC = OUTPUTS / "poc"
FIG = OUTPUTS / "figures"
GD_THR = 0.35
N_EXAMPLES = 4
COL = {"gdino": "#e4572e", "yoloworld": "#2e86ab"}


def main() -> None:
    det = pd.read_csv(POC / "poc_detections.csv")
    raw = {}
    for r in json.loads((POC / "gdino_raw_detections.json").read_text()):
        raw.setdefault(r["location_id"], r)          # primera aparicion (ignora reproceso determinismo)
    inv = {v: k for k, v in DETECT_PROMPTS.items()}

    g = det[det.tool == "gdino"]
    n_c = g[g.score >= GD_THR].groupby("location_id")["concept"].nunique()
    city = det.drop_duplicates("location_id").set_index("location_id")["city"]
    order = n_c.sort_values(ascending=False)
    chosen, seen = [], set()
    for lid, n in order.items():
        if city[lid] in seen:
            continue
        seen.add(city[lid])
        chosen.append(lid)
        if len(chosen) == N_EXAMPLES:
            break

    fig, axes = plt.subplots(N_EXAMPLES, 4, figsize=(18, 3.6 * N_EXAMPLES),
                             gridspec_kw={"width_ratios": [1.15, 0.75, 1.15, 1.0]})
    heads = ["1  IMAGEN", "2  CONCEPTO -> prompt", "3  DETECCION (bbox+conf.)",
             "4  VECTOR SEMANTICO"]
    for j, h in enumerate(heads):
        axes[0, j].set_title(h, fontsize=11, fontweight="bold", loc="left")

    ex_rows = []
    for i, lid in enumerate(chosen):
        im = Image.open(POC / "images" / f"{lid}.jpg").convert("RGB")
        axes[i, 0].imshow(im); axes[i, 0].axis("off")
        axes[i, 0].text(0.01, 0.02, city[lid], transform=axes[i, 0].transAxes, color="white",
                        fontsize=10, bbox=dict(facecolor="black", alpha=0.55, pad=2))

        ax = axes[i, 1]; ax.axis("off")
        d = det[det.location_id == lid]
        lines = [f"{c}  ->  \"{p}\"" for c, p in DETECT_PROMPTS.items()]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace")

        ax = axes[i, 2]; ax.imshow(im); ax.axis("off")
        r = raw.get(lid)
        if r:
            for lb, sc, bb in zip(r["labels"], r["scores"], r["boxes"]):
                if lb in inv and sc >= GD_THR:
                    x0, y0, x1, y1 = bb
                    if y0 > 0.85 * im.size[1] and x0 < 0.25 * im.size[0]:
                        continue   # marca de agua Google
                    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                   ec=COL["gdino"], lw=1.6))
                    ax.text(x0, max(y0 - 3, 8), f"{lb} {sc:.2f}", color="white", fontsize=7,
                            bbox=dict(facecolor=COL["gdino"], alpha=0.8, pad=1))
                    ex_rows.append({"location_id": lid, "city": city[lid], "concept": inv[lb],
                                    "tool": "gdino", "score": sc, "bbox": ",".join(map(str, bb))})
        yw = d[(d.tool == "yoloworld") & (d.score >= 0.25) & d.bbox.notna()]
        for _, rr in yw.iterrows():
            x0, y0, x1, y1 = map(float, str(rr.bbox).split(","))
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           ec=COL["yoloworld"], lw=1.6, ls="--"))
            ex_rows.append({"location_id": lid, "city": city[lid], "concept": rr.concept,
                            "tool": "yoloworld", "score": rr.score, "bbox": rr.bbox})
        if i == N_EXAMPLES - 1:
            ax.text(0, -0.04, "rojo continuo: Grounding DINO (>=%.2f)   azul discontinuo: YOLO-World (>=0.25)" % GD_THR,
                    transform=ax.transAxes, fontsize=8, va="top")

        ax = axes[i, 3]
        concepts = list(DETECT_PROMPTS)
        piv = d.pivot_table(index="concept", columns="tool", values="score").reindex(concepts)
        y = np.arange(len(concepts)); h = 0.27
        for k, t in enumerate(("gdino", "yoloworld", "clip")):
            if t in piv:
                ax.barh(y + (k - 1) * h, piv[t].fillna(0), h, label=t,
                        color={"gdino": COL["gdino"], "yoloworld": COL["yoloworld"],
                               "clip": "#8d99ae"}[t])
        ax.set_yticks(y); ax.set_yticklabels(concepts, fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.tick_params(axis="x", labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("PoC de conceptos: imagen -> concepto del codebook -> deteccion -> vector semantico\n"
                 "(ilustrativo: por ciudad, la imagen con mas conceptos detectados; sin anotacion humana no hay precision/recall; marca de agua Google enmascarada)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG.mkdir(exist_ok=True)
    out = FIG / "fig_poc_imagen_concepto_deteccion.png"
    fig.savefig(out, dpi=130)
    pd.DataFrame(ex_rows).to_csv(POC / "poc_examples.csv", index=False)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
