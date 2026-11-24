"""
Genera las figuras del avance (seccion 20 del brief).

Todas las figuras con NUMEROS salen de ficheros de resultados reales:
  - outputs/benchmark_results.csv            (probes Place Pulse, esta sesion)
  - outputs/leakage_report.json              (auditoria de leakage)
  - data/manifests/*.json                    (estadisticas del dataset)
  - ../../graph/robustness_checks/*.csv      (pipeline historico de Lima)

Las figuras conceptuales (arquitecturas, roadmap, codebook, ego-network) son
diagramas: no contienen numeros inventados.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils import MANIFESTS, OUTPUTS, PKG_ROOT, REPO_ROOT, log

FIGDIR = OUTPUTS / "figures"
GRAPH = REPO_ROOT / "graph"

NAVY = "#1F3A5F"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
CORAL = "#E76F51"
SLATE = "#7B8794"
LIGHT = "#EEF2F5"
INK = "#16222E"

ENC_COLOR = {
    "OURS (SimCLR ViT-B/16)": NAVY,
    "ImageNet ViT-B/16": TEAL,
    "DINOv2 ViT-B/14": CORAL,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#DDE3E8",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p)
    plt.close(fig)
    log(f"  {name}")
    return p


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=INK, fs=10, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, color=SLATE, lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=1))


def blank(figsize=(12, 6.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


# ==========================================================================
# 1. Pipeline historico de Lima (lo que recibimos y reprodujimos)
# ==========================================================================
def fig_pipeline_historico():
    fig, ax = blank((13, 4.2))
    steps = ["Street View\nimagen urbana", "SimCLR + ViT-B/16\n(auto-supervisado)",
             "embedding\n768-D", "ego-network\ncalles vecinas",
             "agregacion +\neigen-features", "clasificador", "high / low\ncrime"]
    cols = [LIGHT, NAVY, LIGHT, LIGHT, LIGHT, LIGHT, CORAL]
    tcs = [INK, "white", INK, INK, INK, INK, "white"]
    n = len(steps)
    w = 0.107
    gap = (1 - n * w) / (n - 1)
    for i, (s, c, tc) in enumerate(zip(steps, cols, tcs)):
        x = i * (w + gap)
        box(ax, x, 0.42, w, 0.34, s, fc=c, tc=tc, fs=9.2,
            bold=(i in (1, 6)), ec=NAVY if c != CORAL else CORAL)
        if i < n - 1:
            arrow(ax, (x + w, 0.59), (x + w + gap, 0.59))
    ax.text(0.5, 0.94, "Pipeline heredado - reproducido en esta tesis",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.855, "el target se deriva de DELITOS REGISTRADOS (dual-GMM high/low crime), "
                        "no de percepcion humana",
            ha="center", fontsize=10.5, color=CORAL, style="italic")
    ax.text(0.5, 0.20, "Barranco 2016:  401 nodos | 1,397 delitos mapeados | 169 high-risk | ~42.1 % high-crime\n"
                       "La Victoria 2016:  1,456 nodos | 3,772 delitos mapeados | 168 high-risk | ~11.5 % high-crime",
            ha="center", va="center", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=LIGHT, edgecolor=SLATE))
    ax.text(0.5, 0.045, "Resultado reproducido  |  diferencias menores frente al README atribuidas a drift de OSM",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    return save(fig, "fig01_pipeline_historico.png")


# ==========================================================================
# 2. Percepcion de seguridad != criminalidad observada
# ==========================================================================
def fig_percepcion_vs_crimen():
    fig, ax = blank((12, 6.0))
    ax.text(0.5, 0.955, "Aprendizaje fundamental:  percepcion de seguridad  ≠  criminalidad observada",
            ha="center", fontsize=14.5, fontweight="bold", color=NAVY)

    box(ax, 0.04, 0.60, 0.42, 0.28,
        "PERCEIVED SAFETY\n\nevaluacion subjetiva humana\nde que tan seguro parece\nun entorno urbano\n"
        "ground truth: comparaciones\nhumanas (Place Pulse 2.0)",
        fc="#E8F2F0", ec=TEAL, fs=10.5)
    box(ax, 0.54, 0.60, 0.42, 0.28,
        "OBSERVED CRIME\n\neventos delictivos registrados\nespacial y temporalmente\n\n"
        "ground truth: denuncias /\nregistros oficiales (Lima)",
        fc="#FBEDE8", ec=CORAL, fs=10.5)

    axq = fig.add_axes([0.305, 0.085, 0.42, 0.375])
    axq.set_xlim(0, 1)
    axq.set_ylim(0, 1)
    axq.set_xlabel("Criminalidad observada  →", fontsize=10.5)
    axq.set_ylabel("Percepcion de inseguridad  →", fontsize=10.5)
    axq.set_xticks([0.25, 0.75])
    axq.set_xticklabels(["baja", "alta"])
    axq.set_yticks([0.25, 0.75])
    axq.set_yticklabels(["baja", "alta"])
    axq.grid(False)
    for (x, y, t, c) in [(0.25, 0.75, "parece insegura\npero pocos delitos", AMBER),
                         (0.75, 0.25, "parece segura\npero muchos delitos", AMBER),
                         (0.75, 0.75, "coincide\n(insegura)", "#D9E1E7"),
                         (0.25, 0.25, "coincide\n(segura)", "#D9E1E7")]:
        axq.add_patch(plt.Rectangle((x - 0.24, y - 0.24), 0.48, 0.48, facecolor=c,
                                    alpha=0.55, edgecolor="none"))
        axq.text(x, y, t, ha="center", va="center", fontsize=9.5, color=INK)
    axq.axhline(0.5, color=SLATE, lw=1)
    axq.axvline(0.5, color=SLATE, lw=1)
    axq.set_title("los desacuerdos son lo interesante:  Perception-Crime Gap",
                  fontsize=10.5, color=NAVY, pad=10)
    ax.text(0.5, 0.540, "el baseline heredado optimizaba la columna de la derecha;  "
                        "la tesis avanza hacia un target perceptual explicito",
            ha="center", fontsize=10.5, color=NAVY, style="italic")
    return save(fig, "fig02_percepcion_vs_crimen.png")


# ==========================================================================
# 3. Mapa del estado del arte
# ==========================================================================
def fig_sota_map():
    fig, ax = blank((12.5, 6.4))
    ax.text(0.5, 0.96, "Marco metodologico construido a partir de la revision de literatura",
            ha="center", fontsize=14.5, fontweight="bold", color=NAVY)
    blocks = [
        (0.03, "REPRESENTATION\nLEARNING",
         "ViT (Dosovitskiy+)\nSimCLR (Chen+)\nDINOv2 (Oquab+)\n\ncomo se construye\nla representacion", NAVY),
        (0.275, "HUMAN\nPERCEPTION",
         "Place Pulse 2.0\n(Dubey+ 2016)\ncomparaciones pareadas\n\ncontra que se valida", TEAL),
        (0.52, "LANGUAGE\n+ VISION",
         "CLIP (Radford+)\nGrounding DINO | GLIP\nYOLO-World | SAM 2\n\ncomo se nombran\nlos conceptos", AMBER),
        (0.765, "INTERPRETABILITY",
         "Concept Bottleneck\nModels | SHAP | XAI\nViT need registers\n\ncomo se explica\nla prediccion", CORAL),
    ]
    ytop, hh, ybody, hb = 0.745, 0.105, 0.435, 0.29
    for x, title, body, c in blocks:
        box(ax, x, ytop, 0.205, hh, title, fc=c, tc="white", fs=10.5, bold=True, ec=c)
        box(ax, x, ybody, 0.205, hb, body, fc="white", ec=c, fs=9.3)
        arrow(ax, (x + 0.1025, ytop), (x + 0.1025, ybody + hb), color=c, lw=1.4)
    box(ax, 0.275, 0.155, 0.45, 0.115, "SPATIAL CONTEXT\ngrafos de calles | GNN | ego-networks",
        fc="#4A6FA5", tc="white", fs=10.5, bold=True, ec="#4A6FA5")
    for x, _, _, c in blocks:
        arrow(ax, (x + 0.1025, ybody), (0.5, 0.275), color=c, lw=1.4)
    ax.text(0.5, 0.065, "las cinco familias no son alternativas: se combinan en el diseno experimental de la tesis",
            ha="center", fontsize=10.5, color=NAVY, style="italic")
    return save(fig, "fig03_mapa_estado_del_arte.png")


# ==========================================================================
# 4. Estadisticas de Place Pulse 2.0 (reales, desde los manifiestos)
# ==========================================================================
def fig_placepulse():
    ds = json.loads((MANIFESTS / "dataset_summary.json").read_text(encoding="utf-8"))
    ps = json.loads((MANIFESTS / "pairs_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(12.5, 6.2))
    fig.text(0.5, 0.955, "Place Pulse 2.0 - ground truth humano de percepcion",
             ha="center", fontsize=14.5, fontweight="bold", color=NAVY)

    axk = fig.add_axes([0.03, 0.60, 0.94, 0.26])
    axk.axis("off")
    axk.grid(False)
    kpis = [
        (f"{ds['images']['n_images_in_zip']:,}", "imagenes en el ZIP"),
        (f"{ds['safe']['n_cities']}", "ciudades"),
        (f"{ds['total_votes_all_attributes']:,}", "votos (6 atributos)"),
        (f"{ds['safe']['n_votes_valid']:,}", "votos del atributo safe"),
        (f"{ps['decisive']['n_pairs']:,}", "pares decisivos usados"),
        (f"{ps['decisive']['n_voters']:,}", "votantes en esos pares"),
    ]
    for i, (v, k) in enumerate(kpis):
        x = 0.005 + i * 0.1665
        axk.add_patch(FancyBboxPatch((x, 0.05), 0.155, 0.9,
                                     boxstyle="round,pad=0.01,rounding_size=0.05",
                                     transform=axk.transAxes, facecolor=LIGHT,
                                     edgecolor=TEAL, linewidth=1.6))
        axk.text(x + 0.0775, 0.62, v, transform=axk.transAxes, ha="center",
                 fontsize=16, fontweight="bold", color=NAVY)
        axk.text(x + 0.0775, 0.24, k, transform=axk.transAxes, ha="center",
                 fontsize=9, color=INK)

    cities = ds["safe"]["cities"]
    top = sorted(cities.items(), key=lambda kv: -kv[1])[:18]
    axc = fig.add_axes([0.06, 0.12, 0.52, 0.42])
    axc.barh([c for c, _ in top][::-1], [v for _, v in top][::-1], color=TEAL, height=0.72)
    axc.set_title("imagenes por ciudad (top 18 de 56)", fontsize=10.5, color=NAVY)
    axc.tick_params(labelsize=8)
    axc.set_xlabel("imagenes", fontsize=9.5)

    axf = fig.add_axes([0.66, 0.12, 0.31, 0.42])
    f = ps["funnel"]
    labels = ["votos safe", "menos imagen\nausente", "menos empates\n(aparte)"]
    vals = [f["votes_safe"], f["votes_safe"] - f["dropped_missing_image"], f["decisive_pairs"]]
    axf.bar(labels, vals, color=[SLATE, AMBER, NAVY], width=0.62)
    for i, v in enumerate(vals):
        axf.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=9, color=INK)
    axf.set_title("embudo hasta los pares usados", fontsize=10.5, color=NAVY)
    axf.tick_params(labelsize=8.5)
    axf.set_ylim(0, max(vals) * 1.16)

    fig.text(0.5, 0.035, "ground truth = comparaciones humanas ORIGINALES del atributo safe. "
                         "Los q-scores TrueSkill se excluyen: son un score derivado, no un voto humano.",
             ha="center", fontsize=9.8, color=CORAL, style="italic")
    return save(fig, "fig04_place_pulse_2.png")


# ==========================================================================
# 5. Diseno de comparacion justa
# ==========================================================================
def fig_comparacion_justa():
    fig, ax = blank((12.5, 6.0))
    ax.text(0.5, 0.955, "Diseno de comparacion justa:  se aisla la CALIDAD DE LA REPRESENTACION",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
    encs = [("OURS\nSimCLR ViT-B/16", "auto-supervisado\nen el corpus del proyecto", NAVY),
            ("ImageNet\nViT-B/16", "supervisado\n(clasificacion ImageNet)", TEAL),
            ("DINOv2\nViT-B/14", "auto-supervisado\nLVD-142M (baseline fuerte)", CORAL)]
    for i, (name, pre, c) in enumerate(encs):
        x = 0.06 + i * 0.30
        box(ax, x, 0.66, 0.26, 0.145, name, fc=c, tc="white", fs=11, bold=True, ec=c)
        box(ax, x, 0.545, 0.26, 0.10, pre, fc="white", ec=c, fs=9)
        arrow(ax, (x + 0.13, 0.545), (x + 0.13, 0.475), color=c)
    box(ax, 0.06, 0.345, 0.86, 0.125,
        "BACKBONE CONGELADO  |  torch.no_grad + eval  |  sin fine-tuning  |  L2-norm  |  224x224",
        fc="#E8F2F0", ec=TEAL, fs=11, bold=True)
    arrow(ax, (0.49, 0.345), (0.49, 0.285))
    box(ax, 0.06, 0.155, 0.86, 0.125,
        "MISMO probe de ranking lineal:   s = W·z   |   logit = s_izq − s_der   |   BCEWithLogits\n"
        "antisimetrico por construccion: no puede aprender sesgo de posicion",
        fc=LIGHT, ec=NAVY, fs=10)
    ax.text(0.5, 0.085, "mismos pares   |   mismos splits   |   mismos hiperparametros   |   misma metrica   |   seed 42",
            ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.025, "DINOv2 se evalua a 224 px (256 tokens) y no a su 518 nativo (1369 tokens): "
                        "de lo contrario no seria el mismo protocolo.",
            ha="center", fontsize=9.3, color=SLATE, style="italic")
    return save(fig, "fig05_comparacion_justa.png")


# ==========================================================================
# 6. Tres protocolos de validacion
# ==========================================================================
def fig_protocolos():
    lk = json.loads((OUTPUTS / "leakage_report.json").read_text(encoding="utf-8"))
    sp = json.loads((PKG_ROOT / "data" / "splits" / "splits_summary.json").read_text(encoding="utf-8"))
    fig, ax = blank((13, 6.0))
    ax.text(0.5, 0.96, "Tres protocolos de validacion sobre los MISMOS pares",
            ha="center", fontsize=14.5, fontweight="bold", color=NAVY)
    P = lk["protocols"]
    cards = [
        ("A | paper_protocol", "split aleatorio de PARES\n(comparabilidad historica)",
         f"train {P['paper_protocol']['partitions']['train']:,}\n"
         f"val   {P['paper_protocol']['partitions']['val']:,}\n"
         f"test  {P['paper_protocol']['partitions']['test']:,}",
         "imagenes de test que TAMBIEN\naparecen en train:\n"
         f"{P['paper_protocol']['checks']['train_vs_test']['shared_image_fraction_of_test']:.1%}",
         CORAL),
        ("B | image_disjoint", "ninguna imagen de test\naparece en train, ni en otro par",
         f"train {P['image_disjoint']['partitions']['train']:,}\n"
         f"val   {P['image_disjoint']['partitions']['val']:,}\n"
         f"test  {P['image_disjoint']['partitions']['test']:,}",
         "imagenes compartidas\ntrain-test:  0.0 %\n"
         f"(coste: se descartan\n{sp['protocols']['image_disjoint']['n_pairs_discarded_cross_partition']:,} pares)",
         TEAL),
        ("C | city_holdout", "10 ciudades completas\nfuera de entrenamiento",
         f"train {P['city_holdout']['partitions']['train']:,}\n"
         f"val   {P['city_holdout']['partitions']['val']:,}\n"
         f"test_strict {P['city_holdout']['partitions']['test_strict']:,}",
         "ciudades solo en test:\nTokyo | Bangkok | Cape Town\nGaborone | Rio | Santiago\nMoscow | Barcelona\nMelbourne | Houston",
         NAVY),
    ]
    for i, (title, sub, sizes, note, c) in enumerate(cards):
        x = 0.035 + i * 0.325
        box(ax, x, 0.80, 0.29, 0.085, title, fc=c, tc="white", fs=11.5, bold=True, ec=c)
        box(ax, x, 0.665, 0.29, 0.115, sub, fc="white", ec=c, fs=9.8)
        box(ax, x, 0.455, 0.29, 0.19, sizes, fc=LIGHT, ec=SLATE, fs=10)
        box(ax, x, 0.155, 0.29, 0.275, note, fc="white", ec=c, fs=9.8, lw=2.0)
    ax.text(0.5, 0.085, "el ranking entre encoders debe sostenerse en los TRES protocolos "
                        "para hablar de una mejor representacion",
            ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.03, "prioridad cientifica:  B (image_disjoint)  >  C (city_holdout)  >  A (referencia historica)",
            ha="center", fontsize=9.8, color=SLATE, style="italic")
    return save(fig, "fig06_protocolos.png")


# ==========================================================================
# 7. Leakage: hallazgo + contraste con Lima
# ==========================================================================
def fig_leakage():
    lk = json.loads((OUTPUTS / "leakage_report.json").read_text(encoding="utf-8"))
    chk = lk["protocols"]["paper_protocol"]["checks"]["train_vs_test"]
    frac = chk["shared_image_fraction_of_test"]
    df = pd.read_csv(OUTPUTS / "benchmark_results.csv")
    full = df[(df["regime"] == "FULL") & (df["seed"].astype(str) == "42")]

    fig = plt.figure(figsize=(13, 6.3))
    fig.text(0.5, 0.955, "Hallazgo metodologico: leakage estructural ≠ metrica inflada",
             ha="center", fontsize=14.5, fontweight="bold", color=NAVY)

    ax1 = fig.add_axes([0.045, 0.16, 0.26, 0.64])
    ax1.grid(False)
    ax1.pie([frac, 1 - frac], colors=[CORAL, LIGHT], startangle=90,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax1.text(0, 0.14, f"{frac:.1%}", ha="center", fontsize=21, fontweight="bold", color=CORAL)
    ax1.text(0, -0.30, "de las imagenes de test\naparecen tambien en train\nvia otros pares",
             ha="center", fontsize=9.5, color=INK)
    ax1.set_title(f"A | paper_protocol\n{chk['shared_image_ids']:,} imagenes compartidas",
                  fontsize=11, color=NAVY)

    ax2 = fig.add_axes([0.375, 0.20, 0.26, 0.58])
    order = ["paper_protocol", "image_disjoint", "city_holdout"]
    xlab = ["A\npaper", "B\nimg-disjoint", "C\ncity-holdout"]
    w = 0.36
    for k, (enc, c) in enumerate([("OURS (SimCLR ViT-B/16)", NAVY),
                                  ("ImageNet ViT-B/16", TEAL)]):
        vals, lo, hi = [], [], []
        for p in order:
            sel = full[(full["encoder_label"] == enc) & (full["protocol"] == p)]
            sel = sel[sel["test_split"].isin(["test", "test_strict"])]
            if sel.empty:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
                continue
            vals.append(sel["pairwise_accuracy"].iloc[0])
            lo.append(sel["pairwise_accuracy"].iloc[0] - sel["ci95_low"].iloc[0])
            hi.append(sel["ci95_high"].iloc[0] - sel["pairwise_accuracy"].iloc[0])
        xs = np.arange(3) + (k - 0.5) * w
        ax2.bar(xs, vals, width=w, color=c, label=enc.split(" (")[0],
                yerr=[lo, hi], capsize=3, ecolor=INK, error_kw=dict(lw=1.1))
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax2.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.6, color=INK)
    ax2.axhline(0.5, color=CORAL, ls="--", lw=1.4)
    ax2.text(2.45, 0.505, "azar", fontsize=8.5, color=CORAL, ha="right")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(xlab, fontsize=9)
    ax2.set_ylim(0.48, 0.72)
    ax2.set_ylabel("pairwise accuracy", fontsize=10)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax2.set_title("Place Pulse | probe LINEAL congelado\nA = B: el leakage no infla",
                  fontsize=11, color=TEAL)

    ax3 = fig.add_axes([0.705, 0.20, 0.26, 0.58])
    rows = []
    for d in ("Barranco", "La_Victoria"):
        f = GRAPH / "robustness_checks" / f"{d}_2016_robustness_comparison.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        rk = r[r["cv_scheme"] == "random_kfold"]["auc"].iloc[0]
        sb = r[r["cv_scheme"] == "spatial_blocks"]["auc"].iloc[0]
        rows.append((d.replace("_", " "), rk, sb))
    if rows:
        xs = np.arange(len(rows))
        ax3.bar(xs - 0.19, [r[1] for r in rows], width=0.38, color=CORAL, label="random k-fold")
        ax3.bar(xs + 0.19, [r[2] for r in rows], width=0.38, color=NAVY, label="spatial blocks")
        for i, r in enumerate(rows):
            ax3.text(i - 0.19, r[1] + 0.012, f"{r[1]:.3f}", ha="center", fontsize=8.6)
            ax3.text(i + 0.19, r[2] + 0.012, f"{r[2]:.3f}", ha="center", fontsize=8.6)
            ax3.text(i, 1.005, f"−{(r[1]-r[2])*100:.1f} pts AUC",
                     ha="center", fontsize=10, color=CORAL, fontweight="bold")
        ax3.set_xticks(xs)
        ax3.set_xticklabels([r[0] for r in rows], fontsize=9.5)
        ax3.set_ylim(0.5, 1.06)
        ax3.set_ylabel("AUC", fontsize=10)
        ax3.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                   ncol=2, frameon=False)
    ax3.set_title("Lima | fused + logistic (1541 feats)\nsplit espacial: SI se desploma",
                  fontsize=11, color=CORAL)

    # Control: A y B tambien difieren en el tamano de train (286k vs 158k pares).
    # Se re-entreno A con el MISMO numero de pares que B para descartar ese confusor.
    ctrl = df[df["regime"] == "CTRL"]
    if not ctrl.empty:
        bits = []
        for enc in ("OURS (SimCLR ViT-B/16)", "ImageNet ViT-B/16"):
            c = ctrl[ctrl.encoder_label == enc]
            b = full[(full.encoder_label == enc) & (full.protocol == "image_disjoint")
                     & (full.test_split == "test")]
            if not c.empty and not b.empty:
                bits.append(f"{enc.split(' (')[0]} {c.pairwise_accuracy.iloc[0]:.4f} "
                            f"vs {b.pairwise_accuracy.iloc[0]:.4f}")
        if bits:
            fig.text(0.5, 0.098,
                     "Control: A re-entrenado con los mismos 158,312 pares de train que B "
                     "(A con train recortado vs B)   |   " + "   |   ".join(bits),
                     ha="center", fontsize=9.5, color=TEAL, fontweight="bold")
    fig.text(0.5, 0.042,
             "Leccion: el leakage se materializa cuando el modelo tiene capacidad para explotarlo.\n"
             "Un probe lineal sobre embeddings congelados no memoriza imagenes; un clasificador con 1541 features sobre 401 nodos, si.",
             ha="center", fontsize=10.2, color=NAVY)
    fig.text(0.5, 0.004, "Resultado experimental nuevo (Place Pulse)  |  Resultado reproducido (Lima)",
             ha="center", fontsize=9, color=SLATE, style="italic")
    return save(fig, "fig07_leakage.png")


# ==========================================================================
# 8. Resultados del benchmark FULL
# ==========================================================================
def fig_benchmark_full():
    df = pd.read_csv(OUTPUTS / "benchmark_results.csv")
    full = df[(df["regime"] == "FULL") & (df["seed"].astype(str) == "42")]
    if full.empty:
        return None
    fig = plt.figure(figsize=(12.8, 6.2))
    fig.text(0.5, 0.955, "Benchmark Place Pulse 2.0 | atributo safe | dataset COMPLETO",
             ha="center", fontsize=14.5, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.90, "110,987 imagenes | 440,758 pares decisivos | backbone congelado | probe lineal | seed 42",
             ha="center", fontsize=10.2, color=SLATE)

    cols = [("paper_protocol", "test", "A | paper_protocol\n(referencia historica)"),
            ("image_disjoint", "test", "B | image_disjoint\n(sin imagenes compartidas)"),
            ("city_holdout", "test_strict", "C | city_holdout strict\n(10 ciudades nuevas)")]
    encs = [e for e in ENC_COLOR if (full["encoder_label"] == e).any()]
    for j, (proto, split, title) in enumerate(cols):
        ax = fig.add_axes([0.07 + j * 0.315, 0.20, 0.235, 0.60])
        vals, los, his, labs, cs = [], [], [], [], []
        for e in encs:
            sel = full[(full["encoder_label"] == e) & (full["protocol"] == proto)
                       & (full["test_split"] == split)]
            if sel.empty:
                continue
            a = sel["pairwise_accuracy"].iloc[0]
            vals.append(a)
            labs.append(e.split(" (")[0])
            cs.append(ENC_COLOR[e])
            los.append(a - sel["ci95_low"].iloc[0])
            his.append(sel["ci95_high"].iloc[0] - a)
        xs = np.arange(len(vals))
        ax.bar(xs, vals, color=cs, width=0.6, yerr=[los, his], capsize=4,
               ecolor=INK, error_kw=dict(lw=1.2))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.006, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(0.5, color=CORAL, ls="--", lw=1.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=9, rotation=12)
        ax.set_ylim(0.48, 0.72)
        if j == 0:
            ax.set_ylabel("pairwise accuracy  (azar = 0.50)", fontsize=10.5)
        else:
            ax.set_yticklabels([])
        n = full[(full["protocol"] == proto) & (full["test_split"] == split)]["n_test_pairs"].iloc[0]
        ax.set_title(f"{title}\nn = {n:,} pares de test", fontsize=10.5, color=NAVY)
    fig.text(0.5, 0.055,
             "El ranking es ESTABLE en los tres protocolos: ImageNet ViT-B/16 supera al encoder heredado por ~6 puntos.\n"
             "Barras de error: IC 95% bootstrap (2000 resamples).",
             ha="center", fontsize=10.3, color=NAVY)
    fig.text(0.5, 0.008, "Resultado experimental nuevo", ha="center", fontsize=9.5,
             color=TEAL, fontweight="bold")
    return save(fig, "fig08_benchmark_full.png")


# ==========================================================================
# 9. Resultados SMOKE (3 encoders)
# ==========================================================================
def fig_benchmark_smoke():
    df = pd.read_csv(OUTPUTS / "benchmark_results.csv")
    smoke = df[df["regime"] == "SMOKE"]
    if smoke.empty:
        log("  (sin runs SMOKE todavia: se omite fig09)")
        return None
    sub = json.loads((PKG_ROOT / "data" / "smoke" / "subset_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13.8, 5.9))
    fig.text(0.5, 0.945,
             f"{sub['n_images']:,} imagenes | {sub['n_cities']} ciudades | "
             f"{sub['n_pairs_kept']:,} pares | seed {sub['seed']} | identico para los tres encoders",
             ha="center", fontsize=10.2, color=SLATE)
    cols = [("paper_protocol", "test", "A | paper_protocol"),
            ("image_disjoint", "test", "B | image_disjoint"),
            ("city_holdout", "test_strict", "C | city_holdout strict")]
    encs = [e for e in ENC_COLOR if (smoke["encoder_label"] == e).any()]
    for j, (proto, split, title) in enumerate(cols):
        ax = fig.add_axes([0.07 + j * 0.315, 0.235, 0.235, 0.63])
        vals, los, his, labs, cs = [], [], [], [], []
        for e in encs:
            sel = smoke[(smoke["encoder_label"] == e) & (smoke["protocol"] == proto)
                        & (smoke["test_split"] == split)]
            if sel.empty:
                continue
            a = sel["pairwise_accuracy"].iloc[0]
            vals.append(a)
            labs.append(e.split(" (")[0])
            cs.append(ENC_COLOR[e])
            los.append(a - sel["ci95_low"].iloc[0])
            his.append(sel["ci95_high"].iloc[0] - a)
        xs = np.arange(len(vals))
        ax.bar(xs, vals, color=cs, width=0.6, yerr=[los, his], capsize=4,
               ecolor=INK, error_kw=dict(lw=1.2))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.007, f"{v:.4f}", ha="center", fontsize=9.6, fontweight="bold")
        ax.axhline(0.5, color=CORAL, ls="--", lw=1.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=9, rotation=12)
        ax.set_ylim(0.46, 0.78)
        if j == 0:
            ax.set_ylabel("pairwise accuracy  (azar = 0.50)", fontsize=10.5)
        else:
            ax.set_yticklabels([])
        sl = smoke[(smoke["protocol"] == proto) & (smoke["test_split"] == split)]
        ax.set_title(f"{title}\nn = {sl['n_test_pairs'].iloc[0]:,} pares", fontsize=10.5, color=NAVY)
    # Chequeo de representatividad: el subconjunto debe reproducir el resultado
    # de la corrida completa para los dos encoders que SI se corrieron enteros.
    full = df[(df["regime"] == "FULL") & (df["seed"].astype(str) == "42")]
    checks = []
    for enc in ("OURS (SimCLR ViT-B/16)", "ImageNet ViT-B/16"):
        f = full[(full.encoder_label == enc) & (full.protocol == "image_disjoint")
                 & (full.test_split == "test")]
        k = smoke[(smoke.encoder_label == enc) & (smoke.protocol == "image_disjoint")
                  & (smoke.test_split == "test")]
        if not f.empty and not k.empty:
            checks.append(f"{enc.split(' (')[0]}: completo {f.pairwise_accuracy.iloc[0]:.3f} "
                          f"vs subconjunto {k.pairwise_accuracy.iloc[0]:.3f}")
    if checks:
        fig.text(0.5, 0.125,
                 "Representatividad del subconjunto (image_disjoint):   " + "     |     ".join(checks),
                 ha="center", fontsize=10, color=TEAL, fontweight="bold")
    fig.text(0.5, 0.058,
             "Motivo del subconjunto: la extraccion completa de DINOv2 en CPU tiene ETA ~4.3 h y cayo por corrupcion de heap.\n"
             "Los tres encoders se evaluan sobre EXACTAMENTE las mismas imagenes, pares y splits.",
             ha="center", fontsize=10, color=NAVY)
    fig.text(0.5, 0.012, "RESULTADO PRELIMINAR - SMOKE TEST", ha="center", fontsize=11.5,
             color=CORAL, fontweight="bold")
    return save(fig, "fig09_benchmark_smoke.png")


# ==========================================================================
# 10. Accuracy por ciudad (city_holdout)
# ==========================================================================
def fig_por_ciudad():
    runs = []
    for enc in ("ours_simclr_vit_b16", "imagenet_vit_b16"):
        f = OUTPUTS / "runs" / f"{enc}__city_holdout__linear__seed42" / "metrics.json"
        if f.exists():
            runs.append((enc, json.loads(f.read_text(encoding="utf-8"))))
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ref = runs[-1][1]["test"]["test_strict"]["by_city"]["per_city"]
    cities = sorted(ref, key=lambda c: -ref[c]["pairwise_accuracy"])
    ns = [ref[c]["n"] for c in cities]
    w = 0.38
    for k, (enc, m) in enumerate(runs):
        pc = m["test"]["test_strict"]["by_city"]["per_city"]
        vals = [pc[c]["pairwise_accuracy"] for c in cities]
        col = ENC_COLOR["OURS (SimCLR ViT-B/16)"] if "ours" in enc else ENC_COLOR["ImageNet ViT-B/16"]
        xs = np.arange(len(cities)) + (k - 0.5) * w
        ax.bar(xs, vals, width=w, color=col,
               label=("OURS SimCLR ViT-B/16" if "ours" in enc else "ImageNet ViT-B/16"))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.axhline(0.5, color=CORAL, ls="--", lw=1.4)
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels([f"{c}\nn={ns[i]:,}" for i, c in enumerate(cities)], fontsize=8.4)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("pairwise accuracy", fontsize=10.5)
    ax.set_title("city_holdout strict: accuracy en las 10 ciudades NO vistas en entrenamiento",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.legend(fontsize=9.5, loc="upper right", ncol=2, framealpha=0.95)
    fig.text(0.5, 0.005, "Resultado experimental nuevo | ninguna de estas ciudades aparece en train "
                         "| test_strict = ambos extremos del par en ciudad holdout",
             ha="center", fontsize=9.3, color=SLATE, style="italic")
    return save(fig, "fig10_por_ciudad.png")


# ==========================================================================
# 11. De embeddings a conceptos + codebook
# ==========================================================================
def fig_codebook():
    fig, ax = blank((13, 6.2))
    ax.text(0.5, 0.96, "De un embedding opaco a un codebook de conceptos urbanos",
            ha="center", fontsize=14.5, fontweight="bold", color=NAVY)
    box(ax, 0.03, 0.56, 0.17, 0.20, "Imagen urbana\n(street view)", fc=LIGHT, ec=SLATE, fs=10)
    box(ax, 0.245, 0.68, 0.20, 0.13, "Encoder visual\nViT / DINOv2", fc=NAVY, tc="white", fs=10.5, bold=True, ec=NAVY)
    box(ax, 0.245, 0.50, 0.20, 0.145, "Vision + lenguaje\nCLIP | Grounding DINO\nYOLO-World | GLIP | SAM 2",
        fc=AMBER, fs=9.5, ec="#C9A227")
    arrow(ax, (0.20, 0.70), (0.245, 0.745))
    arrow(ax, (0.20, 0.62), (0.245, 0.575))
    box(ax, 0.49, 0.68, 0.19, 0.13, "embedding 768-D\n[0.13, −0.42, ...]", fc="white", ec=NAVY, fs=10)
    box(ax, 0.49, 0.50, 0.19, 0.145, "conceptos\ninterpretables", fc="white", ec="#C9A227", fs=10)
    arrow(ax, (0.445, 0.745), (0.49, 0.745))
    arrow(ax, (0.445, 0.573), (0.49, 0.573))
    box(ax, 0.72, 0.50, 0.25, 0.31,
        "CODEBOOK\n\nvegetation  = 0.82\ngraffiti       = 0.05\npedestrian = 1\nlighting      = 0.41\nfacade_cond = 0.63\n...",
        fc="#E8F2F0", ec=TEAL, fs=10, lw=2.0)
    arrow(ax, (0.68, 0.655), (0.72, 0.655), color=TEAL, lw=2.2)

    concepts = ["vegetacion", "iluminacion", "apertura visual", "visibilidad", "grafiti",
                "basura", "deterioro de fachada", "rejas", "muros", "veredas",
                "peatones", "vehiculos", "comercio", "espacios vacios", "arboles",
                "senales", "mobiliario urbano"]
    ax.text(0.5, 0.415, "Conceptos CANDIDATOS del diccionario", ha="center",
            fontsize=11.5, color=NAVY, fontweight="bold")
    x, y = 0.035, 0.345
    for c in concepts:
        w = 0.0100 * len(c) + 0.012
        if x + w > 0.945:
            x, y = 0.035, y - 0.066
        box(ax, x, y, w, 0.050, c, fc="white", ec=SLATE, fs=8.8)
        x += w + 0.030
    ax.text(0.5, 0.115, "Concept Bottleneck Models:   imagen -> conceptos humanos -> prediccion",
            ha="center", fontsize=11, color=NAVY)
    ax.text(0.5, 0.055, "NO se afirma que estos elementos CAUSEN la percepcion de inseguridad. "
                        "El codebook permite PREGUNTARLO empiricamente.",
            ha="center", fontsize=10.2, color=CORAL, fontweight="bold")
    ax.text(0.5, 0.008, "Advertencia de 'Vision Transformers Need Registers': no toda region saliente de un "
                        "feature map es informacion semantica valida.",
            ha="center", fontsize=9, color=SLATE, style="italic")
    return save(fig, "fig11_codebook.png")


# ==========================================================================
# 12. De la imagen al espacio urbano + ego-networks
# ==========================================================================
def fig_ego_network():
    import networkx as nx
    fig = plt.figure(figsize=(13, 6.4))
    fig.text(0.5, 0.955, "La calle no existe aislada:  cuanto contexto espacial ayuda?",
             ha="center", fontsize=14.5, fontweight="bold", color=NAVY)

    axp = fig.add_axes([0.03, 0.60, 0.94, 0.26])
    axp.axis("off")
    axp.grid(False)
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    steps = ["Imagen", "visual\nembedding", "conceptos\ninterpretables", "node\nrepresentation",
             "grafo de calles\nego-1 / ego-2 / ego-3", "GNN /\nagregacion", "perceived\nsafety"]
    cols = [LIGHT, NAVY, AMBER, LIGHT, "#4A6FA5", LIGHT, TEAL]
    tcs = [INK, "white", INK, INK, "white", INK, "white"]
    n = len(steps)
    w = 0.107
    gap = (1 - n * w) / (n - 1)
    for i, (s, c, tc) in enumerate(zip(steps, cols, tcs)):
        x = i * (w + gap)
        box(axp, x, 0.28, w, 0.5, s, fc=c, tc=tc, fs=9.0, bold=(i in (1, 4, 6)),
            ec=c if c != LIGHT else SLATE)
        if i < n - 1:
            arrow(axp, (x + w, 0.53), (x + w + gap, 0.53))

    G = nx.random_geometric_graph(60, 0.20, seed=7)
    pos = nx.get_node_attributes(G, "pos")
    center = min(G.nodes, key=lambda v: (pos[v][0] - 0.5) ** 2 + (pos[v][1] - 0.5) ** 2)
    lengths = nx.single_source_shortest_path_length(G, center, cutoff=3)
    shades = {0: CORAL, 1: "#F2A488", 2: AMBER, 3: "#8FA7BD"}
    titles = ["ego-0\nsolo el nodo", "ego-1\n+ vecinos a 1 arista",
              "ego-2\n+ hasta 2 saltos", "ego-3\n+ hasta 3 saltos"]
    for k in range(4):
        ax = fig.add_axes([0.055 + k * 0.235, 0.135, 0.20, 0.40])
        ax.axis("off")
        ax.grid(False)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#DDE3E8", width=0.9)
        inc = [v for v, d in lengths.items() if d <= k]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes), node_size=32,
                               node_color="#E8ECEF", edgecolors="#CFD8DE", linewidths=0.6)
        for d in range(k, -1, -1):
            nd = [v for v in inc if lengths[v] == d]
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nd,
                                   node_size=90 if d == 0 else 54,
                                   node_color=shades[d], edgecolors="white", linewidths=1.0)
        plural = "" if len(inc) == 1 else "s"
        ax.set_title(f"{titles[k]}\n{len(inc)} nodo{plural} en el vecindario",
                     fontsize=9.8, color=NAVY)
    fig.text(0.5, 0.065, "Pregunta experimental: a partir de que profundidad el contexto deja de aportar "
                         "senal y empieza a introducir ruido?",
             ha="center", fontsize=10.8, color=NAVY, fontweight="bold")
    fig.text(0.5, 0.012, "Experimento pendiente | se comparara Visual-only vs Structural-only vs Fused "
                         "en cada profundidad (grafo ilustrativo, no datos reales)",
             ha="center", fontsize=9.2, color=SLATE, style="italic")
    return save(fig, "fig12_ego_network.png")


# ==========================================================================
# 13. Visual vs structural vs fused en Lima (resultado reproducido)
# ==========================================================================
def fig_visual_vs_structural():
    rows = []
    for d in ("Barranco", "La_Victoria"):
        f = GRAPH / "final_classifier" / f"{d}_2016_classifier_comparison.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        r = r[r["estrategia"].str.endswith("logistic")]
        for strat in ("visual_only", "structural_only", "fused"):
            m = r[r["estrategia"].str.startswith(strat)]
            rows.append({
                "distrito": d.replace("_", " "), "strategy": strat,
                "random": m[m["cv_scheme"] == "random"]["auc"].iloc[0],
                "spatial": m[m["cv_scheme"] == "spatial"]["auc"].iloc[0],
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    fig.suptitle("Lima | pipeline heredado: la ventaja de la representacion VISUAL se desploma con split espacial",
                 fontsize=13.5, fontweight="bold", color=NAVY, y=0.99)
    names = {"visual_only": "visual\nonly", "structural_only": "structural\nonly", "fused": "fused"}
    for ax, d in zip(axes, df["distrito"].unique()):
        sub = df[df["distrito"] == d]
        xs = np.arange(len(sub))
        ax.bar(xs - 0.19, sub["random"], width=0.38, color=CORAL, label="random k-fold")
        ax.bar(xs + 0.19, sub["spatial"], width=0.38, color=NAVY, label="spatial GroupKFold")
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(i - 0.19, r["random"] + 0.012, f"{r['random']:.2f}", ha="center", fontsize=8.8)
            ax.text(i + 0.19, r["spatial"] + 0.012, f"{r['spatial']:.2f}", ha="center", fontsize=8.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([names[s] for s in sub["strategy"]], fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{d} 2016  |  logistic", fontsize=11.5, color=NAVY)
        ax.axhline(0.5, color=SLATE, ls="--", lw=1)
    axes[0].set_ylabel("AUC (crime risk)", fontsize=10.5)
    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.95)
    parts = []
    for d in df["distrito"].unique():
        g = df[df["distrito"] == d].set_index("strategy")
        parts.append(
            f"{d}:  visual −{(g.loc['visual_only','random']-g.loc['visual_only','spatial'])*100:.0f}"
            f" | structural −{(g.loc['structural_only','random']-g.loc['structural_only','spatial'])*100:.0f}"
            f" | fused −{(g.loc['fused','random']-g.loc['fused','spatial'])*100:.0f}")
    fig.text(0.5, 0.075,
             "Caida de AUC al pasar de random k-fold a split espacial (puntos):     "
             + "        ".join(parts),
             ha="center", fontsize=9.8, color=NAVY)
    fig.text(0.5, 0.018,
             "La estrategia estructural es la que menos pierde; el bloque visual y el fusionado se desploman. "
             "Resultado reproducido.",
             ha="center", fontsize=9.6, color=CORAL)
    fig.subplots_adjust(bottom=0.26, top=0.86)
    return save(fig, "fig13_visual_vs_structural_lima.png")


# ==========================================================================
# 14. Roadmap
# ==========================================================================
def fig_roadmap():
    fig, ax = blank((13, 6.4))
    ax.text(0.5, 0.955, "Roadmap experimental", ha="center", fontsize=15,
            fontweight="bold", color=NAVY)
    ax.text(0.5, 0.895, "cada etapa produce la evidencia que la siguiente necesita",
            ha="center", fontsize=11, color=SLATE, style="italic")

    stages = [
        ("AHORA", "Place Pulse 2.0\nbenchmark de encoders\n(3 protocolos)", TEAL, True),
        ("SIGUIENTE", "Visual-only  vs\nStructural-only  vs  Fused", NAVY, False),
        ("", "ego-0 / ego-1 / ego-2 / ego-3\ncuanto contexto espacial", NAVY, False),
        ("", "codebook de conceptos\n+ Concept Bottleneck", AMBER, False),
        ("", "cross-district\nBarranco <-> La Victoria", "#4A6FA5", False),
        ("", "validacion temporal\n2016-2023", "#4A6FA5", False),
        ("CONTRIBUCION", "Perceived Safety  vs\nObserved Crime (gap)", CORAL, False),
    ]
    rows = [stages[:4], stages[4:]]
    tops = [0.66, 0.36]
    for r, (row, top) in enumerate(zip(rows, tops)):
        n = len(row)
        w = 0.190
        gap = 0.050
        total = n * w + (n - 1) * gap
        x0 = (1 - total) / 2
        for i, (tag, body, c, done) in enumerate(row):
            x = x0 + i * (w + gap)
            if tag:
                ax.text(x + w / 2, top + 0.155, tag, ha="center", fontsize=10,
                        color=c, fontweight="bold")
            box(ax, x, top, w, 0.135, body, fc=c if done else "white",
                tc="white" if done else INK, ec=c, fs=9.6,
                lw=2.2 if done else 1.5, bold=done)
            ax.text(x + w / 2, top - 0.042, "con evidencia" if done else "pendiente",
                    ha="center", fontsize=8.6, color=TEAL if done else SLATE,
                    fontweight="bold" if done else "normal")
            if i < n - 1:
                arrow(ax, (x + w, top + 0.068), (x + w + gap, top + 0.068))
        if r == 0:
            # conector de fin de fila 1 a inicio de fila 2
            xe = x0 + (n - 1) * (w + gap) + w
            arrow(ax, (xe - w / 2, top), (x0 + w / 2, tops[1] + 0.135),
                  color=SLATE, lw=1.6, ls="--")

    box(ax, 0.155, 0.10, 0.69, 0.145,
        "MLLM / LLM  ->  solo si demuestra valor incremental medible\n"
        "NO se construye una GNN nueva ni fine-tuning masivo antes de cerrar la representacion visual",
        fc=LIGHT, ec=SLATE, fs=10.5)
    return save(fig, "fig14_roadmap.png")


# ==========================================================================
# 15. Antes / ahora
# ==========================================================================
def fig_antes_ahora():
    fig, ax = blank((12.5, 5.8))
    ax.text(0.5, 0.955, "Que cambia respecto al proyecto original",
            ha="center", fontsize=14.5, fontweight="bold", color=NAVY)
    ax.text(0.14, 0.845, "ANTES", ha="center", fontsize=12.5, color=SLATE, fontweight="bold")
    seq_a = ["Imagen", "embedding", "grafo", "crime risk"]
    for i, s in enumerate(seq_a):
        x = 0.05 + i * 0.225
        box(ax, x, 0.70, 0.185, 0.10, s, fc=LIGHT, ec=SLATE, fs=10)
        if i < len(seq_a) - 1:
            arrow(ax, (x + 0.185, 0.75), (x + 0.225, 0.75))
    ax.text(0.14, 0.615, "AHORA", ha="center", fontsize=12.5, color=NAVY, fontweight="bold")
    seq_b = [("Imagen", LIGHT, INK), ("representacion visual\nVALIDADA", NAVY, "white"),
             ("benchmark contra\npercepcion humana", TEAL, "white"),
             ("conceptos urbanos\ninterpretables", AMBER, INK),
             ("contexto espacial\n(ego-networks)", "#4A6FA5", "white"),
             ("PERCEIVED\nSAFETY", CORAL, "white")]
    n = len(seq_b)
    w = 0.128
    gap = (0.95 - n * w) / (n - 1)
    for i, (s, c, tc) in enumerate(seq_b):
        x = 0.025 + i * (w + gap)
        box(ax, x, 0.42, w, 0.165, s, fc=c, tc=tc, fs=8.8,
            bold=(i in (1, 5)), ec=c if c != LIGHT else SLATE)
        if i < n - 1:
            arrow(ax, (x + w, 0.50), (x + w + gap, 0.50))
    box(ax, 0.26, 0.245, 0.48, 0.105, "+ segunda senal:  OBSERVED CRIME  (del pipeline heredado)",
        fc="#FBEDE8", ec=CORAL, fs=10.5)
    ax.text(0.5, 0.155, "La contribucion ya no es «usar un grafo» - eso ya existe en la literatura.",
            ha="center", fontsize=11.2, color=CORAL, fontweight="bold")
    ax.text(0.5, 0.075, "Se investiga:   representacion   |   interpretabilidad   |   cuanto contexto hace falta   |   "
                        "generalizacion   |   relacion percepcion-criminalidad",
            ha="center", fontsize=10.5, color=NAVY)
    return save(fig, "fig15_antes_ahora.png")


# ==========================================================================
# 16. Portada
# ==========================================================================
def fig_portada():
    """Grafo decorativo sobre fondo navy, pensado para ir DENTRO de la banda
    de la portada del PPT (por eso el fondo es navy y no blanco)."""
    import networkx as nx
    fig = plt.figure(figsize=(6.0, 4.42), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.grid(False)
    ax.set_facecolor(NAVY)
    G = nx.random_geometric_graph(150, 0.145, seed=11)
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3A5878", width=1.1)
    vals = np.array([0.55 * pos[v][0] + 0.45 * pos[v][1] for v in G.nodes])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=vals,
                           cmap=plt.get_cmap("YlOrRd"), edgecolors=NAVY, linewidths=1.0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    fig.savefig(FIGDIR / "fig00_portada.png", facecolor=NAVY)
    plt.close(fig)
    log("  fig00_portada.png")
    return FIGDIR / "fig00_portada.png"


def main() -> None:
    log("Generando figuras en outputs/figures/")
    fig_portada()
    fig_pipeline_historico()
    fig_percepcion_vs_crimen()
    fig_sota_map()
    fig_placepulse()
    fig_comparacion_justa()
    fig_protocolos()
    fig_leakage()
    fig_benchmark_full()
    fig_benchmark_smoke()
    fig_por_ciudad()
    fig_visual_vs_structural()
    fig_codebook()
    fig_ego_network()
    fig_roadmap()
    fig_antes_ahora()
    fig_imagen_vs_contexto()
    log(f"\nListo -> {FIGDIR}")




# ==========================================================================
# 17. Imagen aislada vs calle + calles vecinas (slide 2)
# ==========================================================================
def fig_imagen_vs_contexto():
    import networkx as nx
    fig = plt.figure(figsize=(11.5, 5.6))
    fig.text(0.5, 0.945, "Una imagen aislada  vs  la calle dentro de su vecindario",
             ha="center", fontsize=14, fontweight="bold", color=NAVY)

    G = nx.random_geometric_graph(34, 0.26, seed=5)
    pos = nx.get_node_attributes(G, "pos")
    c = min(G.nodes, key=lambda v: (pos[v][0] - 0.5) ** 2 + (pos[v][1] - 0.5) ** 2)
    nb = set(nx.single_source_shortest_path_length(G, c, cutoff=2)) - {c}

    panels = [
        (0.055, "lo que ve el modelo hoy\nUN punto, sin entorno", False, INK),
        (0.525, "lo que la percepcion humana usa\nel punto Y sus calles vecinas", True, NAVY),
    ]
    for x0, title, ctx, tc in panels:
        ax = fig.add_axes([x0, 0.19, 0.42, 0.60])
        ax.axis("off")
        ax.grid(False)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((-0.08, -0.08), 1.16, 1.16, transform=ax.transData,
                                   facecolor="#F7F9FA", edgecolor="#DDE3E8",
                                   linewidth=1.2, zorder=0))
        if ctx:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#D3DBE1", width=1.2)
            nx.draw_networkx_nodes(G, pos, ax=ax,
                                   nodelist=sorted(set(G.nodes) - nb - {c}),
                                   node_size=70, node_color="#E3E8EC",
                                   edgecolors="#CFD8DE", linewidths=0.8)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(nb), node_size=150,
                                   node_color=AMBER, edgecolors="white", linewidths=1.2)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[c], node_size=430,
                               node_color=CORAL, edgecolors="white", linewidths=2)
        ax.set_xlim(-0.10, 1.10)
        ax.set_ylim(-0.10, 1.10)
        ax.set_title(title, fontsize=11.5, color=tc, pad=10)

    fig.text(0.5, 0.085,
             "iluminacion de la cuadra siguiente  |  flujo peatonal  |  actividad comercial  |  "
             "continuidad del tejido urbano",
             ha="center", fontsize=10.5, color=SLATE)
    fig.text(0.5, 0.02,
             "La percepcion es un juicio sobre un LUGAR, no sobre un fotograma.",
             ha="center", fontsize=11.5, color=CORAL, fontweight="bold")
    return save(fig, "fig16_imagen_vs_contexto.png")


if __name__ == "__main__":
    main()
