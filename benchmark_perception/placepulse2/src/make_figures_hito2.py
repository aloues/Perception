"""Figuras del hito 2 (fases E/F/G/H). Solo lee CSV ya generados; no recalcula nada."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import OUTPUTS

EGO, FIG = OUTPUTS / "ego", OUTPUTS / "figures"
C = {"visual_only": "#2e86ab", "structural_only": "#8d99ae", "fused": "#e4572e"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def fig_depth():
    r = pd.read_csv(EGO / "ego_results.csv")
    r = r[r.cv_scheme == "spatial"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, d in zip(axs, ["Barranco", "La Victoria"]):
        s = r[r.district == d]
        n, prev = s.n_nodes.iloc[0], s.prevalence.iloc[0]
        for strat in ("visual_only", "fused"):
            t = s[s.strategy == strat].sort_values("depth")
            x = np.arange(len(t))
            ax.errorbar(x, t.auc, yerr=[t.auc - t.ci95_low, t.ci95_high - t.auc], marker="o",
                        capsize=4, color=C[strat], label=strat.replace("_", "-"), lw=2)
        st = s[s.strategy == "structural_only"].iloc[0]
        ax.axhline(st.auc, color=C["structural_only"], ls="--", label=f"structural-only ({st.auc:.2f})")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(4)); ax.set_xticklabels(["ego-0", "ego-1", "ego-2", "ego-3"])
        ax.set_title(f"{d}  (n={n}, prevalencia {prev:.0%})", loc="left", fontweight="bold")
        ax.set_ylabel("AUC (GroupKFold espacial)")
    axs[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Contexto espacial: el máximo está en ego-1; ego-3 cae a ~azar en Barranco", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig_h2_ego_depth.png", dpi=150); plt.close(fig)


def fig_coverage():
    cov = json.loads((EGO / "ego_build_summary.json").read_text(encoding="utf-8"))["districts"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    w = 0.38
    for i, (slug, col) in enumerate((("Barranco", "#2e86ab"), ("La_Victoria", "#e4572e"))):
        d = cov[slug]; v = [d["per_depth"][f"ego{k}"]["n_nodes"] / d["n_graph_nodes"] for k in range(4)]
        ax.bar(np.arange(4) + (i - .5) * w, v, w, color=col, label=slug.replace("_", " "))
        for k, x in enumerate(v):
            ax.text(k + (i - .5) * w, x + .01, f"{x:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"ego-{k}" for k in range(4)])
    ax.set_ylabel("% de nodos con cobertura"); ax.set_ylim(0, 1.1); ax.legend(frameon=False)
    ax.set_title("Cobertura de imágenes por profundidad", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig_h2_ego_coverage.png", dpi=150); plt.close(fig)


def fig_cross():
    d = pd.read_csv(EGO / "cross_district_results.csv")
    d = d[(d.strategy == "fused") & d.depth.str.startswith("ego")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (tr, te) in zip(axs, (("Barranco", "La Victoria"), ("La Victoria", "Barranco"))):
        s = d[(d.train == tr) & (d.test == te)].sort_values("depth")
        x = np.arange(len(s))
        ax.errorbar(x - .05, s.auc_transfer, yerr=[s.auc_transfer - s.ci95_low, s.ci95_high - s.auc_transfer],
                    marker="o", capsize=4, color="#e4572e", lw=2, label="transferencia")
        ax.plot(x + .05, s.auc_intra_district_spatialCV, marker="s", color="#2e86ab", lw=2,
                label=f"intra-distrito ({te}, CV espacial)")
        ax.axhline(.5, color="k", lw=.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(s.depth)
        ax.set_title(f"{tr} → {te}", loc="left", fontweight="bold"); ax.legend(frameon=False, fontsize=9)
    axs[0].set_ylabel("AUC (fused)")
    fig.tight_layout(); fig.savefig(FIG / "fig_h2_cross_district.png", dpi=150); plt.close(fig)


def fig_temporal():
    t = pd.read_csv(EGO / "temporal_labels_results.csv")
    t = t[t.year > 2016]
    lab = {"fused_ego1": ("imagen+estructura 2016 (ego-1)", "#e4572e"),
           "visual_ego1": ("imagen 2016 (ego-1)", "#2e86ab"),
           "persistencia_riesgo2016": ("persistencia: crimen 2016", "#8d99ae")}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, d in zip(axs, ["Barranco", "La Victoria"]):
        for k, (name, col) in lab.items():
            s = t[(t.district == d) & (t.predictor == k)]
            ax.plot(s.year, s.auc, marker="o", color=col, label=name, lw=2)
            ax.fill_between(s.year, s.ci95_low, s.ci95_high, color=col, alpha=.15)
        ax.axhline(.5, color="k", lw=.8, ls=":"); ax.set_title(d, loc="left", fontweight="bold")
        ax.axvspan(2019.6, 2020.4, color="gray", alpha=.12)
    axs[0].set_ylabel("AUC vs riesgo del año (top-q por rango)"); axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_h2_temporal_labels.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_depth, fig_coverage, fig_cross, fig_temporal):
        f()
    print("ok")
