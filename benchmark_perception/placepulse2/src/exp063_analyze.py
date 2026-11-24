"""
EXP-063/064 — Análisis: Δ pareado de cada cabeza vs la lineal-BT (mismo encoder, rep y protocolo, mismos pares de test),
IC 95 % por bootstrap de imágenes (pigeonhole, B = 2000) y de pares, McNemar; tabla + figura tipo forest.

Uso:  python src/exp063_analyze.py [--protocols image_disjoint,city_holdout,paper_protocol]
Salida: outputs/exp063/summary_heads.csv, summary_heads.md, fig_heads_<protocolo>.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import mcnemar_p, pair_boot_ci, pigeonhole_ci  # noqa: E402
from utils import OUTPUTS, REPO_ROOT, log  # noqa: E402

OUT = OUTPUTS / "exp063"
FIG = REPO_ROOT / "deliverables" / "figures"
NAMES = {"linear_bt": "Linear BT (baseline)", "mlp_bt": "MLP BT", "ridge": "Ridge → λ", "knn": "KNN (cosine) → λ",
         "svr_rbf": "SVR-RBF → λ", "rf": "Random Forest → λ", "xgb": "XGBoost → λ", "mlp_reg": "MLP → λ",
         "logreg_clf": "Logistic (quartiles)", "linsvm_clf": "Linear SVM (quartiles)",
         "svm_rbf_clf": "RBF-SVM (quartiles)", "knn_clf": "KNN (quartiles)", "rf_clf": "Random Forest (quartiles)",
         "xgb_clf": "XGBoost (quartiles)", "mlp_clf": "MLP (quartiles)"}
TRACK_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}     # validado (dataviz validate_palette, light)
TRACK_MARK = {"A": "o", "B": "s", "C": "^"}
TRACK_LABEL = {"A": "A · trained on pairwise votes", "B": "B · regress per-image λ (BT, train votes)",
               "C": "C · classify top vs bottom quartile of λ"}


def analyze(protocol: str) -> pd.DataFrame:
    od = OUT / protocol
    recs = [json.loads(l) for l in (od / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if "error" not in r]
    rows = []
    for (enc, rep), grp in pd.DataFrame(recs).groupby(["encoder", "rep"]):
        tp = np.load(od / f"test_pairs_{enc}_{rep}.npz")
        li, ri = tp["li"], tp["ri"]
        base_f = od / "scores" / f"{enc}__{rep}__linear_bt__s42__correct.npy"
        cb = np.load(base_f).astype(float) if base_f.exists() else None
        for r in grp.to_dict("records"):
            c = np.load(od / "scores" / f"{enc}__{rep}__{r['head']}__s{r['seed']}__correct.npy").astype(float)
            row = {k: r.get(k) for k in ("protocol", "encoder", "rep", "head", "track", "seed", "val_acc", "test_acc",
                                         "test_ci_img_lo", "test_ci_img_hi", "n_test_pairs", "fit_seconds_total",
                                         "test_logloss", "bin_auroc", "bin_bal_acc", "bin_f1", "bin_brier",
                                         "bin_pos_rate", "bin_n_test_images")}
            row["params"] = json.dumps(r["params"])
            if cb is not None and not (r["head"] == "linear_bt" and r["seed"] == 42):
                d, lo, hi = pigeonhole_ci(c, cb, li, ri, B=2000)
                plo, phi = pair_boot_ci(c - cb)
                row.update({"delta_vs_linear": d, "delta_img_lo": lo, "delta_img_hi": hi, "delta_pair_lo": plo,
                            "delta_pair_hi": phi, "mcnemar_p": mcnemar_p(c, cb)})
            rows.append(row)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, protocol: str, rep: str = "z", enc: str = "dinov2_vit_b14") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = df[(df.protocol == protocol) & (df.rep == rep) & (df.encoder == enc) & (df.seed == 42)].copy()
    if d.empty:
        return
    order = {"A": 0, "B": 1, "C": 2}
    d["o"] = d.track.map(order)
    d = d.sort_values(["o", "test_acc"], ascending=[False, True]).reset_index(drop=True)
    base = d[d["head"] == "linear_bt"]["test_acc"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 2.6))
    for i, r in d.iterrows():
        col, mk = TRACK_COLOR[r.track], TRACK_MARK[r.track]
        ax.plot([r.test_ci_img_lo, r.test_ci_img_hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.test_acc, i, mk, color=col, ms=9, mec="white", mew=2)
        ax.text(r.test_ci_img_hi + 0.001, i, f"{r.test_acc:.3f}", va="center", fontsize=9, color="#3b3b38")
    ax.axvline(base, color="#8a8a85", lw=1, ls="--")
    ax.text(base, -0.75, " linear BT", fontsize=8.5, color="#5e5e59", va="center")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([NAMES.get(h, h) for h in d["head"]], fontsize=10, color="#1b1b19")
    ax.set_xlabel("Pairwise accuracy on the same test pairs (95% CI: image-level bootstrap)", fontsize=10)
    n = int(d.n_test_pairs.iloc[0])
    ax.set_title(f"EXP-063 · frozen DINOv2 ({'PCA-128' if rep != 'z' else '768-d'}) · {protocol} · "
                 f"{n:,} test pairs · seed 42", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    lo = min(d.test_ci_img_lo.min(), base) - 0.004
    ax.set_xlim(lo, d.test_ci_img_hi.max() + 0.008)
    handles = [Line2D([], [], color=TRACK_COLOR[t], marker=TRACK_MARK[t], lw=2, ms=8, label=TRACK_LABEL[t])
               for t in ("A", "B", "C")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=1, fontsize=8.5, frameon=False)
    ax.set_ylim(-1.2, len(d) - 0.5)
    fig.text(0.01, 0.005, "Chance = 0.50 (off-axis). Each head outputs a per-image score g(z); decision = sign(g_i − g_j).\n"
             "Hyper-parameters chosen on validation pairs only. Source: outputs/exp063/summary_heads.csv",
             fontsize=7.5, color="#5e5e59")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"exp063_heads_{protocol}_{rep}.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="image_disjoint,city_holdout,paper_protocol")
    args = ap.parse_args()
    dfs = [analyze(p) for p in args.protocols.split(",") if (OUT / p / "results.jsonl").exists()]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT / "summary_heads.csv", index=False)
    with open(OUT / "summary_heads.md", "w", encoding="utf-8") as f:
        f.write("| Protocolo | Encoder | Rep | Track | Cabeza | seed | val | test [IC img] | Δ vs lineal (pts) [IC img] | McNemar p | AUROC bin | s |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            dd = "—" if pd.isna(getattr(r, "delta_vs_linear", np.nan)) else \
                f"{100 * r.delta_vs_linear:+.2f} [{100 * r.delta_img_lo:+.2f}, {100 * r.delta_img_hi:+.2f}]"
            mp = "—" if pd.isna(getattr(r, "mcnemar_p", np.nan)) else f"{r.mcnemar_p:.1e}"
            au = "—" if pd.isna(r.bin_auroc) else f"{r.bin_auroc:.3f}"
            f.write(f"| {r.protocol} | {r.encoder} | {r.rep} | {r.track} | {r.head} | {r.seed} | {r.val_acc:.4f} | "
                    f"{r.test_acc:.4f} [{r.test_ci_img_lo:.4f}, {r.test_ci_img_hi:.4f}] | {dd} | {mp} | {au} | "
                    f"{r.fit_seconds_total:.0f} |\n")
    for p in df.protocol.unique():
        for rep in df[df.protocol == p].rep.unique():
            figure(df, p, rep)
    log(df[["protocol", "rep", "head", "seed", "val_acc", "test_acc", "delta_vs_linear", "delta_img_lo",
            "delta_img_hi"]].to_string())


if __name__ == "__main__":
    main()
