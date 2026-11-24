"""Figuras en INGLÉS (vectoriales) de EXP-068/069/070 para el EC1, desde los CSV/JSON de outputs (no recalcula nada).
Uso: python src/make_figs_v3_en.py  → deliverables/ec1_final/src/images/exp06{8,9}_*.pdf, exp070_*.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK2, NEUTRAL, SERIES, setup  # noqa: E402
from utils import OUTPUTS, REPO_ROOT  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DST = REPO_ROOT / "deliverables" / "ec1_final" / "src" / "images"
NAMES = {"image_disjoint": "New images", "city_holdout": "New cities"}
EN = {"wealthy": "wealthier", "beautiful": "more beautiful", "lively": "livelier", "boring": "more boring",
      "depressing": "more depressing"}


def exp068():
    tm = pd.read_csv(OUTPUTS / "exp068" / "tie_margin.csv")
    rc = pd.read_csv(OUTPUTS / "exp068" / "risk_coverage.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    w = 0.38
    for k, (p, s) in enumerate(tm.groupby("protocol", sort=False)):
        ax[0].bar(s["margin_quintile"] + (k - 0.5) * w, s["tie_rate"] * 100, width=w, color=SERIES[k], label=NAMES[p])
    ax[0].set_xlabel(r"Quintile of $|s_i-s_j|$ (1 = smallest margin)")
    ax[0].set_ylabel("Pairs voted 'equal' (%)")
    ax[0].set_title("(a) People tie more where the model hesitates", loc="left")
    ax[0].legend()
    for k, (p, s) in enumerate(rc.groupby("protocol", sort=False)):
        ax[1].plot(s["coverage"] * 100, s["accuracy"] * 100, color=SERIES[k], marker="o", ms=3, label=NAMES[p])
    ax[1].axhline(50, color=NEUTRAL, lw=1, ls=":")
    ax[1].set_xlabel("Coverage (% of pairs answered)")
    ax[1].set_ylabel("Accuracy on answered pairs (%)")
    ax[1].set_title("(b) Risk--coverage", loc="left")
    fig.tight_layout()
    fig.savefig(DST / "exp068_ties_abstention.pdf")
    plt.close(fig)


def exp069():
    r = json.loads((OUTPUTS / "exp069" / "summary.json").read_text(encoding="utf-8"))
    tab = pd.DataFrame(r["cross_attribute"])
    res = r["residual"]["score_space_residual"]
    cb = r["combination"]
    labels = [f"'{EN[a]}' head" + (" (sign $-$)" if s < 0 else "") for a, s in zip(tab["attr"], tab["sign"])]
    labels += ["combination of the five", "safe score minus the five", "'safer' head (reference)"]
    vals = list(tab["acc_on_safe_test"]) + [cb["acc"], res["acc"], r["safe_acc_test"]]
    los = list(tab["lo"]) + [np.nan, res["ci"][0], np.nan]
    his = list(tab["hi"]) + [np.nan, res["ci"][1], np.nan]
    cols = [SERIES[0]] * len(tab) + [SERIES[3], SERIES[1], SERIES[2]]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    yy = np.arange(len(vals))
    v = np.array(vals) * 100
    ax.barh(yy, v - 50, left=50, color=cols, height=0.6)
    lo, hi = np.array(los) * 100, np.array(his) * 100
    m = ~np.isnan(lo)
    ax.errorbar(v[m], yy[m], xerr=[v[m] - lo[m], hi[m] - v[m]], fmt="none", ecolor=INK2, capsize=2)
    ax.set_yticks(yy, labels)
    ax.set_xlim(48, 70)
    ax.axvline(50, color=NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("Pairwise accuracy on 'safer' test votes, new images (%)")
    fig.tight_layout()
    fig.savefig(DST / "exp069_construct.pdf")
    plt.close(fig)


def exp070():
    S = pd.read_csv(OUTPUTS / "exp070" / "learning_curve_summary.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    lab = {"aleatorio": "random pairs", "activo": "active pairs"}
    for k, (st, s) in enumerate(S.groupby("strategy", sort=False)):
        ax.errorbar(s["n"] * (1 + 0.04 * (k - 0.5)), s["delta_mean"] * 100, yerr=s["delta_sd"] * 100, marker="o",
                    ms=4, capsize=2, color=SERIES[k], label=lab[st])
    ax.axhline(0, color=NEUTRAL, lw=1, ls=":")
    ax.set_xscale("log")
    ns = sorted(S["n"].unique())
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Local votes used to adapt the head")
    ax.set_ylabel("Gain over no adaptation (pts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DST / "exp070_lima_budget.pdf")
    plt.close(fig)


def exp069_es():
    """Versión en español para la PPT (reemplaza la figura del script del experimento, que cortaba el eje)."""
    r = json.loads((OUTPUTS / "exp069" / "summary.json").read_text(encoding="utf-8"))
    tab = pd.DataFrame(r["cross_attribute"])
    res, cb = r["residual"]["score_space_residual"], r["combination"]
    labels = [f"cabeza «{x}»" + (" (signo −)" if s < 0 else "") for x, s in zip(tab["attr_es"], tab["sign"])]
    labels += ["combinación de las 5", "«seguro» sin lo común", "cabeza «seguro»"]
    vals = np.array(list(tab["acc_on_safe_test"]) + [cb["acc"], res["acc"], r["safe_acc_test"]]) * 100
    lo = np.array(list(tab["lo"]) + [np.nan, res["ci"][0], np.nan]) * 100
    hi = np.array(list(tab["hi"]) + [np.nan, res["ci"][1], np.nan]) * 100
    cols = [SERIES[0]] * len(tab) + [SERIES[3], SERIES[1], SERIES[2]]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    yy = np.arange(len(vals))
    ax.barh(yy, vals - 50, left=50, color=cols, height=0.6)
    m = ~np.isnan(lo)
    ax.errorbar(vals[m], yy[m], xerr=[vals[m] - lo[m], hi[m] - vals[m]], fmt="none", ecolor=INK2, capsize=2)
    ax.set_yticks(yy, labels)
    ax.set_xlim(48, 70)
    ax.axvline(50, color=NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("Acierto en votos de «más seguro», fotos nuevas (%)")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "deliverables" / "figures" / "v3" / "exp069_construct_validity_es.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    setup()
    exp068()
    exp069()
    exp070()
    exp069_es()
    print("ok", DST)
