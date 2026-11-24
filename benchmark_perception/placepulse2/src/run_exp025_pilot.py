"""
EXP-025 (piloto) — lecturas R2 y R3. Requiere haber corrido build_street_graph_pilot.py.

R2 (lectura principal): homogeneidad perceptual del vecindario.
    Puntaje humano por imagen = tasa de victoria en sus comparaciones del split image_disjoint de
    TRAIN (mín. MIN_VOTES). Para cada imagen de train con puntaje: ρ de Spearman entre su puntaje y
    la media de los puntajes de sus vecinas con puntaje. Se compara grafo (C) vs euclídeo con el
    mismo nº de vecinas (B2) sobre el MISMO conjunto de imágenes; IC 95 % bootstrap (1,000, seed 42).
    No usa ningún par de val/test.

R3 (lectura secundaria, poca potencia declarada): train_probe.py sin cambios sobre el split
    image_disjoint filtrado a las 3 ciudades, para ego0 / euc250 / euc_match / graph_k*. Δ pareado
    contra ego0 con bootstrap y McNemar.

Uso:
    python run_exp025_pilot.py            # R2 + R3
    python run_exp025_pilot.py --skip-r3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from utils import MANIFESTS, OUTPUTS, RUNS, SPLITS, log, set_seed

SEED = 42
OUT = OUTPUTS / "street_graph_pilot"
MIN_VOTES = 3
N_BOOT = 1000


def train_scores() -> pd.Series:
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    split = json.loads((SPLITS / "image_disjoint.json").read_text())
    tr = pairs.iloc[split["train"]]
    left_win = tr["label"].astype(int).to_numpy() == 1
    winners = np.where(left_win, tr["left"], tr["right"])
    n = pd.concat([tr["left"], tr["right"]]).value_counts()
    w = pd.Series(winners).value_counts().reindex(n.index, fill_value=0)
    s = (w / n)[n >= MIN_VOTES]
    log(f"puntajes de train: {len(s):,} imágenes con >= {MIN_VOTES} votos")
    return s


def neigh_mean(lists, score: pd.Series) -> np.ndarray:
    out = np.full(len(lists), np.nan)
    for i, L in enumerate(lists):
        v = [score[x] for x in L if x in score.index]
        if v:
            out[i] = float(np.mean(v))
    return out


def boot_delta(a, b_self, c_ctx, d_ctx, rng):
    """ρ(a, c_ctx) − ρ(a, d_ctx) con bootstrap por imágenes."""
    n = len(a)
    deltas = np.empty(N_BOOT)
    for t in range(N_BOOT):
        ix = rng.integers(0, n, n)
        deltas[t] = spearmanr(a[ix], c_ctx[ix])[0] - spearmanr(a[ix], d_ctx[ix])[0]
    return np.percentile(deltas, [2.5, 97.5])


def r2(k_star: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    score = train_scores()
    nb = pd.read_json(OUT / "neighbors.jsonl", lines=True)
    nb = nb[(nb["split"] == "train") & nb["location_id"].isin(score.index)].reset_index(drop=True)
    nb["s"] = nb["location_id"].map(score)
    for col in ("graph", "euc_match", "euc250"):
        nb[f"m_{col}"] = neigh_mean(nb[col], score)

    rows = []
    for city, g in list(nb.groupby("city")) + [("ALL (3 ciudades)", nb)]:
        for ref in ("euc_match", "euc250"):
            m = g[["s", "m_graph", f"m_{ref}"]].dropna()
            a, c, d = m["s"].to_numpy(), m["m_graph"].to_numpy(), m[f"m_{ref}"].to_numpy()
            rho_c, rho_d = spearmanr(a, c)[0], spearmanr(a, d)[0]
            lo, hi = boot_delta(a, None, c, d, rng)
            rows.append(dict(city=city, comparison=f"graph_k{k_star} - {ref}", n_images=len(m),
                             rho_graph=round(rho_c, 4), rho_ref=round(rho_d, 4),
                             delta=round(rho_c - rho_d, 4), ci95_low=round(lo, 4),
                             ci95_high=round(hi, 4), ci_excludes_0=bool(lo > 0 or hi < 0)))
            log(f"  R2 {city:<18} graph vs {ref:<9} n={len(m):5d}  ρ_graph={rho_c:.3f}  "
                f"ρ_ref={rho_d:.3f}  Δ={rho_c - rho_d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "r2_homogeneity.csv", index=False)
    return res


def r3(k_star: int) -> pd.DataFrame:
    variants = ["ego0", "euc250", "euc_match", f"graph_k{k_star}"]
    here = Path(__file__).resolve().parent
    for v in variants:
        run_id = f"exp025__{v}__image_disjoint__seed42"
        if (RUNS / run_id / "metrics.json").exists():
            log(f"  R3 {v}: ya existe, no se re-entrena")
            continue
        subprocess.run([sys.executable, str(here / "train_probe.py"), "--encoder", v,
                        "--protocol", "image_disjoint", "--seed", "42", "--run-id", run_id,
                        "--features-root", str(OUT / "features"),
                        "--splits-root", str(OUT / "splits")], check=True)
    preds = {v: pd.read_csv(RUNS / f"exp025__{v}__image_disjoint__seed42" / "predictions.csv")
             for v in variants}
    base = preds["ego0"]
    rng = np.random.default_rng(SEED)
    rows = []
    for v in variants:
        p = preds[v]
        assert (p["left"].to_numpy() == base["left"].to_numpy()).all()
        c0 = (base["pred"] == base["label"]).to_numpy()
        c1 = (p["pred"] == p["label"]).to_numpy()
        n = len(c1)
        boots = np.array([c1[ix].mean() - c0[ix].mean()
                          for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
        b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
        p_mc = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        rows.append(dict(variant=v, n_test=n, accuracy=round(c1.mean(), 4),
                         delta_vs_ego0=round(c1.mean() - c0.mean(), 4),
                         ci95_low=round(float(np.percentile(boots, 2.5)), 4),
                         ci95_high=round(float(np.percentile(boots, 97.5)), 4),
                         mcnemar_p=round(float(p_mc), 4)))
        log(f"  R3 {v:<12} acc={c1.mean():.4f}  Δ={c1.mean() - c0.mean():+.4f}  p={p_mc:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "r3_probe.csv", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-r3", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    k_star = json.loads((OUT / "build_summary.json").read_text())["k_star"]
    log(f"k* = {k_star}")
    out = {"k_star": k_star, "min_votes": MIN_VOTES, "n_boot": N_BOOT, "seed": SEED}
    out["r2"] = r2(k_star).to_dict("records")
    if not args.skip_r3:
        out["r3"] = r3(k_star).to_dict("records")
    (OUT / "exp025_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    log(f"-> {OUT / 'exp025_results.json'}")


if __name__ == "__main__":
    main()
