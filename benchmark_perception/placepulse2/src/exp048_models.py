"""
EXP-048 (E4) — etapa `models`: contexto Mapillary sobre el grafo de segmentos de Santiago vs la imagen sola.

Mismo entrenador, pares (image_disjoint) y evaluación que EXP-042 (funciones importadas de allí); solo cambia de dónde
sale el contexto: imágenes Mapillary sin etiqueta (exp048_mapillary_context.py --stage select/download/embed).
Diseño en docs/EXPERIMENT_REGISTRY.md (EXP-048). Salidas: outputs/exp048/.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from exp026_additive_context import load_images
from exp042_pp_segment_graph_models import (SNAP_MAX_M, GCNScore, LinearScore, fit_beta, image_scores, metrics,
                                            pair_arrays, paired, predict, train_model)
from exp048_mapillary_context import CITY, OUT, city_graph
from probe_utils import load_pairs, load_split
from segment_graph import normalized_adjacency, reach_matrices, snap_to_segments
from utils import environment_info, log, set_seed

SEED = 42
RADII = [50.0, 100.0, 200.0]
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def project(lonlat: np.ndarray, crs) -> np.ndarray:
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy(lonlat[:, 0], lonlat[:, 1]), crs=4326).to_crs(crs)
    return np.c_[pts.x, pts.y]


def ctx_features(Z, ctx, cnt) -> np.ndarray:
    return np.hstack([Z, ctx, np.log1p(cnt)[:, None]]).astype(np.float32)


def crf_segments(f_sum: np.ndarray, cnt: np.ndarray, A: sp.csr_matrix, lam: float, eps: float = 1e-3) -> np.ndarray:
    """y = argmin Σ n_s (y_s − f̄_s)² + λ Σ_(s,t) (y_s − y_t)² + ε‖y‖²  →  (N + λL + εI) y = Σ f."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    M = (sp.diags(cnt + eps) + lam * (sp.diags(deg) - A)).tocsc()
    return spla.splu(M).solve(f_sum)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    man, Z = load_images("dinov2_vit_b14")
    Z = Z.astype(np.float32)
    n = len(man)
    log(f"norma media de z (PP): {np.linalg.norm(Z, axis=1).mean():.3f}")
    row_of = {lid: i for i, lid in enumerate(man["location_id"])}
    pairs, split = load_pairs(), load_split("image_disjoint")
    tr, va, te = (pair_arrays(pairs, split[k], row_of) for k in ("train", "val", "test"))
    city = (man["city"] == CITY).to_numpy()
    touch = lambda d: city[d[0]] | city[d[1]]                                   # noqa: E731

    # ── grafo y snap de las imágenes PP de Santiago ──
    Gp, seg, A = city_graph()
    n_seg = A.shape[0]
    idx = np.flatnonzero(city)
    xy_pp = np.full((n, 2), np.nan)
    xy_pp[idx] = project(man.loc[idx, ["lon", "lat"]].to_numpy(), Gp.graph["crs"])
    node_xy = {k: (d["x"], d["y"]) for k, d in Gp.nodes(data=True)}
    r = snap_to_segments(seg, A, xy_pp[idx], node_xy, max_dist=SNAP_MAX_M)
    seg_of = np.full(n, -1)
    seg_of[idx] = r.seg_idx

    # ── Mapillary: selección con embedding ──
    S = pd.read_csv(OUT / "mapillary_selection.csv", dtype={"id": str})
    E = np.load(OUT / "mapillary_dinov2.npz")
    pos = {str(k): i for i, k in enumerate(E["ids"])}
    S = S[S["id"].isin(pos)].reset_index(drop=True)
    Zm = E["cls"][[pos[k] for k in S["id"]]].astype(np.float64)
    seg_m = S["seg_idx"].to_numpy()
    xy_m = project(S[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
    cnt_s = np.bincount(seg_m, minlength=n_seg).astype(float)
    sum_s = np.zeros((n_seg, Zm.shape[1]))
    np.add.at(sum_s, seg_m, Zm)
    log(f"Mapillary con embedding: {len(S):,} en {int((cnt_s > 0).sum()):,} segmentos")

    # ── contextos ──
    img = np.flatnonzero(seg_of >= 0)
    src = np.unique(seg_of[img])
    rpos = {s: k for k, s in enumerate(src)}
    R = reach_matrices(A, src, 2)
    feats, cov, cnt_ego1 = {"E0": Z}, [], None
    for K in (0, 1, 2):
        tot, c = R[K] @ sum_s, R[K] @ cnt_s
        ctx, cn = np.zeros_like(Z), np.zeros(n)
        rows = np.array([rpos[s] for s in seg_of[img]])
        ok = c[rows] > 0
        ctx[img[ok]] = (tot[rows][ok] / c[rows][ok, None]).astype(np.float32)
        cn[img] = c[rows]
        feats[f"M-ego{K}"] = ctx_features(Z, ctx, cn)
        cov.append(dict(context=f"ego{K}", median_images=float(np.median(cn[img])),
                        pct_with_context=100 * float((cn[img] > 0).mean())))
        if K == 1:
            cnt_ego1 = cn
    ctx, cn = np.zeros_like(Z), np.zeros(n)
    for i in img:
        k = int(min(cnt_ego1[i], len(Zm)))
        if k:
            ctx[i] = Zm[rng.choice(len(Zm), size=k, replace=False)].mean(0)
            cn[i] = k
    feats["M-rand"] = ctx_features(Z, ctx, cn)
    tree = cKDTree(xy_m)
    radius_feats = {}
    for rad in RADII:
        ctx, cn = np.zeros_like(Z), np.zeros(n)
        for i, nb in zip(idx, tree.query_ball_point(xy_pp[idx], rad)):
            if nb:
                ctx[i] = Zm[nb].mean(0)
                cn[i] = len(nb)
        radius_feats[rad] = ctx_features(Z, ctx, cn)
        cov.append(dict(context=f"radius{int(rad)}", median_images=float(np.median(cn[idx])),
                        pct_with_context=100 * float((cn[idx] > 0).mean())))
    pd.DataFrame(cov).to_csv(OUT / "context_coverage.csv", index=False, float_format="%.2f")
    log(pd.DataFrame(cov).to_string(index=False))

    logits, info = {}, {}

    def run_linear(name, X):
        tt = time.perf_counter()
        mdl, inf = train_model(LinearScore(X), tr, va)
        logits[name] = {"val": predict(mdl, va), "test": predict(mdl, te)}
        info[name] = {**inf, "params": X.shape[1], "seconds": round(time.perf_counter() - tt)}
        log(f"  {name}: {info[name]}")
        return mdl

    e0 = run_linear("E0", feats["E0"])
    for name in ("M-ego0", "M-ego1", "M-ego2", "M-rand"):
        run_linear(name, feats[name])
    best_r, best_v = None, -1.0
    for rad in RADII:
        run_linear(f"M-radius{int(rad)}", radius_feats[rad])
        v = metrics(logits[f"M-radius{int(rad)}"]["val"][touch(va)], va[2][touch(va)])["acc"]
        if v > best_v:
            best_r, best_v = rad, v
    logits["M-radius"] = logits[f"M-radius{int(best_r)}"]
    info["M-radius"] = {"selected_radius_m_on_val": best_r}

    # ── G1: GCN sobre los segmentos a ≤ 2 saltos de las imágenes PP ──
    keep = np.unique(R[2].indices)
    loc = -np.ones(n_seg, int)
    loc[keep] = np.arange(len(keep))
    seg_local = np.where(seg_of >= 0, loc[np.clip(seg_of, 0, None)], -1)
    ck = cnt_s[keep]
    Xseg = np.hstack([sum_s[keep] / np.maximum(ck, 1)[:, None], (ck > 0)[:, None], np.log1p(ck)[:, None]])
    Ahat = normalized_adjacency(A)[keep][:, keep]
    for name, Am in (("G1-gcn", Ahat), ("G1-noedge", sp.identity(len(keep), format="csr"))):
        tt = time.perf_counter()
        mdl, inf = train_model(GCNScore(Z, seg_local, Xseg, Am), tr, va)
        logits[name] = {"val": predict(mdl, va), "test": predict(mdl, te)}
        info[name] = {**inf, "seconds": round(time.perf_counter() - tt)}
        log(f"  {name}: {info[name]}")

    # ── G2: CRF sobre las puntuaciones E0 de las imágenes Mapillary ──
    w0 = e0.w.weight.detach().numpy().ravel().astype(np.float64)
    base = image_scores(e0, n).astype(np.float64)
    train_imgs = np.unique(np.r_[tr[0], tr[1]])
    fm = Zm @ w0 - base[train_imgs].mean()
    f_sum = np.bincount(seg_m, weights=fm, minlength=n_seg)
    crf_rows, best = [], None
    for lam in LAMBDAS:
        y = crf_segments(f_sum, cnt_s, A, lam)
        ctx = np.where(seg_of >= 0, y[np.clip(seg_of, 0, None)], 0.0)
        beta = fit_beta(base, ctx, tr)
        zv = (base[va[0]] + beta * ctx[va[0]]) - (base[va[1]] + beta * ctx[va[1]])
        mv = metrics(zv[touch(va)], va[2][touch(va)])
        crf_rows.append(dict(lam=lam, beta=beta, val_touch_acc=mv["acc"], val_touch_logloss=mv["logloss"]))
        log(f"  CRF λ={lam}: β={beta:.3f} {mv}")
        if lam == 0.0:
            logits["G2-crf-lam0"] = {"test": (base[te[0]] + beta * ctx[te[0]]) - (base[te[1]] + beta * ctx[te[1]])}
        if best is None or mv["logloss"] < best[0]:
            best = (mv["logloss"], lam, beta, ctx)
    pd.DataFrame(crf_rows).to_csv(OUT / "crf_lambda_grid.csv", index=False, float_format="%.5f")
    _, lam, beta, ctx = best
    logits["G2-crf"] = {"test": (base[te[0]] + beta * ctx[te[0]]) - (base[te[1]] + beta * ctx[te[1]])}
    info["G2-crf"] = {"selected_lambda_on_val": lam, "beta_on_train": beta}

    # ── evaluación en los pares de test que tocan Santiago ──
    T = touch(te)
    grp = np.where(city[te[0]], te[0], te[1])[T]
    acc = pd.DataFrame([dict(model=k, **metrics(d["test"][T], te[2][T])) for k, d in logits.items()])
    acc.to_csv(OUT / "accuracy.csv", index=False, float_format="%.5f")
    graph_models = ["M-ego0", "M-ego1", "M-ego2", "M-radius", "G1-gcn", "G1-noedge", "G2-crf"]
    comps = [(m, "E0") for m in graph_models + ["M-rand", "G2-crf-lam0"]] + \
            [(m, "M-rand") for m in ("M-ego1", "M-ego2", "M-radius", "G1-gcn", "G2-crf")] + \
            [("M-ego1", "M-ego0"), ("M-ego2", "M-ego0"), ("G1-gcn", "G1-noedge"), ("G2-crf", "G2-crf-lam0")]
    prow = []
    for other, b in comps:
        r_ = paired(logits[b]["test"][T], logits[other]["test"][T], te[2][T], grp, rng)
        prow.append(dict(comparison=f"{other} - {b}", **r_))
        log(f"  {other:>12} - {b:<11} Δacc={r_['delta_acc']:+.4f} agrupado [{r_['ci_low_grouped']:+.4f},"
            f"{r_['ci_high_grouped']:+.4f}] p={r_['mcnemar_p']:.2g}")
    par = pd.DataFrame(prow).set_index("comparison")
    par.to_csv(OUT / "paired_vs_baselines.csv", float_format="%.5f")
    helps = [m for m in ("M-ego1", "M-ego2", "M-radius", "G1-gcn", "G2-crf")
             if par.loc[f"{m} - E0", "ci_low_grouped"] > 0 and par.loc[f"{m} - M-rand", "ci_low_grouped"] > 0]
    summ = {"city": CITY, "n_test_touch": int(T.sum()), "santiago_images_in_test_pairs": int(len(np.unique(grp))),
            "mapillary_images": len(S), "train_info": info,
            "criteria": {"graph_models_that_help": helps,
                         "connectivity_helps": bool(par.loc["M-ego1 - M-ego0", "ci_low_grouped"] > 0
                                                    or par.loc["G1-gcn - G1-noedge", "ci_low_grouped"] > 0)},
            "seconds": round(time.perf_counter() - t0), "environment": environment_info()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    log(json.dumps(summ["criteria"]))


if __name__ == "__main__":
    main()
