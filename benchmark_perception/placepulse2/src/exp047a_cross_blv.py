"""
EXP-047a — Cruz del compañero en Barranco / La Victoria históricos, sin etiquetas (D7). Piloto técnico.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29, con la enmienda A1 de EXP-047a).

Por spot (V2_4S = media de las vistas 0/90/180/270, SimCLR, igual que el preprocesamiento del compañero):
  cruz-red     en cada brazo N/E/S/O (rumbo centro→vecino), el spot más cercano por distancia de RED ≤ 150 m
  cruz-sector  ídem con distancia en línea recta (puede cruzar manzanas)
  aleatorio    tantos spots del distrito como brazos tenga la cruz-red
  ego-1        todos los spots de los segmentos a ≤ 1 salto del segmento del centro
Vecinos a < 5 m excluidos (mismo panorama). Medida: 1 − cos(centro, media L2 del vecindario).

Uso:  python exp047a_cross_blv.py [--plima] [--encoder simclr|dinov2]
      Salidas: outputs/exp047a/ (B/LV histórico) · outputs/exp047b[_dinov2]/ (S6: 4 distritos P-LIMA, EXP-047b)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from exp041_lima_segment_graph import CORNER_M, DISTRICTS, SNAP_MAX_M, lima_spots, load_graph
from segment_graph import (ARMS, cross_neighbors, dual_adjacency, euclidean_point_distances, network_point_distances,
                           segments_table, snap_to_segments)
from utils import OUTPUTS, environment_info, log, set_seed
from visual_repr import l2n

SEED = 42
OUT = OUTPUTS / "exp047a"
MAX_M, MIN_M = 150.0, 5.0
STRATUM_M = 25.0
N_BOOT = 2000
H4 = (0, 90, 180, 270)


def v2_4s(S: pd.DataFrame, V: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    W = V[V["heading"].astype(int).isin(H4)]
    Z = np.zeros((len(S), E.shape[1]), np.float64)
    np.add.at(Z, W["sidx"].to_numpy(), E[W["row"].to_numpy()])
    return l2n(Z / 4.0)


def dino_v2_4s(S: pd.DataFrame) -> np.ndarray:
    """V2_4S con DINOv2 CLS (media L2 de las vistas 0/90/180/270 de cada spot)."""
    from exp052_connectivity_effect_size import DINO, PLIMA_EXPORTS
    Z = np.full((len(S), 768), np.nan)
    for d, folder in DINO.items():
        z = np.load(PLIMA_EXPORTS / folder / "dinov2.npz")["cls"]
        paths = json.loads((PLIMA_EXPORTS / folder / "dinov2_index.json").read_text(encoding="utf-8"))["paths"]
        P = pd.DataFrame([p.split("/")[1:] for p in paths], columns=["spot_id", "file"])
        P = P[P["file"].isin([f"heading_{h}.jpg" for h in H4])]
        mean = {k: z[g.index.to_numpy()].mean(0) for k, g in P.groupby("spot_id") if len(g) == 4}
        m = (S["district"] == d).to_numpy()
        for i in np.flatnonzero(m):
            if S.at[i, "spot_id"] in mean:
                Z[i] = mean[S.at[i, "spot_id"]]
    assert np.isfinite(Z).all(), "spots sin las 4 vistas DINOv2"
    return l2n(Z)


def dissim(Z: np.ndarray, i: int, nb) -> float:
    nb = [j for j in np.atleast_1d(nb) if j >= 0]
    if not nb:
        return np.nan
    return float(1.0 - Z[i] @ l2n(Z[nb].mean(0)))


def mean_dist(xy, i, nb) -> float:
    nb = [j for j in np.atleast_1d(nb) if j >= 0]
    return float(np.mean(np.hypot(*(xy[nb] - xy[i]).T))) if nb else np.nan


def stratified_delta(d_a, d_b, r_a, r_b) -> float:
    """Δ = d_a − d_b a igual distancia media del vecindario: media de Δ por estrato de 25 m, ponderada por el
    mínimo de observaciones de ambos tipos en el estrato."""
    ok_a, ok_b = np.isfinite(d_a), np.isfinite(d_b)
    sa, sb = np.floor(r_a / STRATUM_M), np.floor(r_b / STRATUM_M)
    num = den = 0.0
    for s in np.unique(np.r_[sa[ok_a], sb[ok_b]]):
        ma, mb = ok_a & (sa == s), ok_b & (sb == s)
        w = min(ma.sum(), mb.sum())
        if w:
            num += w * (d_a[ma].mean() - d_b[mb].mean())
            den += w
    return num / den if den else np.nan


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plima", action="store_true", help="S6: 4 distritos P-LIMA (un spot por panorama)")
    ap.add_argument("--encoder", choices=["simclr", "dinov2"], default="simclr")
    args = ap.parse_args()
    if args.plima:
        OUT = OUTPUTS / ("exp047b" if args.encoder == "simclr" else "exp047b_dinov2")
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    import geopandas as gpd
    S, V, E, _ = lima_spots(args.plima)
    Z = v2_4s(S, V, np.asarray(E)) if args.encoder == "simclr" else dino_v2_4s(S)
    rows, cov_rows, per_spot = [], [], []
    for name in (tuple(DISTRICTS) if args.plima else ("barranco", "la_victoria")):
        Gp, poly = load_graph(name)
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        m = np.flatnonzero((S["district"] == name).to_numpy())
        pts = gpd.GeoSeries(gpd.points_from_xy(S.loc[m, "lon"], S.loc[m, "lat"]), crs=4326).to_crs(Gp.graph["crs"])
        xy = np.c_[pts.x, pts.y]
        node_xy = {n: (d["x"], d["y"]) for n, d in Gp.nodes(data=True)}
        sn = snap_to_segments(seg, A, xy, node_xy, max_dist=SNAP_MAX_M, corner_m=CORNER_M)
        Zd = Z[m]
        Dn = network_point_distances(Gp, seg, sn.seg_idx, xy, MAX_M)
        De = euclidean_point_distances(xy, MAX_M)
        Cn, Ce = cross_neighbors(xy, Dn, MIN_M), cross_neighbors(xy, De, MIN_M)
        centers = np.flatnonzero(sn.seg_idx >= 0)
        # ego-1 de segmento
        A1 = (A + A.T).tocsr()
        by_seg: dict = {}
        for j, s in enumerate(sn.seg_idx):
            if s >= 0:
                by_seg.setdefault(s, []).append(j)
        n = len(m)
        res = {k: np.full(n, np.nan) for k in ("red", "sector", "random", "ego1")}
        rad = {k: np.full(n, np.nan) for k in res}
        size = {k: np.zeros(n, int) for k in res}
        jac = np.full(n, np.nan)
        for i in centers:
            s = sn.seg_idx[i]
            segs = [s] + A1.indices[A1.indptr[s]:A1.indptr[s + 1]].tolist()
            ego = [j for t in segs for j in by_seg.get(t, []) if j != i and np.hypot(*(xy[j] - xy[i])) >= MIN_M]
            k = int((Cn[i] >= 0).sum())
            pool = np.setdiff1d(np.arange(n), [i])
            rnd = rng.choice(pool, size=k, replace=False) if k else np.array([], int)
            for key, nb in (("red", Cn[i]), ("sector", Ce[i]), ("random", rnd), ("ego1", ego)):
                res[key][i], rad[key][i] = dissim(Zd, i, nb), mean_dist(xy, i, nb)
                size[key][i] = len([j for j in np.atleast_1d(nb) if j >= 0])
            a, b = set(Cn[i][Cn[i] >= 0]), set(Ce[i][Ce[i] >= 0])
            if a | b:
                jac[i] = len(a & b) / len(a | b)
        cn_cov = (Cn[centers] >= 0).sum(1)
        ce_cov = (Ce[centers] >= 0).sum(1)
        cov_rows.append(dict(district=name, spots=n, centers_snapped=int(len(centers)),
                             **{f"red_arms_{a}": int((cn_cov == a).sum()) for a in range(5)},
                             red_arms_mean=float(cn_cov.mean()), sector_arms_mean=float(ce_cov.mean()),
                             pct_red_4_arms=100 * float((cn_cov == 4).mean()),
                             pct_sector_4_arms=100 * float((ce_cov == 4).mean()),
                             **{f"red_arm_{ARMS[a]}_pct": 100 * float((Cn[centers, a] >= 0).mean()) for a in range(4)},
                             ego1_size_median=float(np.median(size["ego1"][centers])),
                             jaccard_red_sector_mean=float(np.nanmean(jac[centers])),
                             pct_red_equals_sector=100 * float(np.nanmean(jac[centers] == 1.0))))
        log(f"{name}: {cov_rows[-1]}")
        # comparaciones (Δ > 0 = la cruz-red es más homogénea)
        C = centers
        for other in ("sector", "random", "ego1"):
            both = C[np.isfinite(res["red"][C]) & np.isfinite(res[other][C])]
            d = res[other][both] - res["red"][both]
            boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            strat = stratified_delta(res[other][both], res["red"][both], rad[other][both], rad["red"][both])
            sboot = []
            for _ in range(N_BOOT):
                bb = both[rng.integers(0, len(both), len(both))]
                sboot.append(stratified_delta(res[other][bb], res["red"][bb], rad[other][bb], rad["red"][bb]))
            rows.append(dict(district=name, comparison=f"{other} - red", n_centers=int(len(both)),
                             d_red=float(res["red"][both].mean()), d_other=float(res[other][both].mean()),
                             radius_red_m=float(np.nanmean(rad["red"][both])),
                             radius_other_m=float(np.nanmean(rad[other][both])),
                             delta=float(d.mean()), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                             delta_equal_distance=strat, eq_ci_low=np.nanpercentile(sboot, 2.5),
                             eq_ci_high=np.nanpercentile(sboot, 97.5)))
            log(f"  {rows[-1]}")
        per_spot.append(pd.DataFrame({"district": name, "spot_id": S.loc[m, "spot_id"].to_numpy(),
                                      "seg_idx": sn.seg_idx, "red_arms": (Cn >= 0).sum(1),
                                      "sector_arms": (Ce >= 0).sum(1),
                                      **{f"d_{k}": v for k, v in res.items()},
                                      **{f"radius_{k}_m": v for k, v in rad.items()}}))
    pd.DataFrame(cov_rows).to_csv(OUT / "arm_coverage.csv", index=False, float_format="%.3f")
    pd.DataFrame(rows).to_csv(OUT / "homogeneity_cross.csv", index=False, float_format="%.5f")
    pd.concat(per_spot).to_csv(OUT / "per_spot.csv", index=False, float_format="%.4f")
    (OUT / "run_info.json").write_text(json.dumps({"max_m": MAX_M, "min_m": MIN_M, "stratum_m": STRATUM_M,
                                                   "representation": f"V2_4S {args.encoder}", "plima": args.plima,
                                                   "minutes": round((time.perf_counter() - t0) / 60, 1),
                                                   "env": environment_info()}, indent=2, default=str),
                                       encoding="utf-8")
    log(f"listo en {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
