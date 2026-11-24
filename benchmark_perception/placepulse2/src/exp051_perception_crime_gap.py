"""
EXP-051 — Percepción PREDICHA (probe SimCLR de Place Pulse) vs crimen registrado en Lima (S8, brecha).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

POR QUE
-------
El EC1 (§1.2) parte de que el peligro real y el percibido se relacionan pero no son lo mismo (*perception bias*,
Zhang 2021). Aquí se mide en Lima: el probe BT de Place Pulse (SimCLR epoch_50, el mismo checkpoint de los embeddings
de Lima) puntúa la media de las 12 vistas de cada spot y se compara con los registros de crimen por segmento de calle.
NO es percepción medida en Lima (D10, M01): es lo que un modelo entrenado con juicios humanos de otras ciudades "ve".

Uso:  python exp051_perception_crime_gap.py [--encoder simclr|dinov2]   Salidas: outputs/exp051/ (o exp051_dinov2/)
--encoder dinov2: probe BT sobre DINOv2 ViT-B/14 en PP, aplicado a la media de los CLS de las 12 vistas de cada spot.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp041_lima_segment_graph import SNAP_MAX_M, load_graph, lima_spots
from lima_views import CRIMES
from segment_graph import segments_table
from utils import OUTPUTS, environment_info, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp051"
SNAP = OUTPUTS / "exp041_plima" / "spot_to_segment.csv"
CRIME_DISTRICT = {"barranco": "Barranco", "la_victoria": "La Victoria", "san_isidro": "San Isidro"}
N_BOOT = 2000


def pp_probe(features: str = "ours_simclr_vit_b16"):
    from probe_utils import load_pairs, load_split, train_scores
    from train_probe import load_features
    X, row_of = load_features(features)
    ids = sorted(row_of, key=row_of.get)
    _, info = train_scores(X, ids, load_pairs(), load_split("image_disjoint"))
    w = info["weights"]
    return w, np.sort(X.astype(np.float32) @ w), info["metrics"]["pairwise_accuracy"]


def snap_crimes(seg, Gp, lonlat):
    import geopandas as gpd
    from shapely import STRtree
    pts = gpd.GeoSeries(gpd.points_from_xy(lonlat[:, 0], lonlat[:, 1]), crs=4326).to_crs(Gp.graph["crs"])
    (pi, si), _ = STRtree(list(seg["geometry"])).query_nearest(pts.values, max_distance=SNAP_MAX_M,
                                                              return_distance=True, all_matches=False)
    return np.bincount(si, minlength=len(seg)), len(pi) / max(len(lonlat), 1)


def boot_spearman(x, y, rng):
    r = spearmanr(x, y).statistic
    n = len(x)
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(r), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["simclr", "dinov2"], default="simclr")
    args = ap.parse_args()
    if args.encoder == "dinov2":
        OUT = OUTPUTS / "exp051_dinov2"
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    w, pp_sorted, acc = pp_probe("dinov2_vit_b14" if args.encoder == "dinov2" else "ours_simclr_vit_b16")
    log(f"probe SimCLR PP: exactitud test image_disjoint {acc:.4f}")
    S, V, E, Zspot = lima_spots(plima=True)
    if args.encoder == "dinov2":
        from exp052_connectivity_effect_size import DINO, dino_spot
        dz = {d: dino_spot(f, normalize=False) for d, f in DINO.items()}   # media de 12 vistas, como V1
        Zspot = np.stack([dz[d][s] for d, s in zip(S["district"], S["spot_id"])])
    S["score"] = Zspot @ w
    S["pct_pp"] = 100 * np.searchsorted(pp_sorted, S["score"]) / len(pp_sorted)
    snap = pd.read_csv(SNAP, dtype={"spot_id": str})
    S = S.merge(snap[["district", "spot_id", "seg_idx"]], on=["district", "spot_id"], how="left")
    S.to_csv(OUT / "spot_predicted_perception.csv", index=False, float_format="%.4f")

    C = pd.read_csv(CRIMES, sep=";")
    dist_rows, seg_rows, corr_rows, quad_rows = [], [], [], []
    for d in ("barranco", "la_victoria", "san_isidro", "lince"):
        g = S[S["district"] == d]
        Gp, poly = load_graph(d)
        seg = segments_table(Gp)
        bs = [np.median(g["pct_pp"].to_numpy()[rng.integers(0, len(g), len(g))]) for _ in range(N_BOOT)]
        row = dict(district=d, sampling="street_sampling" if d == "lince" else "crime_points", n_spots=len(g),
                   pct_pp_median=float(g["pct_pp"].median()), ci_low=float(np.percentile(bs, 2.5)),
                   ci_high=float(np.percentile(bs, 97.5)), area_km2=poly.area / 1e6)
        if d in CRIME_DISTRICT:
            Cd = C[C["DISTRITO"] == CRIME_DISTRICT[d]]
            for label, years in (("2016", [2016]), ("2016_2023", list(range(2016, 2024)))):
                Cy = Cd[Cd["YEAR"].isin(years)]
                cnt, frac = snap_crimes(seg, Gp, Cy[["longitude", "latitude"]].to_numpy())
                row[f"crimes_{label}"] = len(Cy)
                row[f"crimes_per_km2_{label}"] = len(Cy) / row["area_km2"]
                row[f"crimes_snapped_{label}"] = frac
                gs = g[g["seg_idx"] >= 0].groupby("seg_idx")["pct_pp"].mean()
                x, y = gs.to_numpy(), np.log1p(cnt[gs.index.to_numpy().astype(int)])
                r, lo, hi = boot_spearman(x, y, rng)
                corr_rows.append(dict(district=d, crimes=label, n_segments=len(gs), spearman=r, ci_low=lo, ci_high=hi))
                if label == "2016":
                    seg_rows.append(pd.DataFrame({"district": d, "seg_idx": gs.index, "pct_pp_mean": x,
                                                  "crimes_2016": cnt[gs.index.to_numpy().astype(int)]}))
                    tp = pd.qcut(pd.Series(x).rank(method="first"), 3, labels=["inseguro", "medio", "seguro"])
                    tc = pd.qcut(pd.Series(y).rank(method="first"), 3, labels=["bajo", "medio", "alto"])
                    ct = pd.crosstab(tp, tc, normalize=True) * 100
                    quad_rows.append(dict(district=d, n_segments=len(gs),
                                          looks_safe_high_crime=float(ct.loc["seguro", "alto"]),
                                          looks_unsafe_low_crime=float(ct.loc["inseguro", "bajo"]),
                                          agree_safe_low=float(ct.loc["seguro", "bajo"]),
                                          agree_unsafe_high=float(ct.loc["inseguro", "alto"])))
        dist_rows.append(row)
        log(f"{d}: {row}")
    D = pd.DataFrame(dist_rows)
    D.to_csv(OUT / "by_district.csv", index=False, float_format="%.4f")
    R = pd.DataFrame(corr_rows)
    R.to_csv(OUT / "segment_spearman.csv", index=False, float_format="%.4f")
    Q = pd.DataFrame(quad_rows)
    Q.to_csv(OUT / "gap_quadrants.csv", index=False, float_format="%.2f")
    pd.concat(seg_rows).to_csv(OUT / "segment_table.csv", index=False, float_format="%.4f")
    summ = {"probe_test_accuracy_pp": acc, "by_district": D.to_dict("records"), "segment_spearman": R.to_dict("records"),
            "gap_quadrants": Q.to_dict("records"), "seconds": round(time.perf_counter() - t0),
            "environment": environment_info()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    log(R.to_string())
    log(Q.to_string())
    log(f"listo en {summ['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
