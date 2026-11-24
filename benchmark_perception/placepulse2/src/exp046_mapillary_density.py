"""
EXP-046 — Auditoría de densidad de Mapillary (solo METADATA; no se descarga ninguna imagen).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

¿Hay suficientes imágenes de Mapillary alrededor de las imágenes de Place Pulse como para que el grafo de segmentos
tenga contexto (E4), y cuánta cobertura hay en los 4 distritos de Lima (capa complementaria, D6)?

  fetch    teselas de 0.005° que contienen una imagen PP piloto (+ 1 tesela de halo) y bbox de cada distrito;
           `image-extraction/mapillary_client.py`, reanudable. Metadata en image-extraction/data/_mapillary/ (no Git).
  analyze  snap de cada imagen Mapillary al segmento más cercano ≤ 30 m (grafos de EXP-025/041); conteos por
           segmento, ego-1 del segmento de cada imagen PP, distancia a la Mapillary más cercana, años, % panorámicas.

Criterio (D6): E4 es viable en una ciudad si la MEDIANA de imágenes Mapillary de perspectiva en el ego-1 del segmento
de las imágenes PP es ≥ 5.

Uso:  python exp046_mapillary_density.py --stage fetch|analyze [--zones a,b]   Salidas: outputs/exp046/
(`--zones` en analyze evita leer la metadata de una ciudad cuyo fetch sigue en curso).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from utils import OUTPUTS, REPO_ROOT, log

sys.path.insert(0, str(REPO_ROOT / "image-extraction"))

OUT = OUTPUTS / "exp046"
MLY = REPO_ROOT / "image-extraction" / "data" / "_mapillary"
CITIES = ["Santiago", "Atlanta", "Berlin"]           # prioridad: Santiago (cercanía a Lima)
LIMA = ["barranco", "la_victoria", "san_isidro", "lince"]
TILE_DEG = 0.005
SNAP_MAX_M = 30.0
MIN_EGO1 = 5


def pp_pilot():
    from exp026_additive_context import load_images
    man, _ = load_images("dinov2_vit_b14")
    return man[man["city"].isin(CITIES)].reset_index(drop=True)


def stage_fetch(zones: list[str]) -> None:
    import geopandas as gpd

    import mapillary_client as mc
    from exp041_lima_segment_graph import SNAPSHOTS
    MLY.mkdir(parents=True, exist_ok=True)
    client = mc.client_from_env(per_s=5.0)
    man = pp_pilot()
    for z in zones:
        t0 = time.perf_counter()
        if z in CITIES:
            g = man[man["city"] == z]
            tiles = mc.tiles_around(list(zip(g["lon"], g["lat"])), TILE_DEG, halo=1)
        else:
            poly = gpd.read_file(SNAPSHOTS / f"{z}_boundary.geojson").to_crs(4326).geometry.iloc[0]
            x0, y0, x1, y1 = poly.bounds
            tiles = mc.tiles_bbox((x0 - 0.002, y0 - 0.002, x1 + 0.002, y1 + 0.002), TILE_DEG)
        client.truncated, client.failed = [], []
        df = mc.fetch_tiles(client, tiles, MLY / f"{z}.csv.gz")
        log(f"{z}: {len(tiles)} teselas → {len(df):,} imágenes, {len(client.truncated)} truncadas, {len(client.failed)} fallidas, "
            f"{(time.perf_counter() - t0) / 60:.1f} min")


def snap_points(seg_geoms, lonlat: np.ndarray, crs) -> tuple[np.ndarray, np.ndarray]:
    """Segmento más cercano ≤ SNAP_MAX_M (vectorizado con STRtree.query_nearest); −1 si no hay."""
    import geopandas as gpd
    from shapely import STRtree
    pts = gpd.GeoSeries(gpd.points_from_xy(lonlat[:, 0], lonlat[:, 1]), crs=4326).to_crs(crs)
    tree = STRtree(list(seg_geoms))
    (pi, si), d = tree.query_nearest(pts.values, max_distance=SNAP_MAX_M, return_distance=True, all_matches=False)
    seg_of = np.full(len(lonlat), -1)
    seg_of[pi] = si
    xy = np.c_[pts.x, pts.y]
    return seg_of, xy


def density_rows(name: str, seg, A, M: pd.DataFrame, seg_of_m: np.ndarray) -> dict:
    n_seg = len(seg)
    persp = ~M["is_pano"].astype(bool).to_numpy()
    cnt_all = np.bincount(seg_of_m[seg_of_m >= 0], minlength=n_seg)
    cnt_p = np.bincount(seg_of_m[(seg_of_m >= 0) & persp], minlength=n_seg)
    yrs = M["year"].dropna().astype(int)
    return dict(zone=name, mapillary_images=int(len(M)), pct_snapped=100 * float((seg_of_m >= 0).mean()),
                pct_pano=100 * float(M["is_pano"].astype(bool).mean()),
                year_min=int(yrs.min()) if len(yrs) else None, year_max=int(yrs.max()) if len(yrs) else None,
                pct_2016=100 * float((yrs == 2016).mean()) if len(yrs) else 0.0,
                pct_2020_plus=100 * float((yrs >= 2020).mean()) if len(yrs) else 0.0,
                segments=n_seg, pct_segments_with_any=100 * float((cnt_all > 0).mean()),
                pct_segments_with_perspective=100 * float((cnt_p > 0).mean()),
                images_per_segment_median=float(np.median(cnt_all)), images_per_segment_p90=float(np.percentile(cnt_all, 90)),
                perspective_per_segment_median=float(np.median(cnt_p))), cnt_all, cnt_p


def stage_analyze(zones: list[str]) -> None:
    import osmnx as ox

    from exp041_lima_segment_graph import load_graph
    from exp042_pp_segment_graph_models import GRAPHS
    from segment_graph import dual_adjacency, segments_table
    OUT.mkdir(parents=True, exist_ok=True)
    man = pp_pilot()
    zone_rows, pp_rows, years = [], [], []
    for city in [c for c in CITIES if c in zones]:
        p = MLY / f"{city}.csv.gz"
        if not p.exists():
            log(f"{city}: sin metadata (¿falta fetch?)")
            continue
        M = pd.read_csv(p, dtype={"id": str}).dropna(subset=["lon", "lat"]).reset_index(drop=True)
        Gp = ox.project_graph(ox.convert.to_undirected(ox.load_graphml(GRAPHS / f"{city}_drive.graphml")))
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        seg_of_m, xy_m = snap_points(seg["geometry"], M[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
        row, cnt_all, cnt_p = density_rows(city, seg, A, M, seg_of_m)
        g = man[man["city"] == city].reset_index(drop=True)
        seg_of_pp, xy_pp = snap_points(seg["geometry"], g[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
        A1 = A + sp.identity(A.shape[0], format="csr")
        ego1_all, ego1_p = A1 @ cnt_all, A1 @ cnt_p
        ok = seg_of_pp >= 0
        d_near, _ = cKDTree(xy_m).query(xy_pp) if len(xy_m) else (np.full(len(g), np.inf), None)
        e0 = np.where(ok, cnt_all[np.clip(seg_of_pp, 0, None)], 0)
        e1 = np.where(ok, ego1_all[np.clip(seg_of_pp, 0, None)], 0)
        e1p = np.where(ok, ego1_p[np.clip(seg_of_pp, 0, None)], 0)
        row.update(pp_images=int(len(g)), pp_pct_snapped=100 * float(ok.mean()),
                   pp_ego0_median=float(np.median(e0)), pp_ego1_median=float(np.median(e1)),
                   pp_ego1_perspective_median=float(np.median(e1p)),
                   pp_pct_ego1_perspective_ge5=100 * float((e1p >= MIN_EGO1).mean()),
                   pp_nearest_mapillary_m_median=float(np.median(d_near)),
                   pp_pct_nearest_le_20m=100 * float((d_near <= 20).mean()),
                   criterion_E4_viable=bool(np.median(e1p) >= MIN_EGO1),
                   truncated_tiles=len(json.loads((MLY / f"{city}.truncated.json").read_text()))
                   if (MLY / f"{city}.truncated.json").exists() else 0)
        zone_rows.append(row)
        pp_rows.append(pd.DataFrame({"location_id": g["location_id"], "city": city, "segment_snapped": ok,
                                     "mly_ego0": e0, "mly_ego1": e1, "mly_ego1_perspective": e1p,
                                     "nearest_mly_m": d_near}))
        years.append(M["year"].value_counts().rename(city))
        log(f"{city}: {row}")
    for z in [z for z in LIMA if z in zones]:
        p = MLY / f"{z}.csv.gz"
        if not p.exists():
            log(f"{z}: sin metadata (¿falta fetch?)")
            continue
        M = pd.read_csv(p, dtype={"id": str}).dropna(subset=["lon", "lat"]).reset_index(drop=True)
        Gp, poly = load_graph(z)
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        seg_of_m, _ = snap_points(seg["geometry"], M[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
        row, *_ = density_rows(z, seg, A, M, seg_of_m)
        zone_rows.append(row)
        years.append(M["year"].value_counts().rename(z))
        log(f"{z}: {row}")
    Z = pd.DataFrame(zone_rows)
    Z.to_csv(OUT / "density_by_zone.csv", index=False, float_format="%.3f")
    if pp_rows:
        pd.concat(pp_rows).to_csv(OUT / "pp_image_context.csv.gz", index=False, compression="gzip", float_format="%.2f")
    pd.concat(years, axis=1).sort_index().fillna(0).astype(int).to_csv(OUT / "years_by_zone.csv")
    viable = [r["zone"] for r in zone_rows if r.get("criterion_E4_viable")]
    (OUT / "summary.json").write_text(json.dumps({"criterion": f"mediana ego-1 perspectiva ≥ {MIN_EGO1}",
                                                  "viable_cities": viable,
                                                  "chosen_city": next((c for c in CITIES if c in viable), None)},
                                                 indent=2), encoding="utf-8")
    log(f"ciudades viables para E4: {viable}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fetch", "analyze", "all"], default="all")
    ap.add_argument("--zones", default=",".join(LIMA + CITIES))
    args = ap.parse_args()
    if args.stage in ("fetch", "all"):
        stage_fetch(args.zones.split(","))
    if args.stage in ("analyze", "all"):
        stage_analyze(args.zones.split(","))


if __name__ == "__main__":
    main()
