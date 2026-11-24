"""
EXP-025 (piloto) — vecindario por conectividad de la red vial vs radio euclídeo, Place Pulse 2.0.

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-025, commit 6b3109f).
Origen: reunión con el asesor del 04/10/2026 (docs/SESSION4_REQUIREMENTS.md), decisión M21.

QUÉ HACE
--------
Para 3 ciudades piloto (NO las 56):
  1. Descarga la red vial de OSM (network_type="drive") sobre el bbox de las imágenes + 1 km,
     la proyecta a UTM y la convierte a no dirigida. Cachea el .graphml.
  2. Asigna cada imagen al nodo más cercano (snap) y registra la distancia.
  3. QA del grafo (R1): snap, componentes, nodos aislados, duplicados, vecinas por k, fuga train/test.
  4. Construye features 1536-d (formato de train_probe.py), iguales a EXP-021:
       ego0      = [z_i, 0]
       euc250    = [mean(z_i, vecinas a <= 250 m), std(...)]          (B1, como EXP-021)
       euc_match = idem con las n vecinas euclídeas más cercanas, n = nº de vecinas en C (B2)
       graph_k*  = idem con vecinas a <= k* saltos en la red vial      (C)
     k* = menor k (1..K_MAX) con <= 25 % de imágenes sin vecinas (agregado de las 3 ciudades).
     Tope 20 vecinas (paridad con EXP-021). Sin GNN.
  5. Guarda listas de vecinas (para R2) y el split image_disjoint filtrado a las 3 ciudades (R3).

Uso:
    python build_street_graph_pilot.py
Salidas: outputs/street_graph_pilot/
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree

from utils import FEATURES, MANIFESTS, OUTPUTS, SPLITS, log, set_seed

SEED = 42
OUT = OUTPUTS / "street_graph_pilot"
PILOT_CITIES = ["Atlanta", "Berlin", "Santiago"]
COORD_RE = re.compile(r"images/(-?\d+\.\d+)_(-?\d+\.\d+)_")
BUFFER_DEG_LAT = 1000 / 110540          # 1 km
EUC_RADIUS_M = 250.0
NEIGH_MAX = 20
K_MAX = 10
K_TARGET_ZERO = 0.25


def parse_coords(p: str):
    m = COORD_RE.match(p)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def get_graph(city: str, g: pd.DataFrame, network_type: str) -> nx.MultiDiGraph:
    path = OUT / "graphs" / f"{city.replace(' ', '_')}_{network_type}.graphml"
    if path.exists():
        log(f"  {city}: grafo en caché {path.name}")
        return ox.load_graphml(path)
    lat0 = g["lat"].mean()
    dlon = BUFFER_DEG_LAT / np.cos(np.radians(lat0))
    bbox = (g["lon"].min() - dlon, g["lat"].min() - BUFFER_DEG_LAT,
            g["lon"].max() + dlon, g["lat"].max() + BUFFER_DEG_LAT)   # (left, bottom, right, top)
    t = time.perf_counter()
    G = ox.graph_from_bbox(bbox, network_type=network_type, simplify=True)
    log(f"  {city}: descargado {len(G):,} nodos en {time.perf_counter() - t:.0f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return G


def agg(Z: np.ndarray, rows: np.ndarray, self_row: int, neigh: list[int]) -> np.ndarray:
    pool = Z[np.r_[self_row, rows[neigh]]] if neigh else Z[[self_row]]
    return np.r_[pool.mean(axis=0), pool.std(axis=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network-type", default="drive")
    ap.add_argument("--encoder", default="dinov2_vit_b14")
    args = ap.parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ── imágenes y embeddings (idéntico a build_placepulse_context.py / EXP-021) ──────────────
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).dropna(subset=["image_path"])
    ll = man["image_path"].apply(parse_coords)
    man["lat"], man["lon"] = ll.apply(lambda t: t[0]), ll.apply(lambda t: t[1])
    man = man.dropna(subset=["lat", "lon"])
    man = man[man["city"].isin(PILOT_CITIES)]

    d = FEATURES / args.encoder
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    shards = sorted(d.glob("shard_*.npy"))
    Z = np.concatenate([np.load(s) for s in shards], axis=0) if shards else np.load(d / "features.npy")
    row_of = {lid: i for i, lid in enumerate(index["location_ids"][: Z.shape[0]])}
    man = man[man["location_id"].isin(row_of)].reset_index(drop=True)
    man["row"] = man["location_id"].map(row_of)
    dim = Z.shape[1]
    log(f"piloto: {len(man):,} imágenes en {PILOT_CITIES}; embeddings {Z.shape}")

    # ── pertenencia de imágenes al split image_disjoint (para medir fuga por el grafo) ────────
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    split = json.loads((SPLITS / "image_disjoint.json").read_text())
    img_split = {}
    for key in ("train", "val", "test"):
        sub = pairs.iloc[split[key]]
        for lid in pd.concat([sub["left"], sub["right"]]).unique():
            img_split.setdefault(lid, key)
    man["split"] = man["location_id"].map(img_split).fillna("none")

    # ── por ciudad: grafo, snap, vecindarios ──────────────────────────────────────────────────
    qa_rows, hop_rows, neigh_records = [], [], []
    per_city = {}
    for city in PILOT_CITIES:
        g = man[man["city"] == city]
        G = get_graph(city, g, args.network_type)
        Gp = ox.project_graph(G)
        Gu = ox.convert.to_undirected(Gp)
        Gs = nx.Graph(Gu)                                   # simple, para BFS por saltos
        crs = Gp.graph["crs"]

        import geopandas as gpd
        pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"].astype(float), g["lat"].astype(float)),
                            crs="EPSG:4326").to_crs(crs)
        X, Y = pts.x.to_numpy(), pts.y.to_numpy()
        nodes, dist = ox.distance.nearest_nodes(Gp, X, Y, return_dist=True)
        nodes, dist = np.asarray(nodes), np.asarray(dist, dtype=float)

        comps = sorted(nx.connected_components(Gs), key=len, reverse=True)
        comp_of = {n: ci for ci, c in enumerate(comps) for n in c}
        in_largest = np.array([comp_of[n] == 0 for n in nodes])
        isolated = sum(1 for n in Gs.nodes if Gs.degree(n) == 0)
        node_counts = pd.Series(nodes).value_counts()
        coord_dup = g.duplicated(subset=["lat", "lon"], keep=False).sum()

        qa_rows.append(dict(
            city=city, network_type=args.network_type, n_images=len(g),
            graph_nodes=Gs.number_of_nodes(), graph_edges=Gs.number_of_edges(),
            n_components=len(comps), largest_component_nodes=len(comps[0]),
            isolated_nodes=isolated,
            pct_images_snapped=100.0 * np.isfinite(dist).mean(),
            snap_dist_median_m=float(np.median(dist)), snap_dist_p90_m=float(np.percentile(dist, 90)),
            snap_dist_p99_m=float(np.percentile(dist, 99)),
            pct_snap_le_100m=100.0 * (dist <= 100).mean(),
            pct_images_in_largest_component=100.0 * in_largest.mean(),
            distinct_snapped_nodes=int(node_counts.size),
            pct_images_sharing_node=100.0 * (pd.Series(nodes).map(node_counts) > 1).mean(),
            images_same_coordinates=int(coord_dup),
        ))

        # imágenes por nodo
        imgs_at = {}
        for j, n in enumerate(nodes):
            imgs_at.setdefault(n, []).append(j)

        # vecinas por saltos (k = 0..K_MAX), ordenadas por (salto, distancia euclídea)
        xy = np.c_[X, Y]
        hop_lists = []
        for j, n in enumerate(nodes):
            reach = nx.single_source_shortest_path_length(Gs, n, cutoff=K_MAX)
            cand = [(h, jj) for nn, h in reach.items() for jj in imgs_at.get(nn, []) if jj != j]
            cand.sort(key=lambda t: (t[0], float(np.hypot(*(xy[t[1]] - xy[j])))))
            hop_lists.append(cand)

        tree = cKDTree(xy)
        per_city[city] = dict(g=g, nodes=nodes, dist=dist, xy=xy, tree=tree, hop_lists=hop_lists)

        for k in range(1, K_MAX + 1):
            nk = np.array([sum(1 for h, _ in c if h <= k) for c in hop_lists])
            hop_rows.append(dict(city=city, k=k, mean_neighbors=float(nk.mean()),
                                 median_neighbors=float(np.median(nk)),
                                 pct_zero_neighbors=100.0 * float((nk == 0).mean())))
        e = np.array([len(tree.query_ball_point(xy[j], EUC_RADIUS_M)) - 1 for j in range(len(g))])
        hop_rows.append(dict(city=city, k="euc250", mean_neighbors=float(e.mean()),
                             median_neighbors=float(np.median(e)),
                             pct_zero_neighbors=100.0 * float((e == 0).mean())))
        log(f"  {city}: snap mediana {np.median(dist):.1f} m, {in_largest.mean():.1%} en componente mayor")

    hop = pd.DataFrame(hop_rows)
    pooled = (hop[hop["k"] != "euc250"].groupby("k")
              .apply(lambda t: np.average(t["pct_zero_neighbors"],
                                          weights=[len(per_city[c]["g"]) for c in t["city"]]))
              .sort_index())
    ok = pooled[pooled <= 100 * K_TARGET_ZERO]
    k_star = int(ok.index.min()) if len(ok) else K_MAX
    log(f"k* = {k_star} (pct sin vecinas agregado por k: {pooled.round(1).to_dict()})")

    # ── features y vecinas ────────────────────────────────────────────────────────────────────
    variants = {v: np.zeros((len(man), 2 * dim), dtype=np.float32)
                for v in ("ego0", "euc250", "euc_match", f"graph_k{k_star}")}
    leak_rows = []
    for city, pc in per_city.items():
        g, xy, tree = pc["g"], pc["xy"], pc["tree"]
        idx_global = g.index.to_numpy()
        rows = g["row"].to_numpy()
        splits_local = g["split"].to_numpy()
        n_test, n_test_ctx_train = 0, 0
        n_train, n_train_ctx_test = 0, 0
        for j in range(len(g)):
            gi = idx_global[j]
            C = [jj for h, jj in pc["hop_lists"][j] if h <= k_star][:NEIGH_MAX]
            B1 = [jj for jj in tree.query_ball_point(xy[j], EUC_RADIUS_M) if jj != j]
            if len(B1) > NEIGH_MAX:
                B1 = sorted(B1, key=lambda jj: np.hypot(*(xy[jj] - xy[j])))[:NEIGH_MAX]
            if C:
                _, nn = tree.query(xy[j], k=len(C) + 1)
                B2 = [jj for jj in np.atleast_1d(nn) if jj != j][: len(C)]
            else:
                B2 = []
            variants["ego0"][gi, :dim] = Z[rows[j]]
            variants["euc250"][gi] = agg(Z, rows, rows[j], B1)
            variants["euc_match"][gi] = agg(Z, rows, rows[j], B2)
            variants[f"graph_k{k_star}"][gi] = agg(Z, rows, rows[j], C)
            lid = lambda L: [g["location_id"].iloc[jj] for jj in L]
            neigh_records.append(dict(location_id=g["location_id"].iloc[j], city=city,
                                      split=splits_local[j], graph=lid(C), euc_match=lid(B2),
                                      euc250=lid(B1)))
            if splits_local[j] == "test":
                n_test += 1
                n_test_ctx_train += any(splits_local[jj] == "train" for jj in C)
            if splits_local[j] == "train":
                n_train += 1
                n_train_ctx_test += any(splits_local[jj] == "test" for jj in C)
        leak_rows.append(dict(city=city, k=k_star, n_test_images=n_test,
                              pct_test_with_train_in_context=100.0 * n_test_ctx_train / max(n_test, 1),
                              n_train_images=n_train,
                              pct_train_with_test_in_context=100.0 * n_train_ctx_test / max(n_train, 1)))

    ids = man["location_id"].tolist()
    for name, F in variants.items():
        sub = OUT / "features" / name
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "features.npy", F)
        (sub / "index.json").write_text(json.dumps({
            "encoder_base": args.encoder, "variant": name, "embed_dim": int(F.shape[1]),
            "n_images": len(ids), "location_ids": ids, "cities": PILOT_CITIES,
            "network_type": args.network_type, "k_star": k_star, "neigh_max": NEIGH_MAX,
            "euclidean_radius_m": EUC_RADIUS_M}, indent=2), encoding="utf-8")

    # split image_disjoint filtrado a pares con ambas imágenes en las 3 ciudades (R3)
    S = set(PILOT_CITIES)
    both = (pairs["city_left"].isin(S) & pairs["city_right"].isin(S)).to_numpy()
    filt = {k: [i for i in v if both[i]] for k, v in split.items() if isinstance(v, list)}
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    (OUT / "splits" / "image_disjoint.json").write_text(json.dumps(filt), encoding="utf-8")

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUT / "graph_qa.csv", index=False)
    hop.to_csv(OUT / "neighbors_by_k.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(OUT / "graph_leakage.csv", index=False)
    pd.DataFrame(neigh_records).to_json(OUT / "neighbors.jsonl", orient="records", lines=True)
    (OUT / "build_summary.json").write_text(json.dumps({
        "seed": SEED, "cities": PILOT_CITIES, "network_type": args.network_type,
        "k_star": k_star, "pct_zero_by_k_pooled": {int(k): round(float(v), 2) for k, v in pooled.items()},
        "neigh_max": NEIGH_MAX, "euclidean_radius_m": EUC_RADIUS_M,
        "split_sizes_filtered": {k: len(v) for k, v in filt.items()},
        "osmnx_version": ox.__version__, "networkx_version": nx.__version__,
        "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
    log(qa.T.to_string())
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
