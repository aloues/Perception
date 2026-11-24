"""
EXP-041 — Lima: grafo de SEGMENTOS (dual) en Barranco, La Victoria, San Isidro y Lince; ego-k; y
¿son los vecinos por RED visualmente más parecidos que los vecinos por RADIO? (sin etiquetas).

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-041, 2026-10-23).

POR QUE
-------
EXP-026 mostró que en Place Pulse la comparación red vs radio no es identificable: con ~1 imagen cada
150 m ambas definiciones eligen las mismas imágenes en el 75–92 % de los casos. Las 12 vistas de
Lima son mucho más densas (puntos de crimen 2016, ~1,800 spots en dos distritos). Aquí se construye la
unidad que pidió el asesor (la CALLE = segmento entre intersecciones, como nodo de un grafo dual), se
definen ego-0/1/2 y se mide, sin etiquetas, si los segmentos conectados se parecen más que los cercanos
en línea recta. Similitud visual ≠ percepción: esto informa el diseño del grafo, no la tesis perceptual.

Uso:
    python exp041_lima_segment_graph.py            # B/LV histórico → outputs/exp041/
    python exp041_lima_segment_graph.py --plima    # S4/S5 en los 4 distritos (P-LIMA) → outputs/exp041_plima/
En --plima se conserva un spot por panorama (M28) y el vocabulario BoVW de cada distrito se ajusta con su pareja
(Barranco↔La Victoria, San Isidro↔Lince).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import figstyle
from lima_views import attach_coordinates, load_views, spots_table
from segment_graph import (dual_adjacency, euclidean_count_matched, hop_distances, reach_matrices,
                           segment_midpoints, segments_table, snap_to_segments)
from utils import OUTPUTS, REPO_ROOT, environment_info, log, set_seed
from visual_repr import BoVW, histograms, js_divergence, kl_divergence, l2n

SEED = 42
OUT = OUTPUTS / "exp041"
DISTRICTS = {"barranco": "Barranco, Lima, Peru", "la_victoria": "La Victoria, Lima, Peru",
             "san_isidro": "San Isidro, Lima, Peru", "lince": "Lince, Lima, Peru"}
SNAP_MAX_M, CORNER_M = 30.0, 15.0
K_BOVW = 100
N_BOOT = 2000


SNAPSHOTS = REPO_ROOT / "external_assets" / "osm_cache" / "graphs"


def load_graph(name: str):
    """Snapshot congelado (GraphML + límite GeoJSON) de graph_from_place(drive, simplify) — ver snapshot_info.json."""
    import geopandas as gpd
    import osmnx as ox
    G = ox.load_graphml(SNAPSHOTS / f"{name}_drive.graphml")
    poly = gpd.read_file(SNAPSHOTS / f"{name}_boundary.geojson")
    Gp = ox.project_graph(ox.convert.to_undirected(G))
    return Gp, poly.to_crs(Gp.graph["crs"]).geometry.iloc[0]


def graph_qa(name: str, Gp, seg: pd.DataFrame, A, poly) -> dict:
    import networkx as nx
    from shapely.geometry import Point
    deg = np.array([d for _, d in Gp.degree()])
    comps = nx.number_connected_components(nx.Graph(Gp))
    dual_deg = np.asarray(A.sum(axis=1)).ravel()
    R = reach_matrices(A, np.arange(len(seg)), 3)
    dead_near_border = sum(1 for n, d in Gp.degree() if d == 1 and
                           poly.exterior.distance(Point(Gp.nodes[n]["x"], Gp.nodes[n]["y"])) < 30)
    return {"district": name, "intersections_nodes": Gp.number_of_nodes(), "segments": len(seg),
            "components_primal": comps, "dual_edges": int(A.nnz // 2),
            "segment_length_median_m": round(float(seg["length_m"].median()), 1),
            "segment_length_p90_m": round(float(seg["length_m"].quantile(0.9)), 1),
            "node_degree_mean": round(float(deg.mean()), 2), "dead_ends": int((deg == 1).sum()),
            "dead_ends_within_30m_of_border": dead_near_border,
            "segments_bridge": int(seg["bridge"].sum()), "segments_tunnel": int(seg["tunnel"].sum()),
            "segments_highway_trunk_or_motorway": int(seg["highway"].str.contains("trunk|motorway").sum()),
            "dual_degree_mean": round(float(dual_deg.mean()), 2),
            "isolated_segments_in_dual": int((dual_deg == 0).sum()),
            "ego1_size_median": float(np.median(np.asarray(R[1].sum(axis=1)).ravel())),
            "ego2_size_median": float(np.median(np.asarray(R[2].sum(axis=1)).ravel())),
            "ego3_size_median": float(np.median(np.asarray(R[3].sum(axis=1)).ravel()))}


VOCAB_PAIR = {"barranco": "la_victoria", "la_victoria": "barranco", "san_isidro": "lince", "lince": "san_isidro"}


def lima_spots(plima: bool = False):
    V, E = load_views(plima=plima)
    V = attach_coordinates(V)
    S = spots_table(V)
    S = S[S["complete_12"] & S["coord_ok"]].reset_index(drop=True)
    if plima:                                           # M28: un spot por panorama (vistas idénticas)
        pano = V.groupby(["district", "spot_id"])["pano_id"].first()
        S["pano_id"] = [pano.get((d, k)) for d, k in zip(S["district"], S["spot_id"])]
        dup = S["pano_id"].notna() & S.duplicated(["district", "pano_id"])
        log(f"spots con panorama repetido descartados: {S.loc[dup, 'district'].value_counts().to_dict()}")
        S = S[~dup].drop(columns="pano_id").reset_index(drop=True)
    V = V.merge(S[["district", "spot_id"]].reset_index().rename(columns={"index": "sidx"}),
                on=["district", "spot_id"])
    E = np.asarray(E)
    Zspot = np.zeros((len(S), E.shape[1]), np.float32)
    np.add.at(Zspot, V["sidx"].to_numpy(), E[V["row"].to_numpy()])
    Zspot /= 12.0                                       # V1: media de las 12 vistas
    return S, V, E, Zspot


def bootstrap_mean(x: np.ndarray, rng) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def homogeneity(name, seg, A, spot_seg, Zspot, Hspot, rng, label: str) -> tuple[list, pd.DataFrame, dict]:
    """Distancia del segmento al promedio de (red anillo-1 | euclídeo | euclídeo disjunto | aleatorio)."""
    F = np.unique(spot_seg[spot_seg >= 0])                         # segmentos con features
    pos = {s: i for i, s in enumerate(F)}
    Zseg = np.zeros((len(F), Zspot.shape[1]))
    Hseg = np.zeros((len(F), Hspot.shape[1]))
    nsp = np.zeros(len(F))
    for i, s in enumerate(spot_seg):
        if s >= 0:
            Zseg[pos[s]] += Zspot[i]
            Hseg[pos[s]] += Hspot[i]
            nsp[pos[s]] += 1
    Zseg = l2n(Zseg / nsp[:, None])
    D = hop_distances(A, F, 4).tocsr()
    Fset = set(F.tolist())
    ring = {k: [] for k in range(1, 5)}
    for r in range(len(F)):
        cols, vals = D.indices[D.indptr[r]:D.indptr[r + 1]], D.data[D.indptr[r]:D.indptr[r + 1]]
        for k in range(1, 5):
            ring[k].append(np.array([pos[c] for c, v in zip(cols, vals) if v == k and c in Fset], dtype=int))
    mid = segment_midpoints(seg)[F]
    counts = np.array([len(x) for x in ring[1]])
    euc = euclidean_count_matched(mid, np.arange(len(F)), np.arange(len(F)), counts)
    euc_dis = euclidean_count_matched(mid, np.arange(len(F)), np.arange(len(F)), counts,
                                      exclude=[set(x.tolist()) for x in ring[1]])
    rand = [rng.choice(np.setdiff1d(np.arange(len(F)), [r]), size=c, replace=False) if c else np.array([], int)
            for r, c in enumerate(counts)]
    sets = {"network_ring1": ring[1], "euclid_count_matched": euc, "euclid_disjoint": euc_dis, "random": rand}
    rows = []
    focal = np.flatnonzero(counts > 0)
    dist = {}
    for key, L in sets.items():
        dc, dj, dk = [], [], []
        for r in focal:
            m = l2n(Zseg[L[r]].mean(axis=0))
            dc.append(1 - float(Zseg[r] @ m))
            pooled = Hseg[L[r]].sum(axis=0, keepdims=True)
            p = Hseg[r:r + 1] / Hseg[r].sum()
            q = pooled / pooled.sum()
            dj.append(float(js_divergence(p, q)[0]))
            dk.append(float(kl_divergence(p, q)[0]))
        dist[key] = {"cos": np.array(dc), "js": np.array(dj), "kl": np.array(dk)}
    overlap = np.array([len(set(ring[1][r]) & set(euc[r])) / counts[r] for r in focal])
    for metric in ("cos", "js", "kl"):
        for key in sets:
            m, lo, hi = bootstrap_mean(dist[key][metric], rng)
            rows.append(dict(district=name, features=label, metric=metric, comparison=key, n_focal=len(focal),
                             mean=m, ci_low=lo, ci_high=hi))
        for other in ("euclid_count_matched", "euclid_disjoint", "random"):
            m, lo, hi = bootstrap_mean(dist[other][metric] - dist["network_ring1"][metric], rng)
            rows.append(dict(district=name, features=label, metric=metric,
                             comparison=f"{other} - network_ring1", n_focal=len(focal), mean=m, ci_low=lo, ci_high=hi))
    # decaimiento: similitud par a par por saltos y por distancia euclídea
    dec = []
    for k in range(1, 5):
        sims = [float(Zseg[r] @ Zseg[j]) for r in range(len(F)) for j in ring[k][r]]
        if sims:
            dec.append(dict(district=name, features=label, axis="hops", bin=k, mean_cos=np.mean(sims), n=len(sims)))
    from scipy.spatial import cKDTree
    pairs = cKDTree(mid).query_pairs(600, output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(mid[pairs[:, 0]] - mid[pairs[:, 1]], axis=1)
        s = np.einsum("nd,nd->n", Zseg[pairs[:, 0]], Zseg[pairs[:, 1]])
        b = (d // 50).astype(int)
        for k in np.unique(b):
            dec.append(dict(district=name, features=label, axis="euclid_50m", bin=int(k) * 50 + 25,
                            mean_cos=float(s[b == k].mean()), n=int((b == k).sum())))
    # POST-HOC (no estaba en el diseño): el control "euclídeo disjunto" excluye a los vecinos de red,
    # que suelen ser los más cercanos → confunde conectividad con distancia. Aquí se comparan pares
    # conectados (1 salto) vs NO conectados (≥ 2 saltos o inalcanzables a 4) A IGUAL DISTANCIA entre
    # puntos medios (estratos de 25 m hasta 300 m); Δ estratificado, bootstrap por segmento focal.
    hop_of = {}
    for r in range(len(F)):
        for k in range(1, 5):
            for j in ring[k][r]:
                hop_of[(r, j)] = k
    pr = cKDTree(mid).query_pairs(300, output_type="ndarray")
    dm = []
    if len(pr):
        dd = np.linalg.norm(mid[pr[:, 0]] - mid[pr[:, 1]], axis=1)
        conn = np.array([hop_of.get((a, b), 99) == 1 for a, b in pr])
        sim = np.einsum("nd,nd->n", Zseg[pr[:, 0]], Zseg[pr[:, 1]])
        js_ = js_divergence(Hseg[pr[:, 0]] / Hseg[pr[:, 0]].sum(1, keepdims=True),
                            Hseg[pr[:, 1]] / Hseg[pr[:, 1]].sum(1, keepdims=True))
        strata = (dd // 25).astype(int)

        def strat_delta(sel_pairs):
            num = den = 0.0
            out = {}
            for metric, v in (("cos_sim", sim), ("js", js_)):
                num = den = 0.0
                for s in np.unique(strata[sel_pairs]):
                    m = sel_pairs & (strata == s)
                    c, nc = m & conn, m & ~conn
                    if c.sum() and nc.sum():
                        w = min(c.sum(), nc.sum())
                        num += w * (v[c].mean() - v[nc].mean())
                        den += w
                out[metric] = num / den if den else np.nan
            return out

        base = strat_delta(np.ones(len(pr), bool))
        focal_of = pr[:, 0]
        uf = np.unique(focal_of)
        boots = {"cos_sim": [], "js": []}
        for _ in range(500):
            pick = rng.choice(uf, len(uf))
            cnt = np.bincount(pick, minlength=len(F))
            w_sel = cnt[focal_of] > 0
            b = strat_delta(w_sel)
            for k_ in boots:
                boots[k_].append(b[k_])
        for metric in ("cos_sim", "js"):
            rows.append(dict(district=name, features=label, metric=f"posthoc_{metric}",
                             comparison="connected - not_connected @ same distance (25 m strata, ≤300 m)",
                             n_focal=int(len(uf)), mean=base[metric],
                             ci_low=np.nanpercentile(boots[metric], 2.5), ci_high=np.nanpercentile(boots[metric], 97.5)))
        dm = dict(pairs_le_300m=int(len(pr)), connected_pairs=int(conn.sum()),
                  median_dist_connected=float(np.median(dd[conn])) if conn.any() else None,
                  median_dist_not_connected=float(np.median(dd[~conn])))
    info = {"featured_segments": int(len(F)), "focal_with_ring1": int(len(focal)), "posthoc_pairs": dm,
            "spots_per_featured_segment_median": float(np.median(nsp)),
            "spots_per_featured_segment_p90": float(np.percentile(nsp, 90)),
            "ring1_featured_neighbors_median": float(np.median(counts[focal])),
            "overlap_network_vs_euclid_mean": float(overlap.mean()),
            "pct_identical_sets": float(100 * (overlap == 1).mean())}
    return rows, pd.DataFrame(dec), info


def plot_maps(graphs: dict, focal: dict, spots_xy: dict, planned_xy: dict) -> None:
    import matplotlib.pyplot as plt
    figstyle.setup()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, (Gp, seg, A, poly)) in zip(axes.ravel(), graphs.items()):
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for g in seg["geometry"]:
            ax.plot(*g.xy, color=figstyle.GRID, lw=0.8, zorder=1)
        x, y = poly.exterior.xy
        ax.plot(x, y, color=figstyle.NEUTRAL, lw=1.0, ls="--", zorder=1)
        f = focal[name]
        R = reach_matrices(A, np.array([f]), 2)
        e2 = set(R[2].indices) - set(R[1].indices)
        e1 = set(R[1].indices) - {f}
        for idxs, c, lw, lab in ((e2, figstyle.SERIES[1], 2.0, "ego-2 (2 saltos)"),
                                 (e1, figstyle.SERIES[0], 2.5, "ego-1 (1 salto)"),
                                 ({f}, figstyle.INK, 3.5, "ego-0 (segmento focal)")):
            for j, i in enumerate(sorted(idxs)):
                ax.plot(*seg["geometry"].iloc[i].xy, color=c, lw=lw, zorder=3, label=lab if j == 0 else None)
        if name in spots_xy:
            xy = spots_xy[name]
            ax.scatter(xy[:, 0], xy[:, 1], s=4, color=figstyle.SERIES[2], zorder=2, label="spots 12 vistas")
        elif name in planned_xy:
            xy = planned_xy[name]
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color=figstyle.SERIES[3], zorder=2,
                       label="spots PLANIFICADOS (no descargados)")
        ax.set_title(f"{name.replace('_', ' ').title()} — {len(seg):,} segmentos")
        ax.legend(loc="lower left", fontsize=7)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "exp041_segment_graphs_ego.png", dpi=150)
    plt.close(fig)
    # zoom en el focal de Barranco
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, name in zip(axes, ("barranco", "la_victoria")):
        Gp, seg, A, poly = graphs[name]
        f = focal[name]
        R = reach_matrices(A, np.array([f]), 2)
        idx = set(R[2].indices)
        geoms = seg["geometry"].iloc[sorted(idx)]
        minx, miny, maxx, maxy = np.array([g.bounds for g in geoms]).T
        pad = 120
        box = (minx.min() - pad, maxx.max() + pad, miny.min() - pad, maxy.max() + pad)
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for g in seg["geometry"]:
            ax.plot(*g.xy, color=figstyle.GRID, lw=1.2, zorder=1)
        e1 = set(R[1].indices) - {f}
        for idxs, c, lw, lab in ((idx - e1 - {f}, figstyle.SERIES[1], 2.5, "ego-2"), (e1, figstyle.SERIES[0], 3, "ego-1"),
                                 ({f}, figstyle.INK, 4, "ego-0")):
            for j, i in enumerate(sorted(idxs)):
                ax.plot(*seg["geometry"].iloc[i].xy, color=c, lw=lw, zorder=3, label=lab if j == 0 else None)
        xy = spots_xy[name]
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=figstyle.SERIES[2], zorder=4, label="spots (12 vistas)")
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_aspect("equal")
        ax.set_title(f"{name.replace('_', ' ').title()}: segmento focal y ego-k")
        ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "exp041_ego_zoom.png", dpi=160)
    plt.close(fig)


def plot_decay(dec: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    figstyle.setup()
    d = dec[dec["features"] == "all_spots"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, axis, xl in ((axes[0], "hops", "saltos en el grafo dual"), (axes[1], "euclid_50m", "distancia entre puntos medios (m)")):
        for c, (name, g) in zip(figstyle.SERIES, d[d["axis"] == axis].groupby("district")):
            g = g[g["n"] >= 30]
            ax.plot(g["bin"], g["mean_cos"], "-o", color=c, label=name.replace("_", " ").title())
        ax.set_xlabel(xl)
        ax.set_ylabel("coseno medio entre segmentos")
        ax.legend()
    axes[0].set_title("Similitud visual vs conectividad")
    axes[1].set_title("Similitud visual vs distancia en línea recta")
    fig.tight_layout()
    fig.savefig(OUT / "exp041_decay.png", dpi=160)
    plt.close(fig)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plima", action="store_true")
    args = ap.parse_args()
    if args.plima:
        OUT = OUTPUTS / "exp041_plima"
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    graphs, qa = {}, []
    for name in DISTRICTS:
        Gp, poly = load_graph(name)
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        graphs[name] = (Gp, seg, A, poly)
        qa.append(graph_qa(name, Gp, seg, A, poly))
        log(f"{name}: {qa[-1]['segments']} segmentos, {qa[-1]['dual_edges']} aristas duales")
    pd.DataFrame(qa).to_csv(OUT / "graph_qa.csv", index=False)

    S, V, E, Zspot = lima_spots(args.plima)
    measured = tuple(DISTRICTS) if args.plima else ("barranco", "la_victoria")
    import geopandas as gpd
    snaps, spots_xy, rows, decs, infos, focal = [], {}, [], [], {}, {}
    for name in measured:
        Gp, seg, A, poly = graphs[name]
        m = (S["district"] == name).to_numpy()
        pts = gpd.GeoSeries(gpd.points_from_xy(S.loc[m, "lon"], S.loc[m, "lat"]), crs=4326).to_crs(Gp.graph["crs"])
        xy = np.c_[pts.x, pts.y]
        spots_xy[name] = xy
        node_xy = {n: (d["x"], d["y"]) for n, d in Gp.nodes(data=True)}
        r = snap_to_segments(seg, A, xy, node_xy, max_dist=SNAP_MAX_M, corner_m=CORNER_M)
        sn = pd.DataFrame({"district": name, "spot_id": S.loc[m, "spot_id"].to_numpy(),
                           "segment": [seg["seg"].iloc[i] if i >= 0 else "" for i in r.seg_idx],
                           "seg_idx": r.seg_idx, "snap_dist_m": r.dist_m.round(2),
                           "second_nonadjacent_dist_m": r.second_dist_m.round(2), "ambiguous": r.ambiguous,
                           "near_intersection": r.near_intersection,
                           "inside_polygon": [poly.covers(p) for p in pts]})
        snaps.append(sn)
        # vocabulario BoVW ajustado con el OTRO distrito (holdout de distrito)
        other = VOCAB_PAIR[name]
        Vo = V[V["district"] == other]
        bv = BoVW(K_BOVW, seed=0).fit(E[Vo["row"].to_numpy()])
        Vd = V[V["district"] == name]
        words = bv.assign(E[Vd["row"].to_numpy()])
        local = {s: i for i, s in enumerate(np.flatnonzero(m))}
        Hspot = histograms(words, Vd["sidx"].map(local).to_numpy(), K_BOVW, int(m.sum()), norm="none")
        Zs = Zspot[m]
        for label, keep in (("all_spots", np.ones(m.sum(), bool)), ("no_corner_spots", ~r.near_intersection)):
            ss = np.where(keep, r.seg_idx, -1)
            rr, dd, info = homogeneity(name, seg, A, ss, Zs, Hspot, rng, label)
            rows += rr
            decs.append(dd)
            infos[f"{name}|{label}"] = info
            log(f"  {name} {label}: {info}")
        # focal: segmento con features cuyo ego-2 tiene tamaño mediano (determinista)
        F = np.unique(r.seg_idx[r.seg_idx >= 0])
        R2 = np.asarray(reach_matrices(A, F, 2)[2].sum(axis=1)).ravel()
        focal[name] = int(F[np.argsort(np.abs(R2 - np.median(R2)), kind="stable")[0]])
    snap = pd.concat(snaps, ignore_index=True)
    snap.to_csv(OUT / "spot_to_segment.csv", index=False)
    snap_qa = snap.groupby("district").agg(n_spots=("spot_id", "size"),
                                           pct_snapped=("seg_idx", lambda s: 100 * (s >= 0).mean()),
                                           snap_dist_median_m=("snap_dist_m", lambda s: s[np.isfinite(s)].median()),
                                           pct_ambiguous=("ambiguous", lambda s: 100 * s.mean()),
                                           pct_near_intersection=("near_intersection", lambda s: 100 * s.mean()),
                                           pct_inside_polygon=("inside_polygon", lambda s: 100 * s.mean())).round(2)
    snap_qa.to_csv(OUT / "snap_qa.csv")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "homogeneity.csv", index=False, float_format="%.5f")
    dec = pd.concat(decs, ignore_index=True)
    dec.to_csv(OUT / "decay.csv", index=False, float_format="%.5f")

    planned = {}
    for name, folder in (("san_isidro", "StreetSample-San_Isidro-s50"), ("lince", "StreetSample-Lince-s50")):
        if name in measured:
            continue
        p = REPO_ROOT / "image-extraction" / "data" / folder / "_manifest" / "spots_plan.csv"
        if p.exists():
            P = pd.read_csv(p)
            g = gpd.GeoSeries(gpd.points_from_xy(P["lon_req"], P["lat_req"]), crs=4326).to_crs(graphs[name][0].graph["crs"])
            planned[name] = np.c_[g.x, g.y]
        mid = segment_midpoints(graphs[name][1])
        c = np.array(graphs[name][3].centroid.coords[0])
        focal[name] = int(np.argmin(np.linalg.norm(mid - c, axis=1)))
    plot_maps(graphs, focal, spots_xy, planned)
    plot_decay(dec)

    crit = {}
    for name in measured:
        r = res[(res.district == name) & (res.features == "all_spots") & (res.metric == "cos")].set_index("comparison")
        crit[name] = {"network_more_homogeneous_than_euclid_disjoint":
                      bool(r.loc["euclid_disjoint - network_ring1", "ci_low"] > 0),
                      "sanity_network_better_than_random": bool(r.loc["random - network_ring1", "ci_low"] > 0)}
    summ = {"graph_qa": qa, "snap_qa": snap_qa.reset_index().to_dict("records"), "homogeneity_info": infos,
            "criteria_cos": crit, "seconds": round(time.perf_counter() - t0), "environment": environment_info()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    log(json.dumps(crit))
    log(f"listo en {summ['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
