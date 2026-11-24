"""
EXP-030 — Barrios delimitados por la red vial (comunidades de Louvain) vs celdas cuadradas de igual número.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa EXP-028..031, 2026-10-09,
con enmienda del bootstrap antes de ejecutar).

EXP-026 mostró que a 150–400 m la red y el radio eligen las mismas imágenes. A escala de BARRIO (~1 km²) las
barreras (vías expresas, ríos sin puentes, parques) sí pueden partir la ciudad de forma distinta a una
cuadrícula. Pregunta: ¿las unidades que delimita la conectividad de la red agrupan lugares que la gente
percibe de forma más parecida (mayor fracción de varianza entre unidades, ICC(1)) que una cuadrícula?

Uso:  python exp030_network_neighborhoods.py        Salidas: outputs/exp030/
"""

from __future__ import annotations

import json
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from exp026_additive_context import load_images
from network_distance import BARRIER_HIGHWAYS, drop_highways, to_weighted_simple
from probe_utils import load_pairs, load_split, train_scores, votes
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp030"
GRAPHS = OUTPUTS / "street_graph_pilot" / "graphs"
PILOT = ["Atlanta", "Berlin", "Santiago"]
CELL = 1000.0
OFFSETS = [0.0, 250.0, 500.0, 750.0]
BLOCK = 3000.0
N_BOOT = 1000
MIN_VOTES = 3


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1) de un ANOVA de un factor; solo grupos con >= 2 observaciones."""
    df = pd.DataFrame({"x": values, "g": groups})
    n_g = df.groupby("g")["x"].transform("size")
    df = df[n_g >= 2]
    if df["g"].nunique() < 3:
        return np.nan
    grand = df["x"].mean()
    st = df.groupby("g")["x"].agg(["size", "mean"])
    k, N = len(st), len(df)
    msb = (st["size"] * (st["mean"] - grand) ** 2).sum() / (k - 1)
    msw = ((df["x"] - df["g"].map(st["mean"])) ** 2).sum() / (N - k)
    n0 = (N - (st["size"] ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def communities(S: nx.Graph, node_of_img: np.ndarray, target: int) -> tuple[dict, float, int]:
    """Louvain con resolución ajustada (bisección en log) para ~target comunidades ocupadas por imágenes."""
    H = S.copy()
    for u, v, d in H.edges(data=True):
        d["w"] = 1.0 / max(d["length"], 1.0)
    lo, hi = np.log(1e-3), np.log(1e3)
    best = None
    for _ in range(12):
        r = float(np.exp((lo + hi) / 2))
        comms = nx.community.louvain_communities(H, weight="w", resolution=r, seed=SEED)
        comm_of = {n: i for i, c in enumerate(comms) for n in c}
        occ = len({comm_of[n] for n in node_of_img})
        if best is None or abs(occ - target) < abs(best[2] - target):
            best = (comm_of, r, occ)
        if abs(occ - target) <= 0.05 * target:
            break
        lo, hi = (np.log(r), hi) if occ < target else (lo, np.log(r))
    return best


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].tolist()
    pairs = load_pairs()
    sp = load_split("image_disjoint")
    s, _ = train_scores(Z, ids, pairs, sp, "test")
    model = pd.Series(s, index=ids)
    vt = votes(pairs, sp["train"]).groupby("id")["win"].agg(["size", "mean"])
    human = vt.loc[vt["size"] >= MIN_VOTES, "mean"]

    rows, info = [], []
    for city in PILOT:
        g = man[man["city"] == city].reset_index(drop=True)
        Gp = ox.project_graph(ox.load_graphml(GRAPHS / f"{city}_drive.graphml"))
        pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").to_crs(Gp.graph["crs"])
        X, Y = pts.x.to_numpy(), pts.y.to_numpy()
        grids = [(np.floor((X + o) / CELL).astype(int).astype(str) + "_"
                  + np.floor((Y + o) / CELL).astype(int).astype(str)) for o in OFFSETS]
        target = int(np.mean([len(set(gr)) for gr in grids]))
        parts = {f"grid_{int(o)}": gr for o, gr in zip(OFFSETS, grids)}
        for tag, G in (("NET", Gp), ("NETB", drop_highways(Gp, BARRIER_HIGHWAYS))):
            edges = ox.distance.nearest_edges(G, X, Y)
            node_of_img = np.array([u for u, v, k in edges])
            comm_of, res, occ = communities(to_weighted_simple(G), node_of_img, target)
            parts[f"comm_{tag}"] = np.array([str(comm_of[n]) for n in node_of_img])
            info.append(dict(city=city, partition=f"comm_{tag}", resolution=res, occupied_units=occ,
                             target_units=target))
            log(f"  {city} {tag}: resolución {res:.3g}, {occ} comunidades ocupadas (objetivo {target}) "
                f"({time.perf_counter() - t0:.0f}s)")
        for o, gr in zip(OFFSETS, grids):
            info.append(dict(city=city, partition=f"grid_{int(o)}", resolution=np.nan,
                             occupied_units=len(set(gr)), target_units=target))

        lid = g["location_id"].to_numpy()
        blocks = (np.floor(X / BLOCK).astype(int).astype(str) + "_" + np.floor(Y / BLOCK).astype(int).astype(str))
        ub = np.unique(blocks)
        members = {b: np.flatnonzero(blocks == b) for b in ub}
        for target_name, val in (("humano", human), ("modelo", model)):
            v = pd.Series(lid).map(val).to_numpy(dtype=float)
            ok = ~np.isnan(v)

            def stat(idx_list):
                idx = np.concatenate([m for m, _ in idx_list])
                copy = np.concatenate([np.full(len(m), c) for m, c in idx_list])
                keep = ok[idx]
                out = {}
                for name, part in parts.items():
                    grp = np.char.add(copy[keep].astype(str), np.char.add("|", part[idx][keep].astype(str)))
                    out[name] = icc1(v[idx][keep], grp)
                out["grid"] = np.nanmean([out[f"grid_{int(o)}"] for o in OFFSETS])
                return out

            full = stat([(members[b], 0) for b in ub])
            boots = []
            for _ in range(N_BOOT):
                draw = rng.choice(ub, len(ub))
                boots.append(stat([(members[b], c) for c, b in enumerate(draw)]))
            B = pd.DataFrame(boots)
            for comm in ("comm_NET", "comm_NETB"):
                d = B[comm] - B["grid"]
                lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
                rows.append(dict(city=city, target=target_name, n_images=int(ok.sum()),
                                 partition=comm, icc_comm=full[comm], icc_grid=full["grid"],
                                 delta=full[comm] - full["grid"], ci95_low=lo_, ci95_high=hi_,
                                 ci_excludes_0=bool(lo_ > 0 or hi_ < 0)))
                log(f"  {city:<9} {target_name:<7} {comm:<10} ICC comm={full[comm]:.3f} grid={full['grid']:.3f} "
                    f"Δ={full[comm] - full['grid']:+.3f} [{lo_:+.3f},{hi_:+.3f}]")

    pd.DataFrame(rows).to_csv(OUT / "homogeneity_icc.csv", index=False, float_format="%.4f")
    pd.DataFrame(info).to_csv(OUT / "partitions.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps({
        "seed": SEED, "cell_m": CELL, "offsets_m": OFFSETS, "block_m": BLOCK, "n_boot": N_BOOT,
        "min_votes": MIN_VOTES, "louvain_weight": "1/length", "seconds": round(time.perf_counter() - t0, 1)},
        indent=2), encoding="utf-8")
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
