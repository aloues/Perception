"""
EXP-026 (Parte 2) — ¿el error del modelo de imagen aislada tiene estructura espacial, y la captura
mejor un vecindario por distancia de RED (snap a arista) que uno euclídeo?

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-026, 2026-10-09).

POR QUE ESTA MÉTRICA
--------------------
El contexto espacial solo puede aportar sobre la imagen aislada si el ERROR del modelo de imagen
tiene estructura espacial: si dos lugares cercanos comparten lo que el modelo no ve. Es la
condición previa estándar de cualquier modelo espacial (rezago espacial, GNN). La homogeneidad del
puntaje crudo (R2 de EXP-025) no basta: dos vecinas pueden parecerse en puntaje solo porque se
parecen visualmente, y eso el modelo de imagen ya lo captura.

    s_i  = puntaje del probe lineal ego-0 (DINOv2, entrenado en train de image_disjoint, 56 ciudades)
    r_i  = (victorias_i − Σ_rivales σ(s_i − s_rival)) / n_i       sobre pares de TRAIN, n_i ≥ 3
    ρ_X  = Spearman(r_i, media de r_j de las vecinas X(i) con residuo)

Vecindarios (3 ciudades piloto, grafos drive de EXP-025):
    NET_R    distancia por red ≤ R (snap a arista)            R ∈ {250, 400} m
    NETB_R   idem sin motorway/trunk (vía expresa = barrera)
    EUC_R    euclídeo ≤ R
    EUCM_*   euclídeo con el mismo nº de vecinas que NET_R / NETB_R
    RAND     aleatorias de la misma ciudad, mismo nº que NET_R
    HOP3 / HOP3_EUCM   vecinas de EXP-025 (≤ 3 saltos, snap a nodo) y su control
56 ciudades: EUC250 / EUC400 vs RAND (count-matched) con la proyección de EXP-021.
Tope 20 vecinas más cercanas (por la distancia de cada definición).

Uso:  python exp026_network_residuals.py
Salidas: outputs/exp026/  (residual_autocorr_*.csv, network_qa.csv, network_coverage.csv, ...)
"""

from __future__ import annotations

import json
import re
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from exp026_additive_context import load_images, neighbor_sets
from network_distance import (BARRIER_HIGHWAYS, NetworkDistance, drop_highways, snap_points,
                              to_weighted_simple)
from ranking_head import build_head
from train_probe import predict, run_epoch, to_tensors
from evaluate import evaluate_all
from utils import MANIFESTS, OUTPUTS, SPLITS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp026"
GRAPHS = OUTPUTS / "street_graph_pilot" / "graphs"
PILOT = ["Atlanta", "Berlin", "Santiago"]
RADII = [250.0, 400.0]
NEIGH_MAX = 20
MIN_VOTES = 3
N_BOOT = 1000
DUP_M = 5.0          # vecinas a < 5 m = misma toma / misma coordenada (robustez)


# ── 1. probe ego-0 y residuos ────────────────────────────────────────────────────────────────
def ego0_scores(man: pd.DataFrame, Z: np.ndarray, pairs: pd.DataFrame, split: dict) -> pd.Series:
    """Mismo probe y mismos hiperparámetros que train_probe.py (EXP-001), entrenado en proceso
    para poder puntuar cada imagen. Sanity: acc de test ≈ 0.6785 (EXP-001)."""
    set_seed(SEED)
    id_to_row = {lid: i for i, lid in enumerate(man["location_id"])}
    zl, zr, y, _ = to_tensors(pairs, split["train"], Z, id_to_row, "cpu")
    vl, vr, vy, _ = to_tensors(pairs, split["val"], Z, id_to_row, "cpu")
    head = build_head("linear", embed_dim=Z.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for _ in range(50):
        run_epoch(head, zl, zr, y, opt, 4096)
        v = run_epoch(head, vl, vr, vy, None, 4096)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    head.load_state_dict(state)
    tl, tr, ty, _ = to_tensors(pairs, split["test"], Z, id_to_row, "cpu")
    acc = evaluate_all(predict(head, tl, tr), ty.numpy())
    log(f"probe ego-0: test acc {acc['pairwise_accuracy']:.4f} "
        f"[{acc['ci95_low']:.4f}, {acc['ci95_high']:.4f}] (EXP-001: 0.6785)")
    (OUT / "ego0_probe_sanity.json").write_text(json.dumps(acc, indent=2), encoding="utf-8")
    s = head.score_images(torch.from_numpy(Z)).numpy()
    return pd.Series(s, index=man["location_id"].to_numpy())


def residuals(s: pd.Series, pairs: pd.DataFrame, split: dict) -> pd.DataFrame:
    tr = pairs.iloc[split["train"]]
    sl, sr = s.reindex(tr["left"]).to_numpy(), s.reindex(tr["right"]).to_numpy()
    p = 1 / (1 + np.exp(-(sl - sr)))
    y = tr["label"].astype(int).to_numpy()
    df = pd.DataFrame({"id": np.r_[tr["left"], tr["right"]], "obs": np.r_[y, 1 - y],
                       "exp": np.r_[p, 1 - p]})
    g = df.groupby("id").agg(n=("obs", "size"), obs=("obs", "mean"), exp=("exp", "mean"))
    g = g[g["n"] >= MIN_VOTES]
    g["resid"] = g["obs"] - g["exp"]
    log(f"residuos: {len(g):,} imágenes de train con >= {MIN_VOTES} votos "
        f"(media {g['resid'].mean():+.4f}, sd {g['resid'].std():.4f})")
    return g


# ── 2. autocorrelación ───────────────────────────────────────────────────────────────────────
def neigh_mean(lists: list, val: pd.Series, ids: np.ndarray) -> np.ndarray:
    """lists[i] = índices (en ids) de vecinas; devuelve la media de val sobre las que tienen valor."""
    out = np.full(len(lists), np.nan)
    for i, L in enumerate(lists):
        v = val.reindex(ids[np.asarray(L, dtype=int)]).dropna() if len(L) else []
        if len(v):
            out[i] = float(np.mean(v))
    return out


def compare(r: np.ndarray, mx: np.ndarray, my: np.ndarray, rng) -> dict:
    ok = ~np.isnan(r) & ~np.isnan(mx) & ~np.isnan(my)
    a, x, y = r[ok], mx[ok], my[ok]
    rx, ry = spearmanr(a, x)[0], spearmanr(a, y)[0]
    d = np.empty(N_BOOT)
    bx = np.empty(N_BOOT)
    for t in range(N_BOOT):
        ix = rng.integers(0, len(a), len(a))
        bx[t] = spearmanr(a[ix], x[ix])[0]
        d[t] = bx[t] - spearmanr(a[ix], y[ix])[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(n_images=int(ok.sum()), rho_x=rx, rho_x_ci_low=np.percentile(bx, 2.5),
                rho_x_ci_high=np.percentile(bx, 97.5), rho_y=ry, delta=rx - ry,
                ci95_low=lo, ci95_high=hi, ci_excludes_0=bool(lo > 0 or hi < 0))


# ── 3. vecindarios por red en las ciudades piloto ────────────────────────────────────────────
def pilot_city(city: str, g: pd.DataFrame, rng) -> tuple[dict, list, list]:
    """Devuelve {variante: listas de vecinas (índices locales)} + filas de QA y de cobertura."""
    t0 = time.perf_counter()
    Gp = ox.project_graph(ox.load_graphml(GRAPHS / f"{city}_drive.graphml"))
    pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").to_crs(Gp.graph["crs"])
    X, Y = pts.x.to_numpy(), pts.y.to_numpy()
    xy = np.c_[X, Y]
    tree = cKDTree(xy)
    n = len(g)
    variants, qa = {}, []

    for tag, G in (("NET", Gp), ("NETB", drop_highways(Gp, BARRIER_HIGHWAYS))):
        edges, dist = ox.distance.nearest_edges(G, X, Y, return_dist=True)
        snaps = snap_points(G, list(edges), X, Y, np.asarray(dist, float))
        S = to_weighted_simple(G)
        comps = sorted((len(c) for c in nx.connected_components(S)), reverse=True)
        d = np.array([s.dist for s in snaps])
        on_barrier = np.array([bool(set(s.highway.split("|")) & BARRIER_HIGHWAYS) for s in snaps])
        qa.append(dict(city=city, graph=tag, nodes=S.number_of_nodes(), edges=S.number_of_edges(),
                       components=len(comps), largest_component_share=comps[0] / S.number_of_nodes(),
                       snap_edge_median_m=np.median(d), snap_edge_p90_m=np.percentile(d, 90),
                       pct_snap_le_25m=100 * (d <= 25).mean(),
                       pct_images_on_motorway_trunk=100 * on_barrier.mean()))
        nd = NetworkDistance(S, snaps, cutoff=max(RADII))
        for R in RADII:
            variants[f"{tag}{int(R)}"] = [None] * n
        for i in range(n):
            Di = nd.from_point(i)
            cand = [j for j in tree.query_ball_point(xy[i], max(RADII) + 50.0) if j != i]
            dd = np.array([nd.between(i, j, Di) for j in cand]) if cand else np.array([])
            for R in RADII:
                keep = [(dd[k], cand[k]) for k in range(len(cand)) if dd[k] <= R]
                keep.sort()
                variants[f"{tag}{int(R)}"][i] = [j for _, j in keep[:NEIGH_MAX]]
        log(f"  {city} {tag}: snap a arista mediana {np.median(d):.1f} m, "
            f"{100 * on_barrier.mean():.1f}% en motorway/trunk ({time.perf_counter() - t0:.0f}s)")

    for R in RADII:
        e = []
        for i in range(n):
            nb = [j for j in tree.query_ball_point(xy[i], R) if j != i]
            nb.sort(key=lambda j: np.hypot(*(xy[j] - xy[i])))
            e.append(nb[:NEIGH_MAX])
        variants[f"EUC{int(R)}"] = e
        for tag in ("NET", "NETB"):
            m = []
            for i in range(n):
                k = len(variants[f"{tag}{int(R)}"][i])
                if k == 0:
                    m.append([])
                    continue
                _, nn = tree.query(xy[i], k=min(k + 1, n))
                m.append([j for j in np.atleast_1d(nn) if j != i][:k])
            variants[f"EUCM_{tag}{int(R)}"] = m
        variants[f"RAND{int(R)}"] = [
            list(rng.choice(np.setdiff1d(np.arange(n), [i]), size=len(L), replace=False)) if L else []
            for i, L in enumerate(variants[f"NET{int(R)}"])]

    # duplicados: misma toma (< DUP_M m) — robustez
    variants["_xy"] = xy
    cov = [dict(city=city, variant=v, mean_neighbors=np.mean([len(L) for L in Ls]),
                pct_zero=100 * np.mean([len(L) == 0 for L in Ls]))
           for v, Ls in variants.items() if not v.startswith("_")]
    return variants, qa, cov


def drop_dups(lists: list, xy: np.ndarray) -> list:
    return [[j for j in L if np.hypot(*(xy[j] - xy[i])) >= DUP_M] for i, L in enumerate(lists)]


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    pairs["label"] = pairs["label"].astype(int)
    split = json.loads((SPLITS / "image_disjoint.json").read_text())

    s = ego0_scores(man, Z, pairs, split)
    res = residuals(s, pairs, split)
    res.to_csv(OUT / "ego0_train_residuals.csv")
    resid, winrate = res["resid"], res["obs"]
    ids = man["location_id"].to_numpy()

    # ── 56 ciudades: euclídeo vs aleatorio de la misma ciudad ────────────────────────────────
    rows = []
    q = man["location_id"].isin(resid.index).to_numpy()
    r_all = resid.reindex(ids).to_numpy()
    lat0 = man.groupby("city")["lat"].transform("mean")
    xy_all = np.c_[man["lon"] * 111320 * np.cos(np.radians(lat0)), man["lat"] * 110540]
    for R in RADII:
        N, Rn = neighbor_sets(man, R, np.random.default_rng(SEED))
        mN = neigh_mean([N[i] for i in np.flatnonzero(q)], resid, ids)
        mR = neigh_mean([Rn[i] for i in np.flatnonzero(q)], resid, ids)
        Nd = [[j for j in N[i] if np.hypot(*(xy_all[j] - xy_all[i])) >= DUP_M] for i in np.flatnonzero(q)]
        mNd = neigh_mean(Nd, resid, ids)
        for name, mx, my in ((f"EUC{int(R)} - RAND{int(R)}", mN, mR),
                             (f"EUC{int(R)}_sin_dup - RAND{int(R)}", mNd, mR)):
            c = compare(r_all[q], mx, my, rng)
            rows.append(dict(scope="56 ciudades", comparison=name, target="residuo", **c))
            log(f"  56c {name:<26} ρ_x={c['rho_x']:+.3f} ρ_y={c['rho_y']:+.3f} Δ={c['delta']:+.3f} "
                f"[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}] n={c['n_images']}")
    pd.DataFrame(rows).to_csv(OUT / "residual_autocorr_56cities.csv", index=False, float_format="%.4f")

    # ── 3 ciudades piloto: red vs euclídeo ───────────────────────────────────────────────────
    nb025 = pd.read_json(OUTPUTS / "street_graph_pilot" / "neighbors.jsonl", lines=True)
    rows, qa_rows, cov_rows = [], [], []
    for city in PILOT:
        g = man[man["city"] == city].reset_index(drop=True)
        cid = g["location_id"].to_numpy()
        V, qa, cov = pilot_city(city, g, rng)
        qa_rows += qa
        cov_rows += cov
        pos = {lid: k for k, lid in enumerate(cid)}
        h = nb025[nb025["city"] == city].set_index("location_id")
        V["HOP3"] = [[pos[x] for x in h.at[lid, "graph"] if x in pos] if lid in h.index else [] for lid in cid]
        V["HOP3_EUCM"] = [[pos[x] for x in h.at[lid, "euc_match"] if x in pos] if lid in h.index else []
                          for lid in cid]
        xy = V.pop("_xy")
        qmask = np.isin(cid, resid.index)
        qi = np.flatnonzero(qmask)
        for target, val in (("residuo", resid), ("puntaje_crudo", winrate)):
            rv = val.reindex(cid).to_numpy()[qi]
            M = {k: neigh_mean([L[i] for i in qi], val, cid) for k, L in V.items()}
            Md = {k: neigh_mean([L[i] for i in qi], val, cid)
                  for k, L in ((k, drop_dups(L, xy)) for k, L in V.items())}
            comps = []
            for R in (250, 400):
                comps += [(f"NET{R}", f"EUCM_NET{R}"), (f"NETB{R}", f"EUCM_NETB{R}"),
                          (f"NETB{R}", f"NET{R}"), (f"EUC{R}", f"RAND{R}"), (f"NET{R}", f"RAND{R}")]
            comps += [("HOP3", "HOP3_EUCM")]
            for x, y in comps:
                for dup_tag, MM in (("", M), ("_sin_dup", Md)):
                    if target == "puntaje_crudo" and dup_tag:
                        continue
                    c = compare(rv, MM[x], MM[y], rng)
                    rows.append(dict(scope=city, comparison=f"{x}{dup_tag} - {y}{dup_tag}", target=target, **c))
                    if target == "residuo" and not dup_tag:
                        log(f"  {city:<9} {x:>6} - {y:<11} ρ_x={c['rho_x']:+.3f} ρ_y={c['rho_y']:+.3f} "
                            f"Δ={c['delta']:+.3f} [{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}] n={c['n_images']}")
    pd.DataFrame(rows).to_csv(OUT / "residual_autocorr_pilot.csv", index=False, float_format="%.4f")
    pd.DataFrame(qa_rows).to_csv(OUT / "network_qa.csv", index=False, float_format="%.3f")
    pd.DataFrame(cov_rows).to_csv(OUT / "network_coverage.csv", index=False, float_format="%.3f")
    (OUT / "part2_summary.json").write_text(json.dumps({
        "seed": SEED, "radii_m": RADII, "neigh_max": NEIGH_MAX, "min_votes": MIN_VOTES,
        "n_boot": N_BOOT, "dup_threshold_m": DUP_M, "pilot_cities": PILOT,
        "barrier_highways": sorted(BARRIER_HIGHWAYS), "osmnx": ox.__version__,
        "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
