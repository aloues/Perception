"""
Diagnosticos espaciales previos a cualquier GNN (target = CRIMEN OBSERVADO, no percepcion).

Preguntas / decisiones que dependen de cada una
  D1  Moran's I del riesgo: hay autocorrelacion espacial (=> hay que validar por bloques y los IC
      por bootstrap de nodos son optimistas)?
  D2  Sensibilidad del AUC ego-1 al numero/tamano de bloques espaciales de validacion.
  D3  Nulo espacial: permutar features DENTRO de cada bloque (mantiene el nivel de bloque,
      rompe la alineacion nodo-feature): cuanto del AUC es solo estructura a escala de bloque?
  D4  Vecindario por DISTANCIA (radio en metros) vs por SALTOS (ego-k): el efecto ego-1 es
      topologico o simplemente de escala espacial?
  D5  Agregacion simple (media) vs mas rica (media+std, MLP): hace falta algo mas que agregar
      para justificar una GNN?
Todos con las mismas etiquetas, nodos comunes, logistic + StandardScaler dentro del fold, seed 42.
Salida: outputs/ego/spatial_diagnostics.{json,md}, spatial_diag_*.csv
"""
from __future__ import annotations

import json
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_ego_tensors as B
from run_ego_experiment import DISTRICTS, EGO, GRAPH_DIR, SEED, auc, auc_ci, structural_features
from utils import log, set_seed

warnings.filterwarnings("ignore")
N_PERM = 100


def lr():
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])


def oof(X, y, groups, model=None, n_splits=5):
    return cross_val_predict(model or lr(), X, y, cv=GroupKFold(min(n_splits, len(np.unique(groups)))),
                             groups=groups, method="predict_proba")[:, 1]


def morans_i(G, nodes, x, n_perm=999, rng=None):
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for u, v in G.subgraph(nodes).edges():
        W[idx[u], idx[v]] = W[idx[v], idx[u]] = 1
    has = W.sum(1) > 0
    W = W[np.ix_(has, has)]; x = np.asarray(x, float)[has]
    W = W / W.sum(1, keepdims=True)
    def I(v):
        z = v - v.mean(); return float(len(v) / W.sum() * (z @ W @ z) / (z @ z))
    obs = I(x); null = np.array([I(rng.permutation(x)) for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1)), int(has.sum())


def main():
    set_seed(SEED); rng = np.random.default_rng(SEED)
    E, meta = B.load_embeddings()
    res = {"D1": [], "D2": [], "D3": [], "D4": [], "D5": []}
    for slug in DISTRICTS:
        dname = slug.replace("_", " ")
        log(f"\n=== {dname} ===")
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv"); nodes["node_id"] = nodes["node_id"].astype(str)
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        G = nx.Graph(); G.add_nodes_from(nodes["node_id"]); G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))
        common = json.loads((EGO / f"{slug}_2016_common_nodes.json").read_text())
        nd = nodes.set_index("node_id"); y = (nd.loc[common, "label"] == "peligroso").astype(int).to_numpy()
        xy = nd.loc[common, ["lon", "lat"]].to_numpy()

        def tensor(k):
            ids = json.loads((EGO / f"{slug}_2016_ego{k}_nodes.json").read_text()); pos = {n: i for i, n in enumerate(ids)}
            return np.load(EGO / f"{slug}_2016_ego{k}_tensor.npy")[[pos[n] for n in common]]
        T1 = tensor(1); vis1 = np.hstack([T1.mean(1), T1.std(1)])
        groups5 = KMeans(5, n_init=10, random_state=SEED).fit_predict(xy)

        # D1
        for name, v in (("label_peligroso", nd.loc[common, "label"].eq("peligroso").astype(int).to_numpy()),
                        ("neigh_total", nd.loc[common, "neigh_total"].to_numpy())):
            I, p, n = morans_i(G, common, v, rng=rng)
            res["D1"].append(dict(district=dname, variable=name, morans_I=round(I, 3), p_perm=round(p, 4), n_nodes_with_neighbours=n,
                                  weights="adyacencia del grafo (fila-estandarizada), subgrafo de nodos comunes"))
            log(f"  D1 Moran {name}: I={I:.3f} p={p:.3f}")
        # D2
        for nb in (3, 5, 8, 10):
            g = KMeans(nb, n_init=10, random_state=SEED).fit_predict(xy)
            sizes = np.bincount(g)
            p = oof(vis1, y, g); a = auc(y, p); lo, hi = auc_ci(y, p, n_boot=500)
            res["D2"].append(dict(district=dname, n_blocks=nb, min_block=int(sizes.min()), max_block=int(sizes.max()),
                                  auc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            log(f"  D2 blocks={nb}: AUC={a:.3f}")
        # D3
        a_obs = auc(y, oof(vis1, y, groups5)); null = []
        for _ in range(N_PERM):
            Xp = vis1.copy()
            for b in np.unique(groups5):
                ix = np.where(groups5 == b)[0]; Xp[ix] = vis1[rng.permutation(ix)]
            null.append(auc(y, oof(Xp, y, groups5)))
        null = np.array(null)
        res["D3"].append(dict(district=dname, auc_observed=round(a_obs, 4), null_mean=round(float(null.mean()), 4),
                              null_p95=round(float(np.percentile(null, 95)), 4),
                              p_empirical=round(float((np.sum(null >= a_obs) + 1) / (N_PERM + 1)), 4), n_perm=N_PERM))
        log(f"  D3 nulo intra-bloque: obs={a_obs:.3f} null_mean={null.mean():.3f} p95={np.percentile(null,95):.3f}")
        # D4 vecindario por distancia
        lat0 = float(nodes["lat"].mean())
        m = lambda lon, lat: np.c_[np.asarray(lon) * 111320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110540]
        allxy = m(nodes["lon"], nodes["lat"]); tree = cKDTree(allxy)
        crimes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_crimes_snapped.csv"); crimes["CRIME_ID"] = crimes["CRIME_ID"].astype(str)
        crimes["nearest_node"] = crimes["nearest_node"].astype(str)
        sub = meta[meta["district"] == dname]
        emb_by_crime = {str(k): g_["row"].to_numpy() for k, g_ in sub.groupby("crime_id")}
        nid = nodes["node_id"].to_numpy(); pos_of = {n: i for i, n in enumerate(nid)}
        seg = np.median([np.linalg.norm(allxy[pos_of[u]] - allxy[pos_of[v]]) for u, v in G.edges() if u in pos_of and v in pos_of])
        log(f"  mediana longitud de arista: {seg:.0f} m")
        for R in (50, 100, 200, 400):
            rgen = np.random.default_rng(SEED + R); rows_ = []
            for n in common:
                near = set(nid[tree.query_ball_point(allxy[pos_of[n]], R)])
                cand = crimes[crimes["nearest_node"].isin(near) & crimes["CRIME_ID"].isin(emb_by_crime)]
                pts, _ = B.sample_points(cand, rgen)
                idx, _ = B.node_tensor(pts, emb_by_crime, rgen) if pts else (None, False)
                rows_.append(idx)
            ok = np.array([r is not None for r in rows_])
            if ok.sum() < 30:
                res["D4"].append(dict(district=dname, radius_m=R, n_nodes=int(ok.sum()), note="cobertura insuficiente")); continue
            Tr = np.stack([E[np.asarray(r, dtype=np.int64)] for r, k_ in zip(rows_, ok) if k_]).astype(np.float32)
            Xr = np.hstack([Tr.mean(1), Tr.std(1)]); yy, gg = y[ok], groups5[ok]
            if len(np.unique(yy)) < 2: continue
            p = oof(Xr, yy, gg); a = auc(yy, p); lo, hi = auc_ci(yy, p, n_boot=500)
            res["D4"].append(dict(district=dname, radius_m=R, n_nodes=int(ok.sum()), auc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                  median_edge_m=round(float(seg), 1)))
            log(f"  D4 radio {R} m: n={ok.sum()} AUC={a:.3f}")
        # D5
        variants = {"mean": T1.mean(1), "std": T1.std(1), "mean+std": vis1}
        P = {}
        for nme, X in variants.items():
            P[nme] = oof(X, y, groups5)
        P["MLP(mean+std)"] = oof(vis1, y, groups5, model=Pipeline([("sc", StandardScaler()),
                              ("mlp", MLPClassifier((64,), alpha=1e-1, max_iter=500, random_state=SEED))]))
        for nme, p in P.items():
            a = auc(y, p); lo, hi = auc_ci(y, p, n_boot=500)
            res["D5"].append(dict(district=dname, aggregation=nme, auc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            log(f"  D5 {nme}: AUC={a:.3f}")

    for k, v in res.items():
        pd.DataFrame(v).to_csv(EGO / f"spatial_diag_{k}.csv", index=False)
    (EGO / "spatial_diagnostics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Diagnosticos espaciales (target = crimen observado)", ""]
    titles = {"D1": "D1 Moran's I", "D2": "D2 Sensibilidad al numero de bloques (ego-1 visual)", "D3": "D3 Nulo espacial intra-bloque (ego-1 visual)",
              "D4": "D4 Vecindario por distancia (visual media+std)", "D5": "D5 Agregacion (ego-1)"}
    for k, v in res.items():
        d = pd.DataFrame(v); L += [f"## {titles[k]}", "", d.to_csv(index=False), ""]
    (EGO / "spatial_diagnostics.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
