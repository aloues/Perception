"""
FASES E + F — Visual-only vs Structural-only vs Fused, a profundidad ego-0..3.

Hipotesis
---------
H1  La representacion visual de un nodo aislado (ego-0) aporta poca senal de
    crime-risk por si sola; el contexto del vecindario la mejora.
H2  Existe una profundidad optima: a partir de cierto k el contexto deja de
    aportar y empieza a meter ruido (nodos lejanos irrelevantes).
H3  La senal estructural (topologia de calles) es mas robusta al split
    espacial que la visual (ya observado en el baseline heredado).

Diseno
------
- Features VISUAL   : mean ++ std sobre los 20 slots del tensor = 1,536-d,
                      exactamente como el `visual_only` del pipeline heredado.
- Features STRUCTURAL: degree, betweenness, closeness, clustering,
                      avg_neighbor_degree = 5-d, identicas a las del heredado.
                      Son puramente topologicas: NO usan crime_count ni
                      neigh_total, que serian circulares respecto a la etiqueta.
- FUSED             : concatenacion = 1,541-d.
- Clasificador      : regresion logistica; el escalado se ajusta SOLO con train
                      dentro de cada fold (Pipeline de sklearn).
- Validacion        : GroupKFold sobre 5 bloques espaciales (KMeans lat/lon)
                      como metrica PRIMARIA; StratifiedKFold aleatorio solo
                      como referencia de cuanto infla el leakage espacial.
- Metrica           : AUC sobre predicciones out-of-fold, con IC 95% bootstrap.

Comparacion justa entre profundidades
-------------------------------------
La cobertura depende de k (en ego-0 muchos nodos no tienen punto de crimen
propio). Para que la comparacion mida CONTEXTO y no COBERTURA, todas las
profundidades se evaluan sobre el mismo conjunto de nodos: los que tienen
cobertura en las cuatro profundidades (`common_nodes`). Se reporta ademas la
cobertura por profundidad, que es en si un resultado.

Salidas:
    outputs/ego/ego_results.csv
    outputs/ego/ego_results.md
    outputs/ego/ego_results.json

Uso:
    python run_ego_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import OUTPUTS, REPO_ROOT, log, set_seed

GRAPH_DIR = REPO_ROOT / "graph" / "graph_output_v2"
LIMA = REPO_ROOT / "external_assets" / "lima_graph"
EGO = OUTPUTS / "ego"
SEED = 42
N_BLOCKS = 5
DEPTHS = (0, 1, 2, 3)
DISTRICTS = ("Barranco", "La_Victoria")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC por el estadistico de Mann-Whitney (sin dependencias extra)."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # promediar rangos en empates
    s = pd.Series(p)
    ranks = s.rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(auc(y[idx], p[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def structural_features(G: nx.Graph, node_ids: list[str]) -> np.ndarray:
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, seed=SEED)
    clo = nx.closeness_centrality(G)
    clu = nx.clustering(G)
    andg = nx.average_neighbor_degree(G)
    return np.array([[deg.get(n, 0), btw.get(n, 0.0), clo.get(n, 0.0),
                      clu.get(n, 0.0), andg.get(n, 0.0)] for n in node_ids], float)


def spatial_blocks(nodes_df: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    d = nodes_df.set_index("node_id").loc[node_ids]
    xy = d[["lon", "lat"]].to_numpy()
    return KMeans(n_clusters=N_BLOCKS, n_init=10, random_state=SEED).fit_predict(xy)


def main() -> None:
    set_seed(SEED)
    rows = []

    for slug in DISTRICTS:
        log(f"\n=== {slug} ===")
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
        nodes["node_id"] = nodes["node_id"].astype(str)
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        G = nx.Graph()
        G.add_nodes_from(nodes["node_id"])
        G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))

        common = json.loads((EGO / f"{slug}_2016_common_nodes.json").read_text())
        # `label` es el neighbourhood_label del dual-GMM heredado:
        # 'peligroso' = high crime = 1, 'seguro' = low crime = 0.
        # Se verifica contra eigen_features/*_labels.npy del pipeline original.
        label_of = {n: int(v == "peligroso")
                    for n, v in zip(nodes["node_id"], nodes["label"])}
        y = np.array([label_of[n] for n in common])
        ref = LIMA / "eigen_features" / f"{slug}_2016_labels.npy"
        if ref.exists():
            n_pos_ref = int(np.load(ref).sum())
            n_pos_here = int(sum(label_of.values()))
            log(f"  etiquetas: {n_pos_here} peligrosos en nodes.csv vs "
                f"{n_pos_ref} en labels.npy del heredado")
        log(f"  nodos comunes a las 4 profundidades: {len(common)} "
            f"| positivos {int(y.sum())} ({y.mean():.1%})")

        struct_all = structural_features(G, common)
        groups = spatial_blocks(nodes, common)
        log(f"  bloques espaciales: {np.bincount(groups).tolist()}")

        for k in DEPTHS:
            ids_k = json.loads((EGO / f"{slug}_2016_ego{k}_nodes.json").read_text())
            T = np.load(EGO / f"{slug}_2016_ego{k}_tensor.npy")
            pos = {n: i for i, n in enumerate(ids_k)}
            sel = np.array([pos[n] for n in common])
            Tc = T[sel]                                  # (n_common, 20, 768)
            visual = np.hstack([Tc.mean(axis=1), Tc.std(axis=1)])   # 1536-d

            feats = {"visual_only": visual,
                     "structural_only": struct_all,
                     "fused": np.hstack([visual, struct_all])}

            for name, X in feats.items():
                # structural no depende de k: se evalua una sola vez (en k=0)
                if name == "structural_only" and k != 0:
                    continue
                for scheme in ("spatial", "random"):
                    clf = Pipeline([("sc", StandardScaler()),
                                    ("lr", LogisticRegression(max_iter=2000,
                                                              random_state=SEED))])
                    if scheme == "spatial":
                        cv, g = GroupKFold(n_splits=N_BLOCKS), groups
                    else:
                        cv, g = StratifiedKFold(n_splits=N_BLOCKS, shuffle=True,
                                                random_state=SEED), None
                    p = cross_val_predict(clf, X, y, cv=cv, groups=g,
                                          method="predict_proba")[:, 1]
                    a = auc(y, p)
                    lo, hi = auc_ci(y, p)
                    rows.append({
                        "district": slug.replace("_", " "),
                        "strategy": name,
                        "depth": "n/a" if name == "structural_only" else f"ego-{k}",
                        "cv_scheme": scheme,
                        "n_nodes": len(common), "n_features": X.shape[1],
                        "positives": int(y.sum()), "prevalence": round(float(y.mean()), 4),
                        "auc": round(a, 4), "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4), "seed": SEED,
                    })
                    log(f"  {name:<16} ego-{k} {scheme:<8} AUC={a:.4f} "
                        f"[{lo:.4f}, {hi:.4f}]")

    df = pd.DataFrame(rows)
    EGO.mkdir(parents=True, exist_ok=True)
    df.to_csv(EGO / "ego_results.csv", index=False)

    # cobertura por profundidad (resultado en si mismo)
    cov = json.loads((EGO / "ego_build_summary.json").read_text(encoding="utf-8"))

    L = ["# Fases E + F — contexto espacial en Lima (crime-risk)", "",
         "Metrica: AUC sobre predicciones out-of-fold, IC 95% bootstrap (2000 resamples).",
         "Esquema PRIMARIO: GroupKFold sobre 5 bloques espaciales. `random` se muestra",
         "solo como referencia de cuanto infla el leakage espacial.", "",
         "Todas las profundidades se evaluan sobre los MISMOS nodos (los que tienen",
         "cobertura en ego-0..3), para que la comparacion mida contexto y no cobertura.", ""]

    L += ["## Cobertura por profundidad (resultado)", "",
          "| Distrito | ego-0 | ego-1 | ego-2 | ego-3 | nodos del grafo | comunes |",
          "|---|---|---|---|---|---|---|"]
    for slug, d in cov["districts"].items():
        pd_ = d["per_depth"]
        L.append(f"| {slug.replace('_',' ')} | " +
                 " | ".join(f"{pd_[f'ego{k}']['n_nodes']}" for k in DEPTHS) +
                 f" | {d['n_graph_nodes']} | {d['n_common_nodes']} |")
    L.append("")

    for scheme in ("spatial", "random"):
        tag = ("PRIMARIA — GroupKFold espacial" if scheme == "spatial"
               else "referencia — k-fold aleatorio (leakage espacial)")
        L += [f"## {tag}", "",
              "| Distrito | Estrategia | Profundidad | n nodos | n feats | AUC | IC 95% |",
              "|---|---|---|---|---|---|---|"]
        for _, r in df[df.cv_scheme == scheme].iterrows():
            L.append(f"| {r.district} | `{r.strategy}` | {r.depth} | {r.n_nodes} | "
                     f"{r.n_features} | **{r.auc:.4f}** | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] |")
        L.append("")

    (EGO / "ego_results.md").write_text("\n".join(L), encoding="utf-8")
    (EGO / "ego_results.json").write_text(
        json.dumps({"rows": rows, "coverage": cov}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log(f"\n-> {EGO / 'ego_results.csv'}")


if __name__ == "__main__":
    main()
