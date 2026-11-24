"""
EXP-013 — Contexto espacial SIN el artefacto de muestreo (FASE 5 rehecha).

PREGUNTA
--------
Una vez eliminado el sobremuestreo dependiente de la clase, ¿queda señal VISUAL real en el
vecindario para predecir crime-risk, o toda la señal era un artefacto del muestreo?

TARGET = CRIMEN OBSERVADO (dual-GMM 2016). NO es percepción.

POR QUÉ HACE FALTA
------------------
El tensor heredado usa 5 puntos x 4 imágenes = 20 slots, sobremuestreando CON REEMPLAZO cuando
hay menos de 5 puntos de crimen. Como la etiqueta se deriva del conteo de crímenes, los nodos
"seguros" reciben imágenes repetidas y los "peligrosos" imágenes diversas. EXP-009 demostró que
el mero número de filas únicas del tensor -sin ningún contenido visual- alcanza AUC 0.768/0.706,
igualando o superando a las features "visuales".

DISEÑO LIMPIO
-------------
Para cada K en {4, 8, 12}:
  - se incluye un nodo solo si su vecindario tiene AL MENOS K imágenes únicas disponibles;
  - se muestrean EXACTAMENTE K imágenes únicas, sin reemplazo, uniformemente del pool.
Así el número de filas únicas es CONSTANTE por construcción: el artefacto de EXP-009 queda
eliminado por diseño (esa feature pasa a tener información cero).

La selección introduce sesgo de prevalencia (se excluyen nodos con pocas imágenes, que tienden a
ser de clase "segura"). Por eso se reporta SIEMPRE n y prevalencia junto al AUC, y se incluye un
techo no visual explícito:

  - `n_imgs_disponibles (log)` : cuántas imágenes había en el pool antes de recortar a K.
    NO entra en el tensor, pero sigue siendo un proxy de densidad delictiva. Es la referencia
    contra la que debe compararse cualquier señal visual.

Una señal visual solo es creíble si supera de forma estable a ese techo no visual.

Validación: GroupKFold sobre 5 bloques espaciales (KMeans lat/lon), idéntica a EXP-007.
Métrica: AUC out-of-fold con IC 95% bootstrap.

Salidas:
    outputs/ego/ego_clean_results.csv
    outputs/ego/ego_clean_results.md

Uso:  python run_ego_clean.py
"""

from __future__ import annotations

import json
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_ego_tensors as B
from run_ego_experiment import EGO, GRAPH_DIR, N_BLOCKS, SEED, auc, auc_ci, structural_features
from utils import log, set_seed

warnings.filterwarnings("ignore")

KS = (4, 8, 12)
DEPTHS = (0, 1, 2)
DISTRICTS = {"Barranco": "Barranco", "La_Victoria": "La Victoria"}
MIN_NODES = 60          # por debajo de esto la estimación no es interpretable
MIN_MINORITY = 15       # mínimo de la clase minoritaria


def oof(X, y, g):
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
    n = min(N_BLOCKS, len(np.unique(g)))
    return cross_val_predict(clf, X, y, cv=GroupKFold(n), groups=g, method="predict_proba")[:, 1]


def main() -> None:
    set_seed(SEED)
    E, meta = B.load_embeddings()
    rows = []

    for slug, dname in DISTRICTS.items():
        log(f"\n=== {dname} ===")
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
        nodes["node_id"] = nodes["node_id"].astype(str)
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        crimes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_crimes_snapped.csv")
        crimes["CRIME_ID"] = crimes["CRIME_ID"].astype(str)
        crimes["nearest_node"] = crimes["nearest_node"].astype(str)

        G = nx.Graph()
        G.add_nodes_from(nodes["node_id"])
        G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))

        sub = meta[meta["district"] == dname]
        emb_by_crime = {str(k): g["row"].to_numpy() for k, g in sub.groupby("crime_id")}
        crimes = crimes[crimes["CRIME_ID"].isin(emb_by_crime)]
        lab = dict(zip(nodes["node_id"], (nodes["label"] == "peligroso").astype(int)))
        nd = nodes.set_index("node_id")

        for k in DEPTHS:
            # pool de imágenes únicas disponibles por nodo, a profundidad k
            pool = {}
            for nid in nodes["node_id"]:
                if nid not in G:
                    continue
                ego = {nid} if k == 0 else set(nx.ego_graph(G, nid, radius=k).nodes())
                cids = crimes.loc[crimes["nearest_node"].isin(ego), "CRIME_ID"].unique()
                if len(cids) == 0:
                    continue
                r = np.unique(np.concatenate([emb_by_crime[c] for c in cids]))
                pool[nid] = r

            for K in KS:
                ids = sorted(n for n, r in pool.items() if len(r) >= K)
                if len(ids) < MIN_NODES:
                    log(f"  ego-{k} K={K:<2}: solo {len(ids)} nodos -> se omite")
                    continue
                y = np.array([lab[n] for n in ids])
                if min(y.sum(), len(y) - y.sum()) < MIN_MINORITY:
                    log(f"  ego-{k} K={K:<2}: clase minoritaria {min(y.sum(), len(y)-y.sum())} -> se omite")
                    continue

                rng = np.random.default_rng(SEED + 100 * k + K)
                sel = np.stack([rng.choice(pool[n], size=K, replace=False) for n in ids])
                T = E[sel]                                   # (n, K, 768) -- K imágenes ÚNICAS
                mean, std = T.mean(1), T.std(1)
                navail = np.log1p([len(pool[n]) for n in ids])[:, None]

                xy = nd.loc[ids, ["lon", "lat"]].to_numpy()
                g = KMeans(N_BLOCKS, n_init=10, random_state=SEED).fit_predict(xy)
                st = structural_features(G, ids)

                feats = {
                    "NO-VISUAL techo: log(n imgs disponibles)": navail,
                    "visual mean": mean,
                    "visual mean+std": np.hstack([mean, std]),
                    "structural": st,
                    "fused (mean+std+struct)": np.hstack([mean, std, st]),
                }
                for name, X in feats.items():
                    p = oof(X, y, g)
                    a = auc(y, p)
                    lo, hi = auc_ci(y, p, n_boot=1000)
                    rows.append(dict(district=dname, depth=f"ego-{k}", K=K, feature=name,
                                     n_nodes=len(y), prevalence=round(float(y.mean()), 3),
                                     n_features=X.shape[1], auc=round(a, 4),
                                     ci95_low=round(lo, 4), ci95_high=round(hi, 4), seed=SEED))
                    log(f"  ego-{k} K={K:<2} {name:<42} n={len(y):4d} prev={y.mean():.2f} "
                        f"AUC={a:.4f} [{lo:.3f},{hi:.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(EGO / "ego_clean_results.csv", index=False)

    L = ["# EXP-013 — Contexto espacial SIN artefacto de muestreo", "",
         "**TARGET = CRIMEN OBSERVADO (Lima 2016). NO es percepción.**", "",
         "Cada nodo aporta EXACTAMENTE K imágenes ÚNICAS (sin reemplazo), así que el número de filas",
         "únicas del tensor es constante y el artefacto detectado en EXP-009 queda eliminado por diseño.",
         "Solo se incluyen nodos con al menos K imágenes disponibles, por lo que n y la prevalencia",
         "cambian con K: se reportan siempre.", "",
         "`NO-VISUAL techo` = log del número de imágenes disponibles antes de recortar a K. No entra en",
         "el tensor, pero sigue siendo proxy de densidad delictiva. **Una señal visual solo es creíble",
         "si supera de forma estable a ese techo.**", "",
         "Validación: GroupKFold sobre 5 bloques espaciales. AUC out-of-fold, IC 95% bootstrap.", "",
         "| Distrito | Prof. | K | Feature | n | prev | AUC | IC 95% |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.district} | {r.depth} | {r.K} | {r.feature} | {r.n_nodes} | {r.prevalence:.2f} | "
                 f"**{r.auc:.4f}** | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] |")
    (EGO / "ego_clean_results.md").write_text("\n".join(L), encoding="utf-8")
    log(f"\n-> {EGO / 'ego_clean_results.csv'}")


if __name__ == "__main__":
    main()
