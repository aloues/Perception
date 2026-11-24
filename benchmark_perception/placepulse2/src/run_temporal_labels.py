"""
FASE H (parcial) — Validez temporal de las etiquetas de crime-risk (2016 -> 2017..2023).

Que SI se puede hacer con datos reales
--------------------------------------
Hay delitos reales 2016-2023 (`distritos/crimen_distritos_seleccionados.csv`) pero
imagenes de calle SOLO para los puntos de 2016 (embeddings_export, todos 2016).
Por tanto NO se puede hacer validacion temporal de IMAGENES (imagen de anio t vs
imagen de anio t'): eso queda BLOQUEADO y no se inventan datos 2023.

Si se puede responder una pregunta distinta y valida:
    "Las features 2016 (imagen del vecindario + estructura), entrenadas SOLO con la
     etiqueta 2016, predicen el crime-risk FUTURO de cada nodo? Anaden algo sobre
     el baseline de persistencia (simplemente conocer el crimen 2016)?"

Diseno
------
- Delitos del anio Y se asignan al nodo del grafo mas cercano (KD-tree en metros
  locales). Riesgo de vecindario_Y(nodo) = delitos del anio Y en ego-1 del nodo.
- Etiqueta_Y = 1 si el riesgo_Y esta en el top-q de los nodos comunes, con q =
  prevalencia 2016 del distrito. Corte por RANGO y no por umbral absoluto, porque
  el volumen cambia mucho entre anios (caida 2020) y un umbral fijo mediria el
  volumen, no la distribucion espacial.
- Predictor imagen: probabilidades OUT-OF-FOLD del modelo entrenado con etiqueta
  2016 y GroupKFold espacial (5 bloques). Ningun nodo se predice con un modelo que
  vio su bloque espacial. Features: ego-1 visual y fused.
- Baseline de persistencia: riesgo 2016 (ego-1) como puntuacion. NOTA: usa el
  crimen mismo, es un techo/referencia, no un modelo comparable en coste.
- Metrica: AUC contra etiqueta_Y con IC95% bootstrap.

Etiqueta = crime-risk observado. NO es percepcion.
Uso:  python run_temporal_labels.py
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_ego_experiment import (DISTRICTS, EGO, GRAPH_DIR, N_BLOCKS, SEED, auc,
                                auc_ci, spatial_blocks, structural_features)
from utils import REPO_ROOT, log, set_seed

YEARS = list(range(2016, 2024))
CRIME_CSV = REPO_ROOT / "distritos" / "crimen_distritos_seleccionados.csv"
K = 1   # profundidad usada (la mejor en E/F)


def main() -> None:
    set_seed(SEED)
    crimes = pd.read_csv(CRIME_CSV, sep=";")
    rows = []
    for slug in DISTRICTS:
        dname = slug.replace("_", " ")
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
        nodes["node_id"] = nodes["node_id"].astype(str)
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        G = nx.Graph()
        G.add_nodes_from(nodes["node_id"])
        G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))
        common = json.loads((EGO / f"{slug}_2016_common_nodes.json").read_text())

        lat0 = float(nodes["lat"].mean())
        def to_m(lon, lat):
            return np.c_[(np.asarray(lon) * 111320 * np.cos(np.radians(lat0))),
                         np.asarray(lat) * 110540]
        tree = cKDTree(to_m(nodes["lon"], nodes["lat"]))
        ids = nodes["node_id"].to_numpy()
        ego1 = {n: set(nx.ego_graph(G, n, radius=K).nodes()) for n in common}

        risk = {}
        for y in YEARS:
            c = crimes[(crimes.DISTRITO == dname) & (crimes.YEAR == y)]
            d, j = tree.query(to_m(c.longitude, c.latitude))
            cnt = pd.Series(ids[j][d < 500]).value_counts()   # descarta >500 m del grafo
            risk[y] = np.array([sum(cnt.get(m, 0) for m in ego1[n]) for n in common], float)
            log(f"  {dname} {y}: {len(c):,} delitos, {int((d < 500).sum()):,} asignados")

        lab16 = np.array([int(v == "peligroso") for v in
                          nodes.set_index("node_id").loc[common, "label"]])
        q = lab16.mean()

        def topq(r):
            thr = np.quantile(r, 1 - q)
            return (r > thr).astype(int) if (r > thr).sum() > 0 else (r >= thr).astype(int)

        # features 2016 ego-1 + estructura
        ids1 = json.loads((EGO / f"{slug}_2016_ego{K}_nodes.json").read_text())
        pos = {n: i for i, n in enumerate(ids1)}
        T = np.load(EGO / f"{slug}_2016_ego{K}_tensor.npy")[[pos[n] for n in common]]
        vis = np.hstack([T.mean(axis=1), T.std(axis=1)])
        st = structural_features(G, common)
        groups = spatial_blocks(nodes, common)
        preds = {}
        for name, X in (("visual_ego1", vis), ("fused_ego1", np.hstack([vis, st]))):
            clf = Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
            preds[name] = cross_val_predict(clf, X, lab16, cv=GroupKFold(N_BLOCKS),
                                            groups=groups, method="predict_proba")[:, 1]
        preds["persistencia_riesgo2016"] = risk[2016]

        # acuerdo entre mi etiqueta 2016 (rango, ego-1) y la heredada (dual-GMM)
        agree = float((topq(risk[2016]) == lab16).mean())
        log(f"  acuerdo etiqueta rango-2016 vs dual-GMM heredada: {agree:.1%}")

        for y in YEARS:
            ly = topq(risk[y])
            for name, p in preds.items():
                a = auc(ly, p)
                lo, hi = auc_ci(ly, p, n_boot=1000)
                rows.append({"district": dname, "year": y, "predictor": name,
                             "n_nodes": len(common), "prevalence_year": round(float(ly.mean()), 3),
                             "crimes_year_in_graph": int(risk[y].sum()),
                             "auc": round(a, 4), "ci95_low": round(lo, 4),
                             "ci95_high": round(hi, 4),
                             "label_agreement_2016": round(agree, 3), "seed": SEED})
                log(f"  {dname} {y} {name:<24} AUC={a:.3f} [{lo:.3f},{hi:.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(EGO / "temporal_labels_results.csv", index=False)
    L = ["# Fase H (parcial) — persistencia temporal del crime-risk", "",
         "Features de IMAGEN 2016 (ego-1) + label 2016 -> etiqueta de riesgo del anio Y.",
         "Imagenes solo 2016: la validacion temporal de IMAGENES esta BLOQUEADA.", "",
         "| Distrito | Anio | Predictor | AUC | IC95% | prev. |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.district} | {r.year} | `{r.predictor}` | **{r.auc:.3f}** | "
                 f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.prevalence_year:.2f} |")
    (EGO / "temporal_labels_results.md").write_text("\n".join(L), encoding="utf-8")
    log(f"-> {EGO / 'temporal_labels_results.csv'}")


if __name__ == "__main__":
    main()
