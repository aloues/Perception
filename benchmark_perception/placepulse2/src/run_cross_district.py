"""
FASE G — Generalizacion entre distritos: Barranco -> La Victoria y La Victoria -> Barranco.

Pregunta cientifica
-------------------
Un clasificador de crime-risk entrenado con las imagenes/estructura de UN distrito,
generaliza a otro distrito de Lima con distinta prevalencia y morfologia?
Decision de tesis que depende de ello: si el modelo puede desplegarse en distritos
sin etiquetas (transferencia) o si hay que calibrar/entrenar por distrito.

Hipotesis
---------
H1  La transferencia cae respecto a la validacion espacial intra-distrito.
H2  La senal estructural (topologica) transfiere peor que la visual, porque sus
    escalas dependen del tamano del grafo de cada distrito.

Diseno (identico a run_ego_experiment.py para que sea comparable)
-----------------------------------------------------------------
- Mismos features (visual mean++std 1,536-d / structural 5-d / fused 1,541-d),
  mismos nodos comunes, misma regresion logistica con StandardScaler ajustado
  SOLO con el distrito de entrenamiento.
- Entrenamiento: todos los nodos comunes del distrito origen.
  Test: todos los nodos comunes del distrito destino (nunca vistos, sin solape
  espacial alguno con el entrenamiento: son distritos distintos).
- Metrica: AUC (insensible a prevalencia) con IC95% bootstrap; ademas se reporta
  la probabilidad media predicha vs la prevalencia real en destino para
  cuantificar el corrimiento de calibracion (prevalencia 63% vs 26%).
- Referencia: AUC intra-distrito con GroupKFold espacial (ego_results.csv).

Etiqueta = crime-risk observado (dual-GMM de delitos 2016). NO es percepcion.

Uso:  python run_cross_district.py
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_ego_experiment import (DEPTHS, DISTRICTS, EGO, GRAPH_DIR, SEED, auc,
                                auc_ci, structural_features)
from utils import log, set_seed


def load_district(slug: str, k: int):
    nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
    nodes["node_id"] = nodes["node_id"].astype(str)
    edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
    G = nx.Graph()
    G.add_nodes_from(nodes["node_id"])
    G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))
    common = json.loads((EGO / f"{slug}_2016_common_nodes.json").read_text())
    lab = {n: int(v == "peligroso") for n, v in zip(nodes["node_id"], nodes["label"])}
    y = np.array([lab[n] for n in common])
    ids = json.loads((EGO / f"{slug}_2016_ego{k}_nodes.json").read_text())
    pos = {n: i for i, n in enumerate(ids)}
    T = np.load(EGO / f"{slug}_2016_ego{k}_tensor.npy")[[pos[n] for n in common]]
    visual = np.hstack([T.mean(axis=1), T.std(axis=1)])
    return visual, structural_features(G, common), y


def fit_predict(Xtr, ytr, Xte):
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main() -> None:
    set_seed(SEED)
    intra = pd.read_csv(EGO / "ego_results.csv")
    intra = intra[intra.cv_scheme == "spatial"]
    rows = []
    for k in DEPTHS:
        data = {s: load_district(s, k) for s in DISTRICTS}
        for src in DISTRICTS:
            for dst in DISTRICTS:
                if src == dst:
                    continue
                Vs, Ss, ys = data[src]
                Vd, Sd, yd = data[dst]
                sets = {"visual_only": (Vs, Vd), "structural_only": (Ss, Sd),
                        "fused": (np.hstack([Vs, Ss]), np.hstack([Vd, Sd]))}
                for name, (Xtr, Xte) in sets.items():
                    if name == "structural_only" and k != 0:
                        continue
                    p = fit_predict(Xtr, ys, Xte)
                    a = auc(yd, p)
                    lo, hi = auc_ci(yd, p)
                    ref = intra[(intra.district == dst.replace("_", " "))
                                & (intra.strategy == name)
                                & ((intra.depth == f"ego-{k}") | (name == "structural_only"))]
                    ref_auc = float(ref.auc.iloc[0]) if len(ref) else float("nan")
                    rows.append({
                        "train": src.replace("_", " "), "test": dst.replace("_", " "),
                        "strategy": name,
                        "depth": "n/a" if name == "structural_only" else f"ego-{k}",
                        "n_train": len(ys), "n_test": len(yd),
                        "prev_train": round(float(ys.mean()), 3),
                        "prev_test": round(float(yd.mean()), 3),
                        "mean_pred_test": round(float(p.mean()), 3),
                        "auc_transfer": round(a, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                        "auc_intra_district_spatialCV": round(ref_auc, 4),
                        "delta_vs_intra": round(a - ref_auc, 4), "seed": SEED})
                    log(f"ego-{k} {src}->{dst} {name:<16} AUC={a:.3f} "
                        f"[{lo:.3f},{hi:.3f}] intra={ref_auc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(EGO / "cross_district_results.csv", index=False)

    L = ["# Fase G — generalizacion entre distritos (crime-risk, Lima 2016)", "",
         "Etiqueta: crime-risk observado (dual-GMM). NO es percepcion humana.",
         "AUC con IC95% bootstrap sobre los nodos del distrito destino; `intra` = AUC",
         "del mismo distrito destino con GroupKFold espacial (ego_results.csv).", "",
         "| Train | Test | Estrategia | Prof. | AUC transfer | IC95% | AUC intra | Delta | prev test | pred media |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.train} | {r.test} | `{r.strategy}` | {r.depth} | **{r.auc_transfer:.3f}** | "
                 f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.auc_intra_district_spatialCV:.3f} | "
                 f"{r.delta_vs_intra:+.3f} | {r.prev_test:.2f} | {r.mean_pred_test:.2f} |")
    (EGO / "cross_district_results.md").write_text("\n".join(L), encoding="utf-8")
    log(f"-> {EGO / 'cross_district_results.csv'}")


if __name__ == "__main__":
    main()
