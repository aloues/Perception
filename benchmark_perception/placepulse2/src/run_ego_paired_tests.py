"""
FASE F (inferencia) — Diferencias PAREADAS de AUC entre profundidades y estrategias.

Pregunta: la superioridad aparente de ego-1 (y del fusionado) es estadisticamente
distinguible, o cae dentro del ruido de n~235-314 nodos?

Metodo: se recalculan las predicciones out-of-fold (mismo pipeline, misma semilla,
GroupKFold espacial) y se hace bootstrap PAREADO sobre nodos: en cada remuestreo se
calcula AUC(A) - AUC(B) sobre los MISMOS nodos. IC95% percentil, 2000 remuestreos;
p bilateral = 2*min(P(d<=0), P(d>=0)). Se corrige por multiples comparaciones con
Holm dentro de cada distrito.

Salidas: outputs/ego/ego_paired_tests.csv / .md ; ego_oof_predictions.npz
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_ego_experiment import (DEPTHS, DISTRICTS, EGO, GRAPH_DIR, N_BLOCKS, SEED,
                                auc, spatial_blocks, structural_features)
from utils import log, set_seed

N_BOOT = 2000


def paired(y, pa, pb, seed=SEED):
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return auc(y, pa) - auc(y, pb), lo, hi, min(1.0, p)


def holm(ps):
    order = np.argsort(ps)
    m, out, run = len(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        out[i] = min(1.0, run)
    return out


def main() -> None:
    set_seed(SEED)
    rows, store = [], {}
    for slug in DISTRICTS:
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
        nodes["node_id"] = nodes["node_id"].astype(str)
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        G = nx.Graph()
        G.add_nodes_from(nodes["node_id"])
        G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))
        common = json.loads((EGO / f"{slug}_2016_common_nodes.json").read_text())
        lab = dict(zip(nodes["node_id"], (nodes["label"] == "peligroso").astype(int)))
        y = np.array([lab[n] for n in common])
        st = structural_features(G, common)
        groups = spatial_blocks(nodes, common)
        oof = {}
        for k in DEPTHS:
            ids = json.loads((EGO / f"{slug}_2016_ego{k}_nodes.json").read_text())
            pos = {n: i for i, n in enumerate(ids)}
            T = np.load(EGO / f"{slug}_2016_ego{k}_tensor.npy")[[pos[n] for n in common]]
            vis = np.hstack([T.mean(axis=1), T.std(axis=1)])
            for name, X in (("visual", vis), ("fused", np.hstack([vis, st]))):
                clf = Pipeline([("sc", StandardScaler()),
                                ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
                oof[(name, k)] = cross_val_predict(clf, X, y, cv=GroupKFold(N_BLOCKS),
                                                   groups=groups, method="predict_proba")[:, 1]
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
        oof[("structural", 0)] = cross_val_predict(clf, st, y, cv=GroupKFold(N_BLOCKS),
                                                   groups=groups, method="predict_proba")[:, 1]
        for key, v in oof.items():
            store[f"{slug}|{key[0]}|ego{key[1]}"] = v
        store[f"{slug}|y"] = y

        comps = []
        for name in ("visual", "fused"):
            for k in (0, 2, 3):
                comps.append((f"{name} ego-1 vs ego-{k}", oof[(name, 1)], oof[(name, k)]))
        for k in DEPTHS:
            comps.append((f"fused vs visual @ego-{k}", oof[("fused", k)], oof[("visual", k)]))
        comps.append(("fused ego-1 vs structural", oof[("fused", 1)], oof[("structural", 0)]))
        comps.append(("visual ego-1 vs structural", oof[("visual", 1)], oof[("structural", 0)]))
        res = []
        for label, a, b in comps:
            d, lo, hi, p = paired(y, a, b)
            res.append([label, d, lo, hi, p])
            log(f"  {slug} {label:<32} dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
        ph = holm(np.array([r[4] for r in res]))
        for r, p_adj in zip(res, ph):
            rows.append({"district": slug.replace("_", " "), "comparison": r[0],
                         "delta_auc": round(r[1], 4), "ci95_low": round(r[2], 4),
                         "ci95_high": round(r[3], 4), "p_boot": round(r[4], 4),
                         "p_holm": round(float(p_adj), 4),
                         "significant_holm_0.05": bool(p_adj < 0.05), "n_nodes": len(y)})

    df = pd.DataFrame(rows)
    df.to_csv(EGO / "ego_paired_tests.csv", index=False)
    np.savez_compressed(EGO / "ego_oof_predictions.npz", **store)
    L = ["# Fase F — diferencias pareadas de AUC (bootstrap pareado, Holm por distrito)", "",
         "| Distrito | Comparacion | dAUC | IC95% | p Holm | signif. |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.district} | {r.comparison} | {r.delta_auc:+.3f} | "
                 f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.p_holm:.3f} | "
                 f"{'si' if r['significant_holm_0.05'] else 'no'} |")
    (EGO / "ego_paired_tests.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
