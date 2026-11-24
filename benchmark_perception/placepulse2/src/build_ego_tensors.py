"""
FASE F (preparacion) — Tensores de vecindario a PROFUNDIDAD FIJA ego-0..3.

Por que hace falta reconstruirlos
---------------------------------
El tensor heredado (`Paso4_assemble_neighbourhood_tensors.py`) usa profundidad
ADAPTATIVA: expande el ego-graph 1 -> 2 -> 3 y se detiene en cuanto encuentra
>= 5 puntos de crimen. Es decir, cada nodo usa una profundidad distinta y el
`coverage_report.csv` la registra en `depth_used`. Ese diseno es razonable para
maximizar cobertura, pero NO permite responder la pregunta cientifica de la
sesion 3:

    "cuanto contexto espacial aporta informacion y a partir de que
     profundidad empieza a introducir ruido?"

Para responderla hace falta la misma representacion calculada a profundidad
FIJA k = 0, 1, 2, 3.

Definicion de ego-k
-------------------
    ego-0 : solo el propio nodo (puntos de crimen cuyo nearest_node == nodo)
    ego-k : todos los nodos a <= k aristas en el grafo de calles no dirigido

Todo lo demas se mantiene IDENTICO al pipeline heredado para que los resultados
sean comparables: N_POINTS=5 puntos por vecindario, N_IMAGES=4 imagenes por
punto, muestreo espacial estratificado por KMeans sobre lat/lon, seed 42,
sobremuestreo con reemplazo documentado cuando hay menos puntos/imagenes.

Conjunto de evaluacion comun
----------------------------
La cobertura cae al reducir k (en ego-0 muchos nodos no tienen ningun punto de
crimen propio). Comparar profundidades sobre conjuntos de nodos distintos seria
una comparacion injusta. Por eso se marca explicitamente el subconjunto de
nodos con cobertura en TODAS las profundidades (`common_nodes`), que es el que
usara el experimento. Es el mismo principio que se aplico al comparar encoders.

Salidas:
    outputs/ego/<distrito>_2016_ego<k>_tensor.npy        (n_nodos, 20, 768)
    outputs/ego/<distrito>_2016_ego<k>_nodes.json        ids en el orden del tensor
    outputs/ego/<distrito>_2016_ego_coverage.csv         cobertura por nodo y k
    outputs/ego/<distrito>_2016_common_nodes.json        nodos con cobertura en todo k
    outputs/ego/ego_build_summary.json

Uso:
    python build_ego_tensors.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils import OUTPUTS, REPO_ROOT, log, set_seed

warnings.filterwarnings("ignore")

GRAPH_DIR = REPO_ROOT / "graph" / "graph_output_v2"
LIMA = REPO_ROOT / "external_assets" / "lima_graph"
OUT = OUTPUTS / "ego"

N_POINTS = 5          # identico al pipeline heredado
N_IMAGES_PER_PT = 4   # identico al pipeline heredado
SEED = 42
DEPTHS = (0, 1, 2, 3)
DISTRICTS = {"Barranco": "Barranco", "La_Victoria": "La Victoria"}


def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    E = np.load(LIMA / "embeddings_export" / "embeddings.npy")
    meta = json.loads((REPO_ROOT / "external_assets" / "embeddings_export"
                       / "metadata.json").read_text(encoding="utf-8"))
    df = pd.DataFrame({"row": np.arange(len(meta["paths"])),
                       "district": meta["districts"],
                       "crime_id": meta["crime_ids"],
                       "heading": meta["headings"]})
    assert len(df) == E.shape[0], "metadata y embeddings desalineados"
    return E, df


def sample_points(cand: pd.DataFrame, rng: np.random.Generator) -> tuple[list[str], bool]:
    """N_POINTS puntos con muestreo espacial estratificado (KMeans), como el heredado."""
    ids = cand["CRIME_ID"].astype(str).unique().tolist()
    if len(ids) == 0:
        return [], False
    if len(ids) <= N_POINTS:
        # completar con reemplazo, documentado
        picked = list(rng.choice(ids, size=N_POINTS, replace=True))
        return picked, len(ids) < N_POINTS
    xy = cand.drop_duplicates("CRIME_ID")[["longitude", "latitude"]].to_numpy()
    uid = cand.drop_duplicates("CRIME_ID")["CRIME_ID"].astype(str).to_numpy()
    km = KMeans(n_clusters=N_POINTS, n_init=10, random_state=SEED).fit(xy)
    picked = []
    for c in range(N_POINTS):
        members = uid[km.labels_ == c]
        picked.append(str(rng.choice(members)) if len(members) else str(rng.choice(uid)))
    return picked, False


def node_tensor(point_ids, emb_by_crime, rng) -> tuple[np.ndarray | None, bool]:
    """(20, 768) = 5 puntos x 4 imagenes."""
    rows, over_img = [], False
    for pid in point_ids:
        avail = emb_by_crime.get(pid)
        if avail is None or len(avail) == 0:
            return None, over_img
        if len(avail) < N_IMAGES_PER_PT:
            over_img = True
            sel = rng.choice(avail, size=N_IMAGES_PER_PT, replace=True)
        else:
            sel = rng.choice(avail, size=N_IMAGES_PER_PT, replace=False)
        rows.extend(sel.tolist())
    return np.asarray(rows, dtype=np.int64), over_img


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    E, meta = load_embeddings()
    summary = {"seed": SEED, "n_points": N_POINTS, "n_images_per_point": N_IMAGES_PER_PT,
               "depths": list(DEPTHS), "districts": {}}

    for slug, dname in DISTRICTS.items():
        log(f"\n=== {slug} ===")
        edges = pd.read_csv(GRAPH_DIR / f"{slug}_2016_edges.csv")
        nodes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_dualgmm_nodes.csv")
        crimes = pd.read_csv(GRAPH_DIR / f"{slug}_2016_crimes_snapped.csv")
        crimes["CRIME_ID"] = crimes["CRIME_ID"].astype(str)
        crimes["nearest_node"] = crimes["nearest_node"].astype(str)
        nodes["node_id"] = nodes["node_id"].astype(str)

        G = nx.Graph()
        G.add_nodes_from(nodes["node_id"])
        G.add_edges_from(zip(edges["u"].astype(str), edges["v"].astype(str)))

        sub = meta[meta["district"] == dname]
        emb_by_crime: dict[str, np.ndarray] = {
            str(k): g["row"].to_numpy() for k, g in sub.groupby("crime_id")}
        log(f"  nodos {G.number_of_nodes()} | aristas {G.number_of_edges()} | "
            f"puntos con imagenes {len(emb_by_crime):,}")

        cov_rows = []
        tensors: dict[int, dict[str, np.ndarray]] = {k: {} for k in DEPTHS}

        for k in DEPTHS:
            rng = np.random.default_rng(SEED + k)   # determinista por profundidad
            n_ok = 0
            for nid in nodes["node_id"]:
                if nid not in G:
                    continue
                ego = ({nid} if k == 0
                       else set(nx.ego_graph(G, nid, radius=k).nodes()))
                cand = crimes[crimes["nearest_node"].isin(ego)]
                # solo puntos que realmente tienen imagenes
                cand = cand[cand["CRIME_ID"].isin(emb_by_crime)]
                pts, over_pt = sample_points(cand, rng)
                idx, over_img = (node_tensor(pts, emb_by_crime, rng)
                                 if pts else (None, False))
                ok = idx is not None
                if ok:
                    tensors[k][nid] = idx
                    n_ok += 1
                cov_rows.append({"node_id": nid, "depth": k,
                                 "n_ego_nodes": len(ego),
                                 "n_points_with_images": cand["CRIME_ID"].nunique(),
                                 "oversampled_points": bool(over_pt),
                                 "oversampled_images": bool(over_img),
                                 "included": bool(ok)})
            log(f"  ego-{k}: {n_ok}/{len(nodes)} nodos con cobertura "
                f"({n_ok/len(nodes):.1%})")

        cov = pd.DataFrame(cov_rows)
        cov.to_csv(OUT / f"{slug}_2016_ego_coverage.csv", index=False)

        common = sorted(set.intersection(*[set(tensors[k]) for k in DEPTHS]))
        log(f"  nodos con cobertura en TODAS las profundidades: {len(common)}")
        (OUT / f"{slug}_2016_common_nodes.json").write_text(
            json.dumps(common), encoding="utf-8")

        per_depth = {}
        for k in DEPTHS:
            ids = sorted(tensors[k])
            T = np.stack([E[tensors[k][i]] for i in ids]).astype(np.float32)
            assert T.shape[1:] == (N_POINTS * N_IMAGES_PER_PT, E.shape[1])
            np.save(OUT / f"{slug}_2016_ego{k}_tensor.npy", T)
            (OUT / f"{slug}_2016_ego{k}_nodes.json").write_text(
                json.dumps(ids), encoding="utf-8")
            per_depth[f"ego{k}"] = {"n_nodes": len(ids), "shape": list(T.shape)}
            log(f"  ego-{k} tensor {T.shape}")

        summary["districts"][slug] = {
            "n_graph_nodes": int(G.number_of_nodes()),
            "n_edges": int(G.number_of_edges()),
            "n_crime_points_with_images": len(emb_by_crime),
            "per_depth": per_depth,
            "n_common_nodes": len(common),
        }

    (OUT / "ego_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
