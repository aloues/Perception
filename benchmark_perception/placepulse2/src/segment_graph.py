"""
Grafo de SEGMENTOS viales (grafo dual / line graph) y vecindarios ego-k — asesoría 23/10/2026.

Unidades (prompt maestro §9.1):
    vista     una orientación de un spot (spot_id, heading)
    spot      punto donde se capturan las vistas
    segmento  arista del grafo OSM simplificado = tramo entre dos intersecciones (o callejón sin
              salida). Clave estable `"{min(u,v)}_{max(u,v)}_{k}"`. Una calle con el mismo nombre
              tiene VARIOS segmentos; no se fusionan por nombre.
    intersección  nodo OSM; NO es un nodo del grafo de ML.

Grafo de ML (dual): un nodo por segmento; dos segmentos son vecinos si comparten una intersección.
Las aristas del dual NO llevan features visuales (la información visual vive solo en los nodos):
así se evita "poner la misma información en vértices y aristas" (advertencia del asesor). Un puente
o un paso a desnivel no comparte nodo OSM con la vía de abajo → no hay conectividad (test).

ego-k (convención ACUMULATIVA, ≤ k saltos en el dual):
    ego-0  solo el segmento focal (baseline estricto)
    ego-1  focal + segmentos que comparten una intersección con él
    ego-2  focal + vecinos hasta dos saltos
`ring(k)` = exactamente k saltos (para medir decaimiento de similitud).

Variante del asesor ("barrio 0 = esta calle y sus esquinas"): `ego0_plus_corners` añade los spots
de segmentos vecinos situados a ≤ `corner_m` de una intersección del focal. Se reporta aparte; el
baseline principal sigue siendo ego-0 estricto.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point


def segment_key(u, v, k) -> str:
    a, b = (u, v) if u <= v else (v, u)
    return f"{a}_{b}_{k}"


def segments_table(G: nx.MultiGraph | nx.MultiDiGraph) -> pd.DataFrame:
    """Una fila por segmento NO dirigido (geometría en las coordenadas del grafo; usar grafo proyectado).

    Si el grafo es dirigido, (u,v,k) y (v,u,k) son el mismo segmento: se conserva uno.
    """
    rows, seen = [], set()
    for u, v, k, d in G.edges(keys=True, data=True):
        key = segment_key(u, v, k)
        if key in seen:
            continue
        seen.add(key)
        geom = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                                (G.nodes[v]["x"], G.nodes[v]["y"])])
        hw = d.get("highway", "")
        flag = lambda name: any(str(x) not in ("no", "False", "nan", "") for x in np.atleast_1d(d.get(name, "")))
        rows.append({"seg": key, "u": u, "v": v, "k": k, "length_m": float(d.get("length", geom.length)),
                     "highway": hw if isinstance(hw, str) else "|".join(map(str, hw)),
                     "bridge": flag("bridge"), "tunnel": flag("tunnel"), "geometry": geom})
    return pd.DataFrame(rows)


def dual_adjacency(seg: pd.DataFrame) -> sp.csr_matrix:
    """Adyacencia del line graph: i~j si los segmentos comparten un extremo (intersección). Sin lazos."""
    inc = {}
    for i, (u, v) in enumerate(zip(seg["u"], seg["v"])):
        for n in {u, v}:
            inc.setdefault(n, []).append(i)
    rows, cols = [], []
    for members in inc.values():
        for a in members:
            for b in members:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    n = len(seg)
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0                                   # pares con 2 intersecciones comunes cuentan una vez
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def reach_matrices(A: sp.csr_matrix, sources: np.ndarray, K: int) -> list[sp.csr_matrix]:
    """R[k][r, j] = 1 si el segmento j está a ≤ k saltos de sources[r] (acumulativo; R[0] = identidad)."""
    n = A.shape[0]
    R = sp.csr_matrix((np.ones(len(sources), dtype=np.float32), (np.arange(len(sources)), sources)),
                      shape=(len(sources), n))
    out = [R.copy()]
    B = (A + sp.identity(n, format="csr", dtype=np.float32)).tocsr()
    for _ in range(K):
        R = (R @ B).tocsr()
        R.data[:] = 1.0
        out.append(R.copy())
    return out


def hop_distances(A: sp.csr_matrix, sources: np.ndarray, K: int) -> sp.csr_matrix:
    """D[r, j] = nº de saltos (1..K) de sources[r] a j; 0 implícito = no alcanzable o el propio (ver R[0])."""
    R = reach_matrices(A, sources, K)
    D = sp.csr_matrix(R[0].shape, dtype=np.float32)
    for k in range(1, K + 1):
        ring = (R[k] - R[k - 1]).tocsr()
        ring.eliminate_zeros()
        ring.data[:] = k
        D = D + ring
    return D.tocsr()


@dataclass
class SnapResult:
    seg_idx: np.ndarray        # índice de segmento o -1 si supera max_dist
    dist_m: np.ndarray
    second_dist_m: np.ndarray  # distancia al segundo segmento más cercano NO adyacente (inf si no hay)
    ambiguous: np.ndarray      # segundo segmento no adyacente a < ambiguity_m de la primera distancia
    near_intersection: np.ndarray  # a ≤ corner_m de un extremo del segmento asignado


def snap_to_segments(seg: pd.DataFrame, A: sp.csr_matrix, xy: np.ndarray, node_xy: dict,
                     max_dist: float = 30.0, ambiguity_m: float = 3.0, corner_m: float = 15.0) -> SnapResult:
    """Asigna cada punto (coordenadas proyectadas, m) a UN segmento. Regla documentada: el más cercano.

    Un punto a > max_dist queda sin asignar (-1), no se fuerza. Se marca ambiguo si hay otro segmento
    NO adyacente casi a la misma distancia (avenidas paralelas, pasos a desnivel): adyacente = misma
    esquina, y esa ambigüedad es intrínseca a las esquinas (se marca aparte como near_intersection).
    """
    tree = STRtree(list(seg["geometry"]))
    n = len(xy)
    idx = np.full(n, -1)
    d1 = np.full(n, np.inf)
    d2 = np.full(n, np.inf)
    amb = np.zeros(n, bool)
    corner = np.zeros(n, bool)
    A = A.tocsr()
    for i, (x, y) in enumerate(xy):
        p = Point(x, y)
        cand = tree.query(p.buffer(max_dist + ambiguity_m))
        if len(cand) == 0:
            continue
        ds = np.array([seg["geometry"].iloc[c].distance(p) for c in cand])
        order = np.argsort(ds, kind="stable")
        best = cand[order[0]]
        if ds[order[0]] > max_dist:
            d1[i] = ds[order[0]]
            continue
        idx[i], d1[i] = best, ds[order[0]]
        adj = set(A.indices[A.indptr[best]:A.indptr[best + 1]].tolist())
        for o in order[1:]:
            if cand[o] not in adj:
                d2[i] = ds[o]
                break
        amb[i] = d2[i] - d1[i] < ambiguity_m
        u, v = seg["u"].iloc[best], seg["v"].iloc[best]
        corner[i] = min(np.hypot(x - node_xy[u][0], y - node_xy[u][1]),
                        np.hypot(x - node_xy[v][0], y - node_xy[v][1])) <= corner_m
    return SnapResult(idx, d1, d2, amb, corner)


def segment_midpoints(seg: pd.DataFrame) -> np.ndarray:
    return np.array([[g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y]
                     for g in seg["geometry"]])


def euclidean_count_matched(mid: np.ndarray, sources: np.ndarray, candidates: np.ndarray,
                            counts: np.ndarray, exclude: list[set] | None = None) -> list[np.ndarray]:
    """Para cada fuente, los `counts[r]` candidatos más cercanos en línea recta (excluye la propia y `exclude[r]`).

    Control de "radio" con el MISMO número de vecinos que la definición por red (count-matched).
    """
    tree = cKDTree(mid[candidates])
    out = []
    for r, s in enumerate(sources):
        c = int(counts[r])
        if c == 0:
            out.append(np.array([], dtype=int))
            continue
        ex = {s} | (exclude[r] if exclude is not None else set())
        k = min(len(candidates), c + len(ex) + 5)
        while True:
            _, nn = tree.query(mid[s], k=k)
            picked = [candidates[j] for j in np.atleast_1d(nn) if candidates[j] not in ex][:c]
            if len(picked) == c or k >= len(candidates):
                break
            k = min(len(candidates), 2 * k)
        out.append(np.array(picked, dtype=int))
    return out


def radius_sets(mid: np.ndarray, sources: np.ndarray, candidates: np.ndarray, r: float) -> list[np.ndarray]:
    """Segmentos candidatos cuyo punto medio está a ≤ r m del punto medio de la fuente (sin la propia)."""
    tree = cKDTree(mid[candidates])
    return [np.array([candidates[j] for j in tree.query_ball_point(mid[s], r) if candidates[j] != s], dtype=int)
            for s in sources]


def normalized_adjacency(A: sp.csr_matrix, self_loops: bool = True) -> sp.csr_matrix:
    """Â = D^{-1/2}(A + I)D^{-1/2} (Kipf & Welling 2017). Con self_loops=False y A=0 → identidad (control sin aristas)."""
    n = A.shape[0]
    M = (A + sp.identity(n, format="csr")) if self_loops else A.copy()
    deg = np.asarray(M.sum(axis=1)).ravel()
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    D = sp.diags(dinv)
    return (D @ M @ D).tocsr()


# ─── cruz del compañero (D7, prompt maestro 2026-10-29) ───────────────────────
# "embedding del subgrafo = punto central + vecinos a izquierda, derecha, arriba y abajo": vecindario de tamaño FIJO
# (≤ 4 brazos N/E/S/O por el rumbo centro→vecino). Versión "red": en cada brazo, el punto más cercano por DISTANCIA DE
# RED (no cruza manzanas); versión "sector": el más cercano en línea recta. Brazo vacío → −1 (se enmascara).

ARMS = ("N", "E", "S", "O")


def point_offsets(Gp, seg: pd.DataFrame, seg_idx: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para puntos con segmento asignado: distancia por la calle desde el extremo `u` y longitud del segmento."""
    t_u, L = np.full(len(xy), np.nan), np.full(len(xy), np.nan)
    for i, s in enumerate(seg_idx):
        if s < 0:
            continue
        g = seg["geometry"].iloc[s]
        u = seg["u"].iloc[s]
        t = g.project(Point(xy[i]))
        x0, y0 = g.coords[0]
        starts_at_u = np.hypot(x0 - Gp.nodes[u]["x"], y0 - Gp.nodes[u]["y"]) < 1e-6 or seg["u"].iloc[s] == seg["v"].iloc[s]
        t_u[i], L[i] = (t if starts_at_u else g.length - t), g.length
    return t_u, L


def network_point_distances(Gp, seg: pd.DataFrame, seg_idx: np.ndarray, xy: np.ndarray,
                            max_m: float) -> list[dict]:
    """Distancia de RED entre puntos situados sobre segmentos (≤ max_m). Devuelve, por punto, {j: metros}.

    d(i, j) = min( |t_i − t_j| si comparten segmento ;  min_{a ∈ ext(i), b ∈ ext(j)} d_i(a) + D(a, b) + d_j(b) ),
    con D = camino mínimo en el grafo primal (peso `length`). Puntos sin segmento: diccionario vacío.
    """
    t_u, L = point_offsets(Gp, seg, seg_idx, xy)
    at_node: dict = {}
    for j, s in enumerate(seg_idx):
        if s >= 0:
            at_node.setdefault(seg["u"].iloc[s], []).append((j, t_u[j]))
            at_node.setdefault(seg["v"].iloc[s], []).append((j, L[j] - t_u[j]))
    by_seg: dict = {}
    for j, s in enumerate(seg_idx):
        if s >= 0:
            by_seg.setdefault(s, []).append(j)
    cache: dict = {}
    out = [dict() for _ in range(len(xy))]
    for i, s in enumerate(seg_idx):
        if s < 0:
            continue
        best = out[i]
        for j in by_seg[s]:
            if j != i and abs(t_u[i] - t_u[j]) <= max_m:
                best[j] = abs(t_u[i] - t_u[j])
        for a, da in ((seg["u"].iloc[s], t_u[i]), (seg["v"].iloc[s], L[i] - t_u[i])):
            if da > max_m:
                continue
            if a not in cache:
                cache[a] = nx.single_source_dijkstra_path_length(Gp, a, cutoff=max_m, weight="length")
            for b, D in cache[a].items():
                if da + D > max_m:
                    continue
                for j, db in at_node.get(b, ()):
                    d = da + D + db
                    if j != i and d <= max_m and d < best.get(j, np.inf):
                        best[j] = d
    return out


def euclidean_point_distances(xy: np.ndarray, max_m: float) -> list[dict]:
    tree = cKDTree(xy)
    return [{j: float(np.hypot(*(xy[j] - xy[i]))) for j in tree.query_ball_point(xy[i], max_m) if j != i}
            for i in range(len(xy))]


def arm_of(dxy: np.ndarray) -> np.ndarray:
    """Brazo por rumbo (desde el norte, horario): N [315, 45), E [45, 135), S [135, 225), O [225, 315)."""
    b = np.degrees(np.arctan2(dxy[..., 0], dxy[..., 1])) % 360
    return (((b + 45) % 360) // 90).astype(int)


def cross_neighbors(xy: np.ndarray, dists: list[dict], min_m: float = 5.0) -> np.ndarray:
    """[n, 4] índice del vecino más cercano (según `dists`) en cada brazo N/E/S/O, o −1.

    `dists` define la versión: `network_point_distances` (cruz-red) o `euclidean_point_distances` (cruz-sector).
    Vecinos a < `min_m` en línea recta se excluyen (mismo panorama: los puntos de crimen se repiten).
    """
    out = np.full((len(xy), 4), -1)
    for i, d in enumerate(dists):
        if not d:
            continue
        js = np.array(list(d))
        dd = np.array([d[j] for j in js])
        dxy = xy[js] - xy[i]
        keep = np.hypot(dxy[:, 0], dxy[:, 1]) >= min_m
        js, dd, arms = js[keep], dd[keep], arm_of(dxy[keep])
        for a in range(4):
            m = arms == a
            if m.any():
                out[i, a] = js[m][np.argmin(dd[m])]
    return out
