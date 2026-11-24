"""
Distancia por red vial entre puntos con SNAP A ARISTA (EXP-026, Parte 2).

EXP-025 asignaba cada imagen a la intersección más cercana; con cuadras largas eso movía la imagen
decenas de metros (mediana 26–60 m) y hacía que dos imágenes de la misma cuadra quedaran a "0 o 1
saltos" según el lado. Aquí cada punto se proyecta sobre la arista más cercana y se guarda su
posición a lo largo de ella, de modo que

    d(i, j) = |pos_i − pos_j|                                   si comparten arista
    d(i, j) = min_{a ∈ {u_i, v_i}, b ∈ {u_j, v_j}} off_i(a) + SP(a, b) + off_j(b)   en otro caso

con SP el camino más corto (peso = longitud en metros) sobre la red NO dirigida. Es la distancia
que recorrería una persona por la calle, no en línea recta: una vía expresa o un parque sin calles
alargan el camino; si se quitan las vías expresas del grafo (`drop_highways`), pasan a ser barrera.

Funciones puras sobre networkx/shapely para poder probarlas con un grafo sintético
(tests/test_network_distance.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

BARRIER_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link"}


@dataclass(frozen=True)
class Snap:
    u: int            # extremo canónico (min) de la arista
    v: int            # extremo canónico (max)
    length: float     # longitud de la arista (m)
    pos: float        # distancia desde u a lo largo de la arista (m), 0..length
    dist: float       # distancia del punto a la arista (m)
    highway: str      # tipo OSM de la arista


def _highway_set(h) -> set[str]:
    return set(h) if isinstance(h, (list, tuple, set)) else {h}


def drop_highways(G: nx.MultiDiGraph, highways: set[str]) -> nx.MultiDiGraph:
    """Copia del grafo sin las aristas cuyo `highway` cae en `highways`; quita nodos aislados."""
    H = G.copy()
    H.remove_edges_from([(u, v, k) for u, v, k, h in G.edges(keys=True, data="highway")
                         if _highway_set(h) & highways])
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    return H


def edge_geometry(G: nx.MultiDiGraph, u, v, k) -> LineString:
    data = G.edges[u, v, k]
    if "geometry" in data:
        return data["geometry"]
    return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])


def snap_points(G: nx.MultiDiGraph, edges: list[tuple], X: np.ndarray, Y: np.ndarray,
                dists: np.ndarray) -> list[Snap]:
    """Convierte la arista más cercana (u, v, k) de cada punto en un Snap con posición canónica.

    `edges` y `dists` vienen de `ox.distance.nearest_edges(G, X, Y, return_dist=True)` sobre el
    grafo PROYECTADO (metros). La geometría de OSMnx va orientada de u a v.
    """
    out = []
    for (u, v, k), x, y, d in zip(edges, X, Y, dists):
        geom = edge_geometry(G, u, v, k)
        L = float(G.edges[u, v, k].get("length", geom.length))
        frac = geom.project(Point(x, y)) / geom.length if geom.length > 0 else 0.0
        pos_from_u = frac * L
        a, b = (u, v) if u <= v else (v, u)
        pos = pos_from_u if u <= v else L - pos_from_u
        hw = G.edges[u, v, k].get("highway", "")
        out.append(Snap(a, b, L, pos, float(d), hw if isinstance(hw, str) else "|".join(hw)))
    return out


def to_weighted_simple(G: nx.MultiDiGraph) -> nx.Graph:
    """Red no dirigida simple; entre aristas paralelas se conserva la más corta."""
    S = nx.Graph()
    S.add_nodes_from(G.nodes)
    for u, v, L in G.edges(data="length"):
        if u == v:
            continue
        if not S.has_edge(u, v) or L < S[u][v]["length"]:
            S.add_edge(u, v, length=float(L))
    return S


class NetworkDistance:
    """d(i, j) por red entre puntos ya snapeados. Cachea Dijkstra por nodo (cutoff fijo)."""

    def __init__(self, S: nx.Graph, snaps: list[Snap], cutoff: float):
        self.S, self.snaps, self.cutoff = S, snaps, cutoff
        self._cache: dict[int, dict[int, float]] = {}

    def _sp(self, node) -> dict:
        if node not in self._cache:
            self._cache[node] = nx.single_source_dijkstra_path_length(
                self.S, node, cutoff=self.cutoff, weight="length")
        return self._cache[node]

    def from_point(self, i: int) -> dict:
        """Distancia por red desde el punto i a cada nodo alcanzable dentro del cutoff."""
        s = self.snaps[i]
        out: dict = {}
        for node, off in ((s.u, s.pos), (s.v, s.length - s.pos)):
            for n, d in self._sp(node).items():
                t = off + d
                if t < out.get(n, np.inf):
                    out[n] = t
        return out

    def between(self, i: int, j: int, Di: dict | None = None) -> float:
        si, sj = self.snaps[i], self.snaps[j]
        best = np.inf
        if (si.u, si.v) == (sj.u, sj.v) and abs(si.length - sj.length) < 1.0:
            best = abs(si.pos - sj.pos)
        Di = self.from_point(i) if Di is None else Di
        for node, off in ((sj.u, sj.pos), (sj.v, sj.length - sj.pos)):
            if node in Di:
                best = min(best, Di[node] + off)
        return float(best)
