"""Pruebas de la distancia por red con snap a arista (src/network_distance.py).

Riesgo que cubren: que la posición a lo largo de la arista se invierta cuando la arista viene
orientada v -> u, que dos puntos de la misma cuadra se midan por la intersección en vez de a lo
largo de la cuadra, y que quitar una vía expresa la convierta realmente en barrera.

    ..\\..\\.venv\\Scripts\\python.exe tests/test_network_distance.py    (o con pytest, si se instala)
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from network_distance import (BARRIER_HIGHWAYS, NetworkDistance, drop_highways,  # noqa: E402
                              snap_points, to_weighted_simple)


class approx:
    """Sustituto mínimo de pytest.approx (el venv no trae pytest)."""

    def __init__(self, v, tol=1e-6):
        self.v, self.tol = v, tol

    def __eq__(self, other):
        return abs(other - self.v) <= self.tol


def _graph(nodes: dict, edges: list[tuple]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n, (x, y) in nodes.items():
        G.add_node(n, x=x, y=y)
    for u, v, hw in edges:
        L = float(np.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1]))
        G.add_edge(u, v, 0, length=L, highway=hw)
    return G


def _dist(G, pts_edges, pts, cutoff=2000.0):
    X, Y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    snaps = snap_points(G, pts_edges, X, Y, np.zeros(len(pts)))
    return NetworkDistance(to_weighted_simple(G), snaps, cutoff), snaps


def test_straight_road_and_same_edge():
    G = _graph({1: (0, 0), 2: (100, 0), 3: (200, 0)},
               [(1, 2, "residential"), (2, 3, "residential")])
    nd, snaps = _dist(G, [(1, 2, 0), (2, 3, 0), (1, 2, 0)], [(30, 2), (170, -1), (80, 0)])
    assert snaps[0].pos == approx(30)
    assert nd.between(0, 1) == approx(140)      # 70 hasta el nodo 2 + 70
    assert nd.between(0, 2) == approx(50)       # misma cuadra: a lo largo de la arista
    assert nd.between(1, 0) == approx(nd.between(0, 1))


def test_reversed_edge_orientation_is_canonical():
    # la arista viene como 2 -> 1: la posición debe medirse igual desde el nodo canónico 1
    G = _graph({1: (0, 0), 2: (100, 0)}, [(2, 1, "residential")])
    nd, snaps = _dist(G, [(2, 1, 0), (2, 1, 0)], [(30, 0), (90, 0)])
    assert (snaps[0].u, snaps[0].v) == (1, 2)
    assert snaps[0].pos == approx(30)
    assert nd.between(0, 1) == approx(60)


def test_motorway_becomes_barrier_when_dropped():
    # rectángulo 500 x 50: abajo y arriba calles, lado derecho calle, lado izquierdo vía expresa
    nodes = {1: (0, 0), 2: (500, 0), 3: (500, 50), 4: (0, 50)}
    edges = [(1, 2, "residential"), (2, 3, "residential"), (3, 4, "residential"), (4, 1, "motorway")]
    G = _graph(nodes, edges)
    pts, pe = [(50, 0), (50, 50)], [(1, 2, 0), (3, 4, 0)]
    nd, _ = _dist(G, pe, pts)
    assert nd.between(0, 1) == approx(150)      # cruza por la vía expresa
    Gb = drop_highways(G, BARRIER_HIGHWAYS)
    assert not Gb.has_edge(4, 1)
    ndb, _ = _dist(Gb, pe, pts)
    assert ndb.between(0, 1) == approx(950)     # rodea por la calle de la derecha


def test_cutoff_gives_infinite_distance():
    G = _graph({1: (0, 0), 2: (1000, 0)}, [(1, 2, "residential")])
    G.add_node(3, x=5000, y=0)
    G.add_node(4, x=6000, y=0)
    G.add_edge(3, 4, 0, length=1000.0, highway="residential")      # otra componente
    nd, _ = _dist(G, [(1, 2, 0), (3, 4, 0)], [(10, 0), (5010, 0)], cutoff=400)
    assert np.isinf(nd.between(0, 1))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
