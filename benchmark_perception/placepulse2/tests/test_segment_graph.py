"""Grafo de segmentos (dual) y ego-k sobre grafos de juguete con respuesta conocida."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from segment_graph import (dual_adjacency, euclidean_count_matched, hop_distances,  # noqa: E402
                           normalized_adjacency, reach_matrices, segments_table, snap_to_segments)


def chain_graph(n_nodes: int = 5, spacing: float = 100.0) -> nx.MultiGraph:
    """Calle recta 0—1—2—3—4 (4 segmentos) en metros."""
    G = nx.MultiGraph()
    for i in range(n_nodes):
        G.add_node(i, x=i * spacing, y=0.0)
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1, 0, length=spacing)
    return G


def test_chain_dual_and_ego_k():
    seg = segments_table(chain_graph())
    A = dual_adjacency(seg)
    assert A.shape == (4, 4) and A.nnz == 6           # s0-s1, s1-s2, s2-s3 en ambos sentidos
    R = reach_matrices(A, np.array([0]), 3)
    assert [int(r.sum()) for r in R] == [1, 2, 3, 4]  # ego-0..3 acumulativos desde el extremo
    D = hop_distances(A, np.array([1]), 2).toarray()[0]
    assert D.tolist() == [1, 0, 1, 2]                 # 0 = el propio (ver R[0])


def test_bridge_without_shared_node_is_not_connected():
    """Una vía elevada cruza por encima sin nodo común: cerca en el plano, sin conectividad."""
    G = chain_graph(3)                                 # calle de abajo: 0—1—2 en y=0
    G.add_node(10, x=100.0, y=-100.0)
    G.add_node(11, x=100.0, y=100.0)
    G.add_edge(10, 11, 0, length=200.0, bridge="yes")  # puente que pasa sobre el nodo 1 sin tocarlo
    seg = segments_table(G)
    A = dual_adjacency(seg)
    bridge = int(np.flatnonzero(seg["bridge"].to_numpy())[0])
    assert A[bridge].nnz == 0
    R = reach_matrices(A, np.array([bridge]), 3)
    assert int(R[3].sum()) == 1                        # ni a 3 saltos alcanza la calle de abajo


def test_close_but_disconnected_segments_are_not_neighbors():
    G = nx.MultiGraph()
    for n, (x, y) in {0: (0, 0), 1: (100, 0), 2: (0, 5), 3: (100, 5)}.items():
        G.add_node(n, x=float(x), y=float(y))
    G.add_edge(0, 1, 0, length=100.0)                  # dos calzadas paralelas a 5 m
    G.add_edge(2, 3, 0, length=100.0)
    A = dual_adjacency(segments_table(G))
    assert A.nnz == 0


def test_parallel_edges_count_once():
    G = nx.MultiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_edge(0, 1, 0, length=100.0)
    G.add_edge(0, 1, 1, length=120.0)                  # dos vías distintas entre las mismas esquinas
    seg = segments_table(G)
    A = dual_adjacency(seg)
    assert len(seg) == 2 and A.toarray().tolist() == [[0, 1], [1, 0]]


def test_directed_duplicates_collapse_to_one_segment():
    D = nx.MultiDiGraph()
    D.add_node(0, x=0.0, y=0.0)
    D.add_node(1, x=50.0, y=0.0)
    D.add_edge(0, 1, 0, length=50.0)
    D.add_edge(1, 0, 0, length=50.0)
    assert len(segments_table(D)) == 1


def test_snap_assigns_nearest_and_rejects_far_points():
    G = chain_graph(3)
    seg = segments_table(G)
    A = dual_adjacency(seg)
    node_xy = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    xy = np.array([[50.0, 4.0], [150.0, -2.0], [50.0, 80.0], [99.0, 1.0]])
    r = snap_to_segments(seg, A, xy, node_xy, max_dist=30.0)
    assert r.seg_idx.tolist()[:3] == [0, 1, -1]
    assert np.isclose(r.dist_m[0], 4.0)
    assert r.near_intersection[3] and not r.near_intersection[0]
    assert not r.ambiguous[3]                         # el otro candidato es adyacente (misma esquina)


def test_snap_flags_parallel_avenue_as_ambiguous():
    G = nx.MultiGraph()
    for n, (x, y) in {0: (0, 0), 1: (100, 0), 2: (0, 6), 3: (100, 6)}.items():
        G.add_node(n, x=float(x), y=float(y))
    G.add_edge(0, 1, 0, length=100.0)
    G.add_edge(2, 3, 0, length=100.0)
    seg = segments_table(G)
    r = snap_to_segments(seg, dual_adjacency(seg), np.array([[50.0, 3.2]]),
                         {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)})
    assert r.ambiguous[0]


def test_euclidean_count_matched_respects_counts_and_exclusions():
    mid = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [10, 0]], dtype=float)
    out = euclidean_count_matched(mid, np.array([0]), np.arange(5), np.array([2]), exclude=[{1}])
    assert out[0].tolist() == [2, 3]


def test_normalized_adjacency_without_edges_is_identity():
    A = dual_adjacency(segments_table(chain_graph()))
    I_hat = normalized_adjacency(A * 0, self_loops=True)
    assert np.allclose(I_hat.toarray(), np.eye(4))
    Ahat = normalized_adjacency(A).toarray()
    assert np.allclose(Ahat, Ahat.T) and np.all(Ahat >= 0)


def cross_graph() -> nx.MultiGraph:
    """Cruce en + : centro 0 en (0,0), brazos de 100 m a N/E/S/O, y un nodo 5 a 120 m al NE sin calle directa."""
    G = nx.MultiGraph()
    for n, (x, y) in {0: (0, 0), 1: (0, 100), 2: (100, 0), 3: (0, -100), 4: (-100, 0), 5: (85, 85)}.items():
        G.add_node(n, x=float(x), y=float(y))
    for n in (1, 2, 3, 4):
        G.add_edge(0, n, 0, length=100.0)
    G.add_edge(1, 5, 0, length=float(np.hypot(85, 15)))       # el punto NE solo se alcanza por el brazo N
    return G


def test_network_point_distances_follow_streets():
    from segment_graph import network_point_distances
    G = cross_graph()
    seg = segments_table(G)
    key = {tuple(sorted((u, v))): i for i, (u, v) in enumerate(zip(seg["u"], seg["v"]))}
    xy = np.array([[0, 10], [50, 0], [0, -60], [80, 88]], float)   # en N, E, S y en la calle 1–5
    sidx = np.array([key[(0, 1)], key[(0, 2)], key[(0, 3)], key[(1, 5)]])
    D = network_point_distances(G, seg, sidx, xy, max_m=500)
    assert abs(D[0][1] - 60) < 1e-6 and abs(D[0][2] - 70) < 1e-6
    assert D[0][3] > 150                                   # por la red: sube al nodo 1 y baja por 1–5
    assert abs(D[1][3] - (50 + 100 + np.hypot(85, 15) - np.hypot(5, 3))) < 1.0


def test_cross_neighbors_network_vs_sector():
    from segment_graph import cross_neighbors, euclidean_point_distances, network_point_distances
    G = cross_graph()
    seg = segments_table(G)
    key = {tuple(sorted((u, v))): i for i, (u, v) in enumerate(zip(seg["u"], seg["v"]))}
    # centro casi en el cruce; vecinos en N (70 m), E (60 m) y uno al NE a 45 m en línea recta pero lejos por la red
    xy = np.array([[0, 2], [0, 72], [62, 0], [32, 32], [0, 3]], float)
    sidx = np.array([key[(0, 1)], key[(0, 1)], key[(0, 2)], -1, key[(0, 1)]])
    sidx[3] = key[(1, 5)]
    Dn = network_point_distances(G, seg, sidx, xy, max_m=500)
    De = euclidean_point_distances(xy, 500)
    cn, ce = cross_neighbors(xy, Dn)[0], cross_neighbors(xy, De)[0]
    assert cn[0] == 1 and cn[1] == 2                      # N y E por la red
    assert ce[0] == 3 or ce[1] == 3                       # en línea recta gana el punto NE (cruza la manzana)
    assert 4 not in cn and 4 not in ce                    # a 1 m: mismo panorama, excluido
    assert cn[2] == -1 and cn[3] == -1                    # S y O vacíos
