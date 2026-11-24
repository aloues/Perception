"""G0 (ego-k LOO), G1 (GCN / control sin aristas) y G2 (CRF gaussiano LOO) sobre un grafo de juguete."""

import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from exp042_pp_segment_graph_models import GCNScore, crf_loo_context, ego_context  # noqa: E402
from segment_graph import normalized_adjacency  # noqa: E402


def chain_dual(n: int) -> sp.csr_matrix:
    A = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
    return A


def test_ego_context_excludes_own_image_and_respects_k():
    A = chain_dual(4)                                   # segmentos 0-1-2-3
    seg_of_img = np.array([0, 0, 1, 3, -1])             # imagen 4 sin segmento
    Z = np.eye(5, dtype=np.float32)
    c0, n0 = ego_context(Z, seg_of_img, A, 0)
    assert np.allclose(c0[0], Z[1]) and n0[0] == 1      # ego-0: solo la otra imagen del segmento
    assert np.allclose(c0[2], 0) and n0[2] == 0         # sola en su segmento → sin contexto
    c1, n1 = ego_context(Z, seg_of_img, A, 1)
    assert np.allclose(c1[2], (Z[0] + Z[1]) / 2)        # ego-1 de seg 1 = segs 0,1,2 sin la propia
    assert n1[3] == 0 and np.allclose(c1[4], 0)         # seg 3 no tiene vecinos con imágenes a 1 salto
    c2, _ = ego_context(Z, seg_of_img, A, 2)
    assert np.allclose(c2[3], Z[2])                     # a 2 saltos llega a seg 1


def brute_force_crf_loo(f, seg_of_img, A, lam, eps=1e-3):
    n_seg = A.shape[0]
    Ad = A.toarray()
    L = np.diag(Ad.sum(1)) - Ad
    out = np.zeros(len(f))
    for i in np.flatnonzero(seg_of_img >= 0):
        keep = (seg_of_img >= 0) & (np.arange(len(f)) != i)
        cnt = np.bincount(seg_of_img[keep], minlength=n_seg).astype(float)
        b = np.bincount(seg_of_img[keep], weights=f[keep], minlength=n_seg)
        y = np.linalg.solve(np.diag(cnt + eps) + lam * L, b)
        out[i] = y[seg_of_img[i]]
    return out


def test_crf_sherman_morrison_matches_brute_force():
    rng = np.random.default_rng(0)
    A = chain_dual(6)
    seg_of_img = np.array([0, 0, 2, 3, 3, 3, 5, -1])
    f = rng.normal(size=len(seg_of_img))
    for lam in (0.0, 0.5, 3.0):
        fast = crf_loo_context(f, seg_of_img, A, lam)
        slow = brute_force_crf_loo(f, seg_of_img, A, lam)
        assert np.allclose(fast, slow, atol=1e-8), lam


def test_crf_lambda_zero_is_loo_segment_mean():
    A = chain_dual(3)
    seg_of_img = np.array([0, 0, 0, 1])
    f = np.array([1.0, 2.0, 6.0, 5.0])
    c = crf_loo_context(f, seg_of_img, A, 0.0, eps=1e-9)
    assert np.allclose(c[:3], [4.0, 3.5, 1.5], atol=1e-6) and abs(c[3]) < 1e-6


def test_gcn_noedge_uses_only_own_segment():
    torch.manual_seed(0)
    Z = np.zeros((3, 2), np.float32)
    seg_local = np.array([0, 1, -1])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    A = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    m_edge = GCNScore(Z, seg_local, X, normalized_adjacency(A))
    m_none = GCNScore(Z, seg_local, X, sp.identity(2, format="csr"))
    m_none.load_state_dict(m_edge.state_dict())
    m_edge.eval(), m_none.eval()
    s_edge = m_edge(torch.arange(3)).detach().numpy()
    s_none = m_none(torch.arange(3)).detach().numpy()
    assert s_edge[2] == 0 and s_none[2] == 0            # sin segmento → sin término de contexto
    X2 = X.copy()
    X2[1] = [5.0, 5.0]                                  # cambiar el VECINO del segmento 0
    m_none2 = GCNScore(Z, seg_local, X2, sp.identity(2, format="csr"))
    m_none2.load_state_dict(m_none.state_dict())
    m_none2.eval()
    assert np.isclose(m_none2(torch.arange(1)).item(), s_none[0])   # sin aristas, el vecino no influye


def test_graph_experiments_never_use_crime_as_perception_target():
    """El target de G0/G1/G2 son los votos humanos (pairs_safe.csv); ningún script de grafo lee crimen."""
    src = (SRC / "exp042_pp_segment_graph_models.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]                       # sin el docstring (que sí menciona el crimen)
    assert "load_pairs" in code
    assert not re.search(r"crimen_distritos|labels_20\d\d|dualgmm|lima_views|eigen_features", code)
    lima = (SRC / "exp041_lima_segment_graph.py").read_text(encoding="utf-8")
    assert "train_model" not in lima and "load_pairs" not in lima       # Lima: sin entrenamiento supervisado
    from lima_views import spots_table
    import pandas as pd
    V = pd.DataFrame({"district": ["b"] * 2, "spot_id": ["1.0"] * 2, "heading": [0, 30], "latitude": [-12.1] * 2,
                      "longitude": [-77.0] * 2, "coord_ok": [True] * 2, "MODALIDAD": ["ROBO"] * 2})
    assert "MODALIDAD" not in spots_table(V).columns
