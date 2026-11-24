"""Selección de vistas, BoVW y divergencias."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_repr import (BoVW, angular_separation, effective_rank, fuse, geometric_overlap,  # noqa: E402
                         histograms, js_divergence, kl_divergence, select_random, select_stratified)

H12 = np.arange(0, 360, 30)


def test_random_selection_is_deterministic_per_seed_and_distinct():
    a = select_random(H12, 4, np.random.default_rng(7))
    b = select_random(H12, 4, np.random.default_rng(7))
    assert a.tolist() == b.tolist() and len(set(a)) == 4 and set(a) <= set(H12)
    draws = {tuple(select_random(H12, 4, np.random.default_rng(s))) for s in range(20)}
    assert len(draws) > 1                              # la semilla sí cambia el sorteo


def test_random_selection_with_fewer_views_returns_all():
    assert select_random(np.array([0, 90, 180]), 4, np.random.default_rng(0)).tolist() == [0, 90, 180]


def test_stratified_selection_tiles_the_circle():
    for s in range(10):
        h = select_stratified(H12, 4, np.random.default_rng(s))
        assert len(h) == 4 and set(np.diff(h)) == {90} and h[0] in (0, 30, 60)


def test_angular_separation_wraps():
    assert angular_separation(330, 0) == 30 and angular_separation(0, 180) == 180


def test_geometric_overlap_fov90():
    assert geometric_overlap(90, 30) == pytest.approx(2 / 3)
    assert geometric_overlap(90, 90) == 0.0


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    same = np.tile(rng.normal(size=(1, 64)), (12, 1))
    indep = rng.normal(size=(12, 512))
    assert effective_rank(same) == 1.0
    assert effective_rank(indep) > 9


def test_bovw_histograms_have_dimension_k_and_l1_norm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    bv = BoVW(5, seed=0).fit(X[:200])
    w = bv.assign(X[200:])
    H = histograms(w, np.repeat(np.arange(10), 10), 5)
    assert H.shape == (10, 5) and np.allclose(H.sum(axis=1), 1.0)
    assert bv.fitted_on == 200                       # solo se ajustó con los de entrenamiento


def test_bovw_is_fit_only_on_training_descriptors():
    """Un descriptor de test alejado no puede crear su propio centroide si no estaba en train."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(200, 4))
    test_outlier = np.full((5, 4), 50.0)
    bv = BoVW(3, seed=0).fit(train)
    assert np.all(np.abs(bv.km.cluster_centers_) < 10)
    assert len(set(bv.assign(test_outlier))) == 1


def test_divergences_numerically_stable_with_zeros():
    P = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    Q = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    kl = kl_divergence(P, Q)
    js = js_divergence(P, Q)
    assert np.all(np.isfinite(kl)) and np.all(np.isfinite(js))
    assert js[0] == pytest.approx(np.log(2)) and js[1] == pytest.approx(0.0)
    assert kl[1] == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(P[:1], Q[:1] * 0 + [[0.2, 0.3, 0.5]]) != pytest.approx(
        kl_divergence(np.array([[0.2, 0.3, 0.5]]), P[:1]))      # KL es asimétrica


def test_fusion_blocks_are_unit_norm():
    F = fuse(np.ones((2, 4)) * 3, np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(F[:, :4], axis=1), 1) and np.allclose(np.linalg.norm(F[:, 4:], axis=1), 1)
