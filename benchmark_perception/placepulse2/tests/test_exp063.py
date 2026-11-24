"""Pruebas de EXP-063 (cabezas) y de la evaluación por pares compartida con EXP-064."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from exp063_heads import (bt_fit, knn_scores, mcnemar_p, pair_correct, pair_logloss,  # noqa: E402
                          pigeonhole_ci, sigmoid)
from ranking_head import LinearRankingHead, MLPRankingHead  # noqa: E402
from utils import SPLITS  # noqa: E402


def test_pair_probability_is_complementary_and_in_unit_interval():
    rng = np.random.default_rng(0)
    g = rng.normal(size=50)
    i, j = rng.integers(0, 50, 200), rng.integers(0, 50, 200)
    p_ij, p_ji = sigmoid(g[i] - g[j]), sigmoid(g[j] - g[i])
    assert np.allclose(p_ij + p_ji, 1.0)
    assert ((p_ij >= 0) & (p_ij <= 1)).all()


def test_torch_heads_are_antisymmetric():
    torch.manual_seed(0)
    zl, zr = torch.randn(64, 768), torch.randn(64, 768)
    for head in (LinearRankingHead(768), MLPRankingHead(768)):
        head.eval()
        with torch.no_grad():
            assert torch.allclose(head(zl, zr), -head(zr, zl), atol=1e-6)
            # el logit del par es exactamente la diferencia de puntajes por imagen
            assert torch.allclose(head(zl, zr), head.score_images(zl) - head.score_images(zr), atol=1e-6)


def test_swapping_pair_order_does_not_change_accuracy():
    rng = np.random.default_rng(1)
    g = rng.normal(size=100)
    li, ri = rng.integers(0, 100, 500), rng.integers(0, 100, 500)
    y = rng.integers(0, 2, 500).astype(float)
    a = pair_correct(g, li, ri, y)
    b = pair_correct(g, ri, li, 1 - y)
    assert np.array_equal(a, b)


def test_ties_count_half():
    g = np.array([1.0, 1.0, 2.0])
    c = pair_correct(g, np.array([0, 2]), np.array([1, 0]), np.array([1.0, 1.0]))
    assert c.tolist() == [0.5, 1.0]


def test_logloss_matches_definition():
    g = np.array([0.0, 2.0])
    ll = pair_logloss(g, np.array([1]), np.array([0]), np.array([1.0]))
    assert np.isclose(ll, -np.log(sigmoid(2.0)))


def test_bt_recovers_order_of_true_strengths():
    rng = np.random.default_rng(2)
    lam_true = np.linspace(-2, 2, 40)
    li, ri = rng.integers(0, 40, 6000), rng.integers(0, 40, 6000)
    keep = li != ri
    li, ri = li[keep], ri[keep]
    y = (rng.random(len(li)) < sigmoid(lam_true[li] - lam_true[ri])).astype(float)
    rows, lam = bt_fit(li, ri, y)
    from scipy.stats import spearmanr
    assert spearmanr(lam, lam_true[rows]).correlation > 0.95


def test_pigeonhole_ci_covers_and_is_wider_than_pair_bootstrap_when_images_repeat():
    rng = np.random.default_rng(3)
    n_img = 60
    li, ri = rng.integers(0, n_img, 3000), rng.integers(0, n_img, 3000)
    # error correlacionado por imagen: imágenes "difíciles"
    hard = rng.random(n_img) < 0.3
    p = np.where(hard[li] | hard[ri], 0.5, 0.8)
    c = (rng.random(3000) < p).astype(float)
    est, lo, hi = pigeonhole_ci(c, None, li, ri, B=400)
    assert lo <= est <= hi
    from exp063_heads import pair_boot_ci
    plo, phi = pair_boot_ci(c, B=400)
    assert (hi - lo) > (phi - plo)          # ignorar la dependencia subestima la incertidumbre


def test_pigeonhole_delta_zero_for_identical_models():
    rng = np.random.default_rng(4)
    c = rng.integers(0, 2, 500).astype(float)
    li, ri = rng.integers(0, 50, 500), rng.integers(0, 50, 500)
    est, lo, hi = pigeonhole_ci(c, c.copy(), li, ri, B=100)
    assert est == lo == hi == 0.0
    assert mcnemar_p(c, c) == 1.0


def test_knn_matches_bruteforce():
    rng = np.random.default_rng(5)
    Xtr, Xq = rng.normal(size=(200, 16)), rng.normal(size=(30, 16))
    t = rng.normal(size=200)
    out = knn_scores(Xtr, t, Xq, [5])[5]
    A = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    Q = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
    ref = np.array([t[np.argsort(-(q @ A.T))[:5]].mean() for q in Q])
    assert np.allclose(out, ref)


def test_pca_and_threshold_use_train_rows_only():
    """El código de Context ajusta PCA y umbrales con train: aquí se verifica la propiedad con datos sintéticos."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(6)
    Xtr, Xte = rng.normal(size=(300, 20)), rng.normal(loc=5.0, size=(100, 20))
    p1 = PCA(5, random_state=0).fit(Xtr)
    p2 = PCA(5, random_state=0).fit(np.r_[Xtr, Xte])
    assert not np.allclose(np.abs(p1.components_), np.abs(p2.components_))   # incluir test cambiaría la proyección
    src = (SRC / "exp063_heads.py").read_text(encoding="utf-8")
    assert "fit(X[self.train_rows])" in src
    assert "np.percentile(lam, [25, 75])" in src and "bt_fit(self.li, self.ri, self.y)" in src


def test_image_disjoint_split_has_no_shared_images():
    import pandas as pd
    import pytest
    from utils import MANIFESTS
    if not (SPLITS / "image_disjoint.json").exists() or not (MANIFESTS / "pairs_safe.csv").exists():
        pytest.skip("requiere los splits y pairs_safe.csv (no versionados; ver split_dataset.py)")
    split = json.loads((SPLITS / "image_disjoint.json").read_text())
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str, usecols=["left", "right"])
    imgs = {k: set(pairs.iloc[split[k]][["left", "right"]].to_numpy().ravel()) for k in ("train", "val", "test")}
    assert not (imgs["train"] & imgs["test"])
    assert not (imgs["train"] & imgs["val"])
    assert not (imgs["val"] & imgs["test"])
