"""Pruebas de las utilidades V3 (EXP-068/069/070). Rápidas, sin features: datos sintéticos.
Uso: ../../.venv/Scripts/python.exe -m pytest tests/test_v3.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp068_ambiguity import davidson_probs, fit_davidson, voter_dispersion  # noqa: E402
from v3_common import ece_equal_mass, risk_coverage, sigmoid, weighted_auc  # noqa: E402


def test_weighted_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=2000), rng.integers(0, 2, 2000)
    s[:50] = 0.0  # empates de puntaje
    assert abs(weighted_auc(s, y) - roc_auc_score(y, s)) < 1e-9
    w = rng.integers(1, 3, 2000).astype(float)
    assert abs(weighted_auc(s, y, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-9


def test_davidson_probabilities_sum_to_one_and_symmetry():
    d = np.linspace(-4, 4, 41)
    pi, pj, pt = davidson_probs(d, 0.3)
    assert np.allclose(pi + pj + pt, 1)
    assert np.allclose(pi, pj[::-1])          # antisimetría: d ↔ −d
    assert pt.argmax() == 20                  # el empate es máximo cuando d = 0


def test_fit_davidson_recovers_parameters():
    rng = np.random.default_rng(1)
    d0 = rng.normal(0, 1.5, 40000)
    pi, pj, pt = davidson_probs(1.2 * d0, 0.4)
    u = rng.random(len(d0))
    out = np.where(u < pi, 1, np.where(u < pi + pj, 0, 2))
    beta, nu = fit_davidson(d0, out)
    assert abs(beta - 1.2) < 0.06 and abs(nu - 0.4) < 0.03


def test_ece_calibrated_is_small():
    rng = np.random.default_rng(2)
    p = sigmoid(rng.normal(0, 1.5, 50000))
    y = (rng.random(len(p)) < p).astype(int)
    assert ece_equal_mass(p, y) < 0.01


def test_voter_dispersion_null_is_near_one():
    rng = np.random.default_rng(3)
    n_v, per = 300, 20
    voter = np.repeat(np.arange(n_v), per).astype(str)
    p = sigmoid(rng.normal(0, 1.5, n_v * per))
    y = (rng.random(len(p)) < p).astype(int)   # todos el mismo observador calibrado
    out, _ = voter_dispersion_safe(voter, p, y)
    assert 0.8 < out["phi"] < 1.25


def voter_dispersion_safe(voter, p, y):
    import exp068_ambiguity as m
    m.VOTER_LEFT = pd.Series(0.5, index=pd.unique(voter))  # requerido por el chequeo exploratorio de lado
    return voter_dispersion(voter, p, y, boot=200)


def test_risk_coverage_monotone_on_informative_confidence():
    rng = np.random.default_rng(4)
    conf = rng.random(5000)
    correct = (rng.random(5000) < 0.5 + 0.45 * conf).astype(float)
    cov, acc, aurc = risk_coverage(conf, correct)
    assert acc[int(0.2 * len(acc))] > acc[-1]
    assert aurc < 1 - correct.mean()
