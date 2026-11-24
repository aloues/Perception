"""
Metricas del benchmark (seccion 25).

Principal   : pairwise_accuracy
Secundarias : log loss, balanced accuracy, Brier score
Incertidumbre: bootstrap 95% CI sobre los pares de test

Para city_holdout se reporta ademas accuracy por ciudad, con macro-average,
weighted-average y tamano de muestra por ciudad — necesario porque el dataset
tiene un desbalance por ciudad de ~9.5x.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def pairwise_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(((logits > 0).astype(int) == y).mean())


def log_loss(logits: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(_sigmoid(logits), eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_score(logits: np.ndarray, y: np.ndarray) -> float:
    return float(((_sigmoid(logits) - y) ** 2).mean())


def balanced_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = (logits > 0).astype(int)
    accs = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            accs.append(float((pred[mask] == cls).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def bootstrap_ci(
    logits: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """IC percentil del bootstrap para la pairwise accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return (float("nan"), float("nan"))
    correct = (logits > 0).astype(int) == y
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        stats[b] = correct[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def evaluate_all(logits: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    lo, hi = bootstrap_ci(logits, y, seed=seed)
    return {
        "n": int(len(y)),
        "pairwise_accuracy": pairwise_accuracy(logits, y),
        "ci95_low": lo,
        "ci95_high": hi,
        "log_loss": log_loss(logits, y),
        "balanced_accuracy": balanced_accuracy(logits, y),
        "brier_score": brier_score(logits, y),
        "label_balance": float(y.mean()),
    }


def evaluate_per_city(
    logits: np.ndarray,
    y: np.ndarray,
    city_left: np.ndarray,
    city_right: np.ndarray,
    cities: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Un par se asigna a una ciudad si cualquiera de sus dos extremos pertenece a
    ella; un par entre dos ciudades cuenta para ambas.
    """
    if cities is None:
        cities = sorted(set(city_left) | set(city_right))

    per_city = {}
    for c in cities:
        mask = (city_left == c) | (city_right == c)
        if not mask.any():
            continue
        per_city[c] = evaluate_all(logits[mask], y[mask], seed=seed)

    accs = [v["pairwise_accuracy"] for v in per_city.values()]
    sizes = [v["n"] for v in per_city.values()]
    total = sum(sizes) or 1

    return {
        "per_city": per_city,
        "macro_average": float(np.mean(accs)) if accs else float("nan"),
        "weighted_average": float(
            sum(a * n for a, n in zip(accs, sizes)) / total
        ) if accs else float("nan"),
        "n_cities": len(per_city),
        "min_city_n": int(min(sizes)) if sizes else 0,
        "max_city_n": int(max(sizes)) if sizes else 0,
    }
