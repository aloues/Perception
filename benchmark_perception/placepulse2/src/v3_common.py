"""
Utilidades compartidas del programa V3 (EXP-068/069/070), 14/11/2026.

Reutiliza lo ya validado en EXP-063 (carga de features, cabeza lineal-BT con la receta de train_probe.py, bootstrap de
imágenes 'pigeonhole') y añade:
  - reconstrucción de la partición de IMÁGENES de image_disjoint (mismo RNG que split_dataset.py, verificada);
  - asignación de empates y de votos de otros atributos a esa partición;
  - bootstrap pigeonhole genérico para estadísticos arbitrarios (AUROC, ECE, ...);
  - métricas de calibración y de abstención.
Pre-registro: docs/EXPERIMENT_REGISTRY.md, "Programa V3 (14/11/2026)".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MANIFESTS, SPLITS, log  # noqa: E402
from exp063_heads import _torch_pairwise, load_Z, pigeonhole_ci  # noqa: E402,F401

ENCODER = "dinov2_vit_b14"
SEED = 42
REG_ACC = {"image_disjoint": 0.6785, "city_holdout": 0.6715}  # registradas (outputs/runs/*/metrics.json)
LINEAR_PARAMS = {"lr": 1e-3, "wd": 1e-4, "epochs": 50, "patience": 5}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    return df


def load_ties() -> pd.DataFrame:
    return pd.read_csv(MANIFESTS / "pairs_safe_ties.csv", dtype=str)


def load_split(protocol: str) -> dict:
    return json.loads((SPLITS / f"{protocol}.json").read_text())


def image_partition(df: pd.DataFrame, seed: int = SEED) -> dict[str, str]:
    """Reproduce split_dataset.split_image_disjoint: misma lista de imágenes, mismo RNG, mismas fracciones."""
    from split_dataset import IMG_TRAIN_FRAC, IMG_VAL_FRAC
    images = pd.unique(pd.concat([df["left"], df["right"]], ignore_index=True))
    perm = np.random.default_rng(seed).permutation(len(images))
    n_tr, n_va = int(len(images) * IMG_TRAIN_FRAC), int(len(images) * IMG_VAL_FRAC)
    part = np.empty(len(images), dtype=object)
    part[perm[:n_tr]] = "train"
    part[perm[n_tr:n_tr + n_va]] = "val"
    part[perm[n_tr + n_va:]] = "test"
    return dict(zip(images, part))


def verify_partition(df: pd.DataFrame, assign: dict, split: dict) -> None:
    pl, pr = df["left"].map(assign), df["right"].map(assign)
    same = pl == pr
    for p in ("train", "val", "test"):
        rebuilt = df.index[same & (pl == p)].tolist()
        if rebuilt != split[p]:
            raise RuntimeError(f"La partición reconstruida no coincide con image_disjoint.json en {p}")
    log("partición de imágenes reconstruida = image_disjoint.json (igualdad exacta de índices en train/val/test)")


def assign_rows(left: pd.Series, right: pd.Series, assign: dict) -> pd.Series:
    """Partición de un par = la de sus imágenes si coinciden; si no, NaN (par descartado, como en image_disjoint)."""
    pl, pr = left.map(assign), right.map(assign)
    return pl.where(pl == pr)


def pair_idx(left: pd.Series, right: pd.Series, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = left.isin(row).to_numpy() & right.isin(row).to_numpy()
    li = left[keep].map(row).to_numpy().astype(np.int64)
    ri = right[keep].map(row).to_numpy().astype(np.int64)
    return li, ri, keep


def fit_linear_bt(X, li, ri, y, vli, vri, vy, seed: int = SEED):
    """Cabeza lineal-BT con la receta de train_probe.py (vía exp063_heads._torch_pairwise). Devuelve g (puntaje por fila)."""
    g, info = _torch_pairwise("linear", None, li, ri, y.astype(np.float32), vli, vri, vy.astype(np.float32),
                              X, LINEAR_PARAMS, seed)
    return g, info


# ---------------------------------------------------------------- bootstrap pigeonhole genérico
def pigeonhole_stat(stat_fn, li, ri, B: int = 1000, seed: int = SEED, chunk: int = 25):
    """IC 95 % de un estadístico ponderado stat_fn(w) (w = peso por par) remuestreando IMÁGENES con pesos Poisson(1);
    peso del par = w_i·w_j (Owen 2007). stat_fn(None) = estimación puntual."""
    imgs, inv = np.unique(np.r_[li, ri], return_inverse=True)
    a, b = inv[: len(li)], inv[len(li):]
    rng = np.random.default_rng(seed)
    st = []
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        W = rng.poisson(1.0, size=(m, len(imgs))).astype(np.float32)
        for k in range(m):
            st.append(stat_fn(W[k, a] * W[k, b]))
    st = np.asarray(st, dtype=float)
    st = st[np.isfinite(st)]
    return float(stat_fn(None)), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def weighted_auc(score: np.ndarray, label: np.ndarray, w: np.ndarray | None = None) -> float:
    """AUROC ponderado (empates de score con medio crédito)."""
    if w is None:
        w = np.ones_like(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    s, l, ww = score[order], label[order].astype(bool), w[order]
    # agrupar empates de score
    uniq, start = np.unique(s, return_index=True)
    end = np.r_[start[1:], len(s)]
    wp = np.add.reduceat(ww * l, start)
    wn = np.add.reduceat(ww * ~l, start)
    neg_before = np.cumsum(wn) - wn
    num = (wp * (neg_before + 0.5 * wn)).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else np.nan


def ece_equal_mass(p: np.ndarray, y: np.ndarray, bins: int = 15, w: np.ndarray | None = None) -> float:
    """ECE de la predicción binaria simétrica: se usa la confianza max(p, 1−p) frente a acierto, bins de igual masa."""
    if w is None:
        w = np.ones_like(p)
    conf = np.maximum(p, 1 - p)
    corr = ((p >= 0.5) == (y == 1)).astype(float)
    order = np.argsort(conf)
    edges = np.linspace(0, len(conf), bins + 1).astype(int)
    tot, e = w.sum(), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[lo:hi]
        ws = w[idx].sum()
        if ws > 0:
            e += ws / tot * abs((w[idx] * corr[idx]).sum() / ws - (w[idx] * conf[idx]).sum() / ws)
    return float(e)


def nll(p, y, w=None):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    v = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(v, weights=w))


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda t: nll(sigmoid(logit / t), y), bounds=(0.2, 5.0), method="bounded")
    return float(r.x)


def risk_coverage(conf: np.ndarray, correct: np.ndarray, w: np.ndarray | None = None):
    """Curva riesgo–cobertura ordenando por confianza descendente. Devuelve cobertura, exactitud acumulada, AURC."""
    if w is None:
        w = np.ones_like(conf, dtype=float)
    order = np.argsort(-conf, kind="mergesort")
    cw = np.cumsum(w[order])
    cc = np.cumsum((w * correct)[order])
    cov = cw / cw[-1]
    acc = cc / cw
    aurc = float(np.trapz(1 - acc, cov))
    return cov, acc, aurc
