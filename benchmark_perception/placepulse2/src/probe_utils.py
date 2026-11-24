"""
Utilidades compartidas por EXP-028..031: probe lineal BT entrenado en proceso (mismos hiperparámetros que
train_probe.py / EXP-001) para poder PUNTUAR cada imagen, votos por imagen y celdas espaciales.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_all
from ranking_head import build_head
from train_probe import predict, run_epoch, to_tensors
from utils import MANIFESTS, SPLITS, set_seed

SEED = 42


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    pairs["label"] = pairs["label"].astype(int)
    return pairs


def load_split(protocol: str) -> dict:
    return json.loads((SPLITS / f"{protocol}.json").read_text())


def train_scores(X: np.ndarray, ids: list[str], pairs: pd.DataFrame, split: dict,
                 test_key: str = "test", seed: int = SEED) -> tuple[np.ndarray, dict]:
    """Probe lineal BT en `split['train']`, early stopping en val. Devuelve puntaje por imagen y
    métricas de test (`test_key`). Mismos hiperparámetros que train_probe.py."""
    set_seed(seed)
    id_to_row = {lid: i for i, lid in enumerate(ids)}
    X = X.astype(np.float32)
    zl, zr, y, _ = to_tensors(pairs, split["train"], X, id_to_row, "cpu")
    vl, vr, vy, _ = to_tensors(pairs, split["val"], X, id_to_row, "cpu")
    head = build_head("linear", embed_dim=X.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for _ in range(50):
        run_epoch(head, zl, zr, y, opt, 4096)
        v = run_epoch(head, vl, vr, vy, None, 4096)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    head.load_state_dict(state)
    tl, tr, ty, _ = to_tensors(pairs, split[test_key], X, id_to_row, "cpu")
    metrics = evaluate_all(predict(head, tl, tr), ty.numpy())
    s = head.score_images(torch.from_numpy(X)).numpy()
    return s, {"metrics": metrics, "weights": head.score.weight.detach().numpy().ravel()}


def votes(pairs: pd.DataFrame, idx: list[int], keep: set | None = None) -> pd.DataFrame:
    """Un registro por (imagen, voto): id, win ∈ {0,1}. `keep` restringe a esas imágenes."""
    sub = pairs.iloc[idx]
    y = sub["label"].to_numpy()
    v = pd.DataFrame({"id": np.r_[sub["left"], sub["right"]], "win": np.r_[y, 1 - y]})
    return v[v["id"].isin(keep)] if keep is not None else v


def local_xy(man: pd.DataFrame) -> np.ndarray:
    """Proyección equirectangular local por ciudad (la misma de EXP-021/026)."""
    lat0 = man.groupby("city")["lat"].transform("mean").to_numpy()
    return np.c_[man["lon"].to_numpy() * 111320 * np.cos(np.radians(lat0)), man["lat"].to_numpy() * 110540]


def cell_ids(man: pd.DataFrame, side_m: float, offset_m: float = 0.0) -> pd.Series:
    xy = local_xy(man) + offset_m
    return (man["city"] + "|" + np.floor(xy[:, 0] / side_m).astype(int).astype(str) + "|"
            + np.floor(xy[:, 1] / side_m).astype(int).astype(str))
