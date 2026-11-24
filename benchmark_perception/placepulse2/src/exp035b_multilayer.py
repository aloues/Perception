"""
EXP-035 fase B — Protocolo de linear probing del paper de DINOv2 (Oquab et al. 2023): CLS + media de parches y capas
intermedias, con encoder congelado. Los tokens ya existen (`extract_dinov2_tokens.py`: cls_L12, avg_L12, cls_L9..L11).

  --stage build    escribe outputs/features/{dinov2_cls_avg, dinov2_cls_avg_L9_11}/ (L2 por bloque, luego L2 global)
  --stage compare  Δ pareado vs DINOv2 CLS (mismo probe fijo de RQ1) → outputs/exp035/phaseB_paired.csv
Los probes se entrenan con train_probe.py (protocolo FIJO de RQ1: lineal, lr 1e-3, wd 1e-4, seed 42), ver
run_exp035b.sh. Criterio (registro): Δ > 0 con IC que excluye 0 en image_disjoint y city_holdout.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from utils import FEATURES, OUTPUTS, log

SRC = FEATURES / "dinov2_tokens"
VARIANTS = {"dinov2_cls_avg": [0, 1], "dinov2_cls_avg_L9_11": [0, 1, 2, 3, 4]}
CASES = (("paper_protocol", "test"), ("image_disjoint", "test"), ("city_holdout", "test_strict"))


def build() -> None:
    T = np.load(SRC / "features.npy", mmap_mode="r")
    ids = json.loads((SRC / "index.json").read_text())["location_ids"]
    for name, parts in VARIANTS.items():
        out = FEATURES / name
        out.mkdir(parents=True, exist_ok=True)
        X = np.empty((T.shape[0], 768 * len(parts)), np.float32)
        for a in range(0, T.shape[0], 8192):
            B = np.asarray(T[a:a + 8192][:, parts], np.float32)
            B /= np.linalg.norm(B, axis=-1, keepdims=True)
            B = B.reshape(len(B), -1)
            X[a:a + len(B)] = B / np.linalg.norm(B, axis=1, keepdims=True)
        np.save(out / "features.npy", X)
        (out / "index.json").write_text(json.dumps({"encoder": f"DINOv2 ViT-B/14 tokens {parts} (L2 por bloque + L2)",
                                                    "embed_dim": X.shape[1], "n_images": len(ids), "l2_normalized": True,
                                                    "location_ids": ids}), encoding="utf-8")
        log(f"{name}: {X.shape}")


def compare() -> None:
    rng = np.random.default_rng(42)
    rows = []

    def correct(enc, proto, split):
        p = pd.read_csv(OUTPUTS / "runs" / f"{enc}__{proto}__linear__seed42" / "predictions.csv")
        p = p[p["split"] == split]
        return ((p["logit"] > 0).astype(int) == p["label"]).to_numpy(), p[["left", "right"]].to_numpy()

    for proto, split in CASES:
        cb, kb = correct("dinov2_vit_b14", proto, split)
        for name in VARIANTS:
            ca, ka = correct(name, proto, split)
            assert (ka == kb).all()
            d = ca.astype(float) - cb
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            n01, n10 = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
            rows.append(dict(protocol=proto, model=name, acc=ca.mean(), acc_cls=cb.mean(), delta=d.mean(),
                             ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                             mcnemar_p=binomtest(n01, n01 + n10, 0.5).pvalue))
            log(f"{proto:15s} {name:22s} acc={ca.mean():.4f} Δ={100 * d.mean():+.2f} pts "
                f"[{100 * np.percentile(boot, 2.5):+.2f}, {100 * np.percentile(boot, 97.5):+.2f}]")
    pd.DataFrame(rows).to_csv(OUTPUTS / "exp035" / "phaseB_paired.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "compare"], required=True)
    build() if ap.parse_args().stage == "build" else compare()
