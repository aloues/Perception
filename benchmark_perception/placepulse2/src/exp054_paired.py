"""
EXP-054 — Comparación pareada de las CNN contra DINOv2 y ImageNet ViT (mismos pares de test, mismo probe).
Δ exactitud con bootstrap de pares (2,000) y McNemar exacto/χ². Uso: python exp054_paired.py → outputs/exp054/paired.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from utils import OUTPUTS, log

RUNS = OUTPUTS / "runs"
OUT = OUTPUTS / "exp054"
CASES = (("paper_protocol", "test"), ("image_disjoint", "test"), ("city_holdout", "test_strict"))
PAIRS = (("resnet50_in1k", "dinov2_vit_b14"), ("convnext_tiny_in1k", "dinov2_vit_b14"),
         ("resnet50_in1k", "imagenet_vit_b16"), ("convnext_tiny_in1k", "imagenet_vit_b16"),
         ("resnet50_in1k", "ours_simclr_vit_b16"), ("convnext_tiny_in1k", "resnet50_in1k"))


def correct(enc: str, proto: str, split: str) -> np.ndarray:
    p = pd.read_csv(RUNS / f"{enc}__{proto}__linear__seed42" / "predictions.csv")
    p = p[p["split"] == split]
    return ((p["logit"] > 0).astype(int) == p["label"]).to_numpy(), p[["left", "right"]].to_numpy()


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for proto, split in CASES:
        for a, b in PAIRS:
            ca, ka = correct(a, proto, split)
            cb, kb = correct(b, proto, split)
            assert (ka == kb).all(), "pares distintos"
            d = ca.astype(float) - cb
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            n01, n10 = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue
            rows.append(dict(protocol=proto, model=a, baseline=b, n=len(d), acc_model=ca.mean(), acc_base=cb.mean(),
                             delta=d.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                             mcnemar_p=p))
            log(f"{proto:15s} {a:20s} − {b:20s} Δ={100 * d.mean():+.2f} pts "
                f"[{100 * np.percentile(boot, 2.5):+.2f}, {100 * np.percentile(boot, 97.5):+.2f}] p={p:.1e}")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
