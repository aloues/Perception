"""
EXP-061 — Compara el fine-tuning parcial (K últimos bloques de DINOv2) con DINOv2 congelado + misma cabeza lineal BT
(mismos pares de test, mismo orden). Δ pareado con IC 95 % por bootstrap de IMÁGENES (pigeonhole, B = 2000, igual que
EXP-063/064) y de pares, McNemar exacto, log-loss.

Entrada: deliverables/colab/results/finetune_dinov2_K*__<protocolo>__seed42/{predictions.csv, metrics.json}
         (corrida local: src/exp061_finetune_local.py) y outputs/runs/dinov2_vit_b14__<protocolo>__linear__seed42/.
Salida: outputs/exp061/paired_vs_frozen.csv (+ .md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import mcnemar_p, pair_boot_ci, pigeonhole_ci  # noqa: E402
from utils import OUTPUTS, log  # noqa: E402

RES = Path(__file__).resolve().parents[3] / "deliverables" / "colab" / "results"
OUT = OUTPUTS / "exp061"


def logloss(logit: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, logit) - y * logit))


def main() -> None:
    rows = []
    for d in sorted(RES.glob("finetune_dinov2_K*__*")):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        proto, split = m["protocol"], m["test_split"]
        ft = pd.read_csv(d / "predictions.csv", dtype={"left": str, "right": str})
        fr = pd.read_csv(OUTPUTS / "runs" / f"dinov2_vit_b14__{proto}__linear__seed42" / "predictions.csv",
                         dtype={"left": str, "right": str})
        fr = fr[fr["split"] == split].reset_index(drop=True)
        assert len(ft) == len(fr) and (ft[["left", "right", "label"]].to_numpy() ==
                                       fr[["left", "right", "label"]].to_numpy()).all(), "pares de test distintos"
        ids = pd.Index(pd.unique(np.r_[fr["left"], fr["right"]]))
        li, ri = ids.get_indexer(fr["left"]), ids.get_indexer(fr["right"])
        a = ((ft["logit"] > 0).astype(int) == ft["label"]).to_numpy(float)
        b = ((fr["logit"] > 0).astype(int) == fr["label"]).to_numpy(float)
        d_img, lo, hi = pigeonhole_ci(a, b, li, ri, B=2000)
        plo, phi = pair_boot_ci(a - b)
        _, alo, ahi = pigeonhole_ci(a, None, li, ri, B=2000)
        y = ft["label"].to_numpy(float)
        rows.append(dict(run=d.name, protocol=proto, test_split=split, K=m["K"], n=len(a),
                         acc_finetune=a.mean(), acc_ft_img_lo=alo, acc_ft_img_hi=ahi, acc_frozen=b.mean(),
                         delta=d_img, delta_img_lo=lo, delta_img_hi=hi, delta_pair_lo=plo, delta_pair_hi=phi,
                         mcnemar_p=mcnemar_p(a, b), logloss_finetune=logloss(ft["logit"].to_numpy(float), y),
                         logloss_frozen=logloss(fr["logit"].to_numpy(float), y), best_val_acc=m["best_val_acc"],
                         epochs=m["cfg"]["epochs"], n_val_evals=len(m.get("log", []))))
        log(f"{d.name}: {a.mean():.4f} vs congelado {b.mean():.4f}  Δ={100 * d_img:+.2f} pts "
            f"[{100 * lo:+.2f}, {100 * hi:+.2f}] (IC imágenes)")
    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paired_vs_frozen.csv", index=False, float_format="%.5f")
    with open(OUT / "paired_vs_frozen.md", "w", encoding="utf-8") as f:
        f.write("| Protocolo | K | n | Fine-tuning [IC img] | Congelado | Δ (pts) [IC img] | McNemar p | log-loss FT / cong. |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.protocol} | {r.K} | {r.n:,} | {r.acc_finetune:.4f} [{r.acc_ft_img_lo:.4f}, {r.acc_ft_img_hi:.4f}] "
                    f"| {r.acc_frozen:.4f} | {100 * r.delta:+.2f} [{100 * r.delta_img_lo:+.2f}, {100 * r.delta_img_hi:+.2f}] "
                    f"| {r.mcnemar_p:.1e} | {r.logloss_finetune:.4f} / {r.logloss_frozen:.4f} |\n")


if __name__ == "__main__":
    main()
