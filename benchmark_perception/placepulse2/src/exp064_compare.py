"""
EXP-064 — Comparación pareada de encoders con la MISMA cabeza lineal-BT (train_probe.py), mismos pares y seed.

Lee outputs/runs/<encoder>__<protocolo>__linear__seed42/predictions.csv, alinea por (split, left, right, label)
y calcula para cada encoder vs DINOv2: exactitud, Δ, IC 95 % por bootstrap de imágenes (pigeonhole) y de pares, McNemar.

Uso:  python src/exp064_compare.py            Salida: outputs/exp064/paired_vs_dinov2.csv (+ .md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import mcnemar_p, pair_boot_ci, pigeonhole_ci  # noqa: E402
from utils import OUTPUTS, RUNS, log  # noqa: E402

BASE = "dinov2_vit_b14"
ENCODERS = ["dinov3_vitb16_cls", "dinov3_vitb16_meanpatch", "ijepa_vith14_meanpatch",
            "imagenet_vit_b16", "ours_simclr_vit_b16", "lejepa_resnet18_pilot", "resnet18_random_init",
            # EXP-067 (13/11 noche): datos, JEPA vigente y escala
            "ijepa_vith14_in22k_meanpatch", "vjepa2_vitl16_meanpatch", "dinov2_vitl14_cls", "dinov2_vitl14_meanpatch"]
PROTOCOLS = {"image_disjoint": "test", "city_holdout": "test_strict", "paper_protocol": "test"}
OUT = OUTPUTS / "exp064"


def load(enc: str, prot: str, key: str) -> pd.DataFrame | None:
    p = RUNS / f"{enc}__{prot}__linear__seed42" / "predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"left": str, "right": str})
    return d[d["split"] == key].reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for prot, key in PROTOCOLS.items():
        b = load(BASE, prot, key)
        if b is None:
            continue
        ids = pd.Index(pd.unique(np.r_[b["left"], b["right"]]))
        li, ri = ids.get_indexer(b["left"]), ids.get_indexer(b["right"])
        cb = ((b["logit"] > 0).astype(int) == b["label"]).to_numpy(float)
        rows.append({"protocol": prot, "test_key": key, "encoder": BASE, "n": len(b), "acc": cb.mean(),
                     **dict(zip(["acc_img_lo", "acc_img_hi"], pigeonhole_ci(cb, None, li, ri, B=1000)[1:]))})
        for enc in ENCODERS:
            e = load(enc, prot, key)
            if e is None:
                log(f"falta {enc} {prot}")
                continue
            assert (e[["left", "right", "label"]].to_numpy() == b[["left", "right", "label"]].to_numpy()).all(), \
                f"pares no alineados {enc} {prot}"
            ce = ((e["logit"] > 0).astype(int) == e["label"]).to_numpy(float)
            d, lo, hi = pigeonhole_ci(ce, cb, li, ri, B=1000)
            plo, phi = pair_boot_ci(ce - cb)
            m = json.loads((RUNS / f"{enc}__{prot}__linear__seed42" / "metrics.json").read_text())
            rows.append({"protocol": prot, "test_key": key, "encoder": enc, "n": len(e), "acc": ce.mean(),
                         **dict(zip(["acc_img_lo", "acc_img_hi"], pigeonhole_ci(ce, None, li, ri, B=1000)[1:])),
                         "delta_vs_dinov2": d, "delta_img_lo": lo, "delta_img_hi": hi,
                         "delta_pair_lo": plo, "delta_pair_hi": phi, "mcnemar_p": mcnemar_p(ce, cb),
                         "log_loss": m["test"][key]["log_loss"], "embed_dim": None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paired_vs_dinov2.csv", index=False)
    with open(OUT / "paired_vs_dinov2.md", "w", encoding="utf-8") as f:
        f.write("| Protocolo | Encoder | n | Acc [IC img] | Δ vs DINOv2 (pts) [IC img] | McNemar p |\n|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            dd = "—" if pd.isna(getattr(r, "delta_vs_dinov2", np.nan)) else \
                f"{100 * r.delta_vs_dinov2:+.2f} [{100 * r.delta_img_lo:+.2f}, {100 * r.delta_img_hi:+.2f}]"
            mp = "—" if pd.isna(getattr(r, "mcnemar_p", np.nan)) else f"{r.mcnemar_p:.1e}"
            f.write(f"| {r.protocol} | {r.encoder} | {r.n:,} | {r.acc:.4f} [{r.acc_img_lo:.4f}, {r.acc_img_hi:.4f}] "
                    f"| {dd} | {mp} |\n")
    log(df.to_string())


if __name__ == "__main__":
    main()
