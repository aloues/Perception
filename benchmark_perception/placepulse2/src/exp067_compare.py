"""
EXP-067 — Contrastes pareados pre-registrados (13/11 noche) con la MISMA cabeza lineal-BT, mismos pares y seed:

    datos   : I-JEPA ViT-H/14 IN-22k  vs  I-JEPA ViT-H/14 IN-1k        (misma arquitectura y método)
    JEPA    : V-JEPA 2 ViT-L          vs  DINOv2 ViT-B/14                (familia JEPA vigente)
    escala  : DINOv2 ViT-L/14 (CLS)   vs  DINOv2 ViT-B/14 (CLS)          (mismo método y datos)
    + cada encoder nuevo vs DINOv2 ViT-B/14 (referencia de la tesis)

Δ con IC 95 % por bootstrap de imágenes (pigeonhole, Owen 2007) y de pares; McNemar.
Uso:  python src/exp067_compare.py        Salida: outputs/exp067/contrasts.{csv,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import mcnemar_p, pair_boot_ci, pigeonhole_ci  # noqa: E402
from exp064_compare import PROTOCOLS, load  # noqa: E402
from utils import OUTPUTS, RUNS, log  # noqa: E402

OUT = OUTPUTS / "exp067"
REF = "dinov2_vit_b14"
CONTRASTS = [  # (nombre, encoder, referencia, hipótesis)
    ("datos: I-JEPA IN-22k − IN-1k", "ijepa_vith14_in22k_meanpatch", "ijepa_vith14_meanpatch", "H-067a"),
    ("JEPA vigente: V-JEPA 2 L − DINOv2 B", "vjepa2_vitl16_meanpatch", REF, "H-067b"),
    ("escala: DINOv2 L − B (CLS)", "dinov2_vitl14_cls", REF, "H-067c"),
    ("DINOv2 L media de parches − DINOv2 B", "dinov2_vitl14_meanpatch", REF, "—"),
    ("I-JEPA IN-22k − DINOv2 B", "ijepa_vith14_in22k_meanpatch", REF, "—"),
]
LABEL = {REF: "DINOv2 ViT-B/14 (CLS)", "ijepa_vith14_meanpatch": "I-JEPA ViT-H/14 IN-1k",
         "ijepa_vith14_in22k_meanpatch": "I-JEPA ViT-H/14 IN-22k", "vjepa2_vitl16_meanpatch": "V-JEPA 2 ViT-L/16",
         "dinov2_vitl14_cls": "DINOv2 ViT-L/14 (CLS)", "dinov2_vitl14_meanpatch": "DINOv2 ViT-L/14 (mean patch)",
         "dinov3_vitb16_cls": "DINOv3 ViT-B/16 (CLS)"}


def correct(d: pd.DataFrame) -> np.ndarray:
    return ((d["logit"] > 0).astype(int) == d["label"]).to_numpy(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, accs = [], []
    for prot, key in PROTOCOLS.items():
        base = load(REF, prot, key)
        ids = pd.Index(pd.unique(np.r_[base["left"], base["right"]]))
        li, ri = ids.get_indexer(base["left"]), ids.get_indexer(base["right"])
        for enc in LABEL:
            e = load(enc, prot, key)
            if e is None:
                continue
            c = correct(e)
            _, lo, hi = pigeonhole_ci(c, None, li, ri, B=1000)
            m = json.loads((RUNS / f"{enc}__{prot}__linear__seed42" / "metrics.json").read_text())["test"][key]
            accs.append({"protocol": prot, "encoder": enc, "label": LABEL[enc], "acc": c.mean(), "acc_lo": lo,
                         "acc_hi": hi, "log_loss": m["log_loss"]})
        for name, enc, ref, hyp in CONTRASTS:
            e, r = load(enc, prot, key), load(ref, prot, key)
            if e is None or r is None:
                log(f"falta {enc} o {ref} en {prot}")
                continue
            assert (e[["left", "right", "label"]].to_numpy() == r[["left", "right", "label"]].to_numpy()).all()
            ce, cr = correct(e), correct(r)
            d, lo, hi = pigeonhole_ci(ce, cr, li, ri, B=1000)
            plo, phi = pair_boot_ci(ce - cr)
            rows.append({"protocol": prot, "contrast": name, "hypothesis": hyp, "encoder": enc, "reference": ref,
                         "n": len(e), "acc_encoder": ce.mean(), "acc_reference": cr.mean(), "delta": d,
                         "delta_img_lo": lo, "delta_img_hi": hi, "delta_pair_lo": plo, "delta_pair_hi": phi,
                         "mcnemar_p": mcnemar_p(ce, cr)})
    df, acc = pd.DataFrame(rows), pd.DataFrame(accs)
    df.to_csv(OUT / "contrasts.csv", index=False)
    acc.to_csv(OUT / "accuracies.csv", index=False)
    with open(OUT / "contrasts.md", "w", encoding="utf-8") as f:
        f.write("| Protocolo | Contraste | H | Acc enc / ref | Δ (pts) [IC img] | McNemar p |\n|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.protocol} | {r.contrast} | {r.hypothesis} | {r.acc_encoder:.4f} / {r.acc_reference:.4f} | "
                    f"{100 * r.delta:+.2f} [{100 * r.delta_img_lo:+.2f}, {100 * r.delta_img_hi:+.2f}] | "
                    f"{r.mcnemar_p:.1e} |\n")
    log(df.to_string())


if __name__ == "__main__":
    main()
