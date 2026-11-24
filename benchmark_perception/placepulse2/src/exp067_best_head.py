"""
EXP-067 (control post-hoc declarado, 13/11 noche): ¿el déficit de los encoders JEPA es de la REPRESENTACIÓN o de que la
sonda lineal-BT con hiperparámetros fijos rinde menos sobre embeddings anisotrópicos?

Motivo (visto al correr el factorial de V-JEPA 2): Ridge → λ, que también es LINEAL, supera a la lineal-BT en +1.05 pts
con V-JEPA 2 (coseno medio entre imágenes 0.87) y en +0.75 con I-JEPA, pero solo +0.16 con DINOv2. Comparar todos los
encoders con la misma sonda fija puede penalizar a los anisotrópicos.

Regla (fijada antes de mirar test): para cada encoder se elige, entre {lineal-BT, MLP-BT, Ridge → λ}, la cabeza con
MAYOR exactitud en VALIDACIÓN; se compara en test contra (a) DINOv2 lineal-BT (la tesis) y (b) DINOv2 con su propia
cabeza elegida igual. IC 95 % bootstrap de imágenes (pigeonhole), mismos 34,607 pares de `image_disjoint`.

Lee outputs/exp063/image_disjoint/{results.jsonl, scores/*__correct.npy, test_pairs_*.npz}.
Uso:  python src/exp067_best_head.py      Salida: outputs/exp067/best_head.{csv,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import pigeonhole_ci  # noqa: E402
from utils import OUTPUTS, log  # noqa: E402

D = OUTPUTS / "exp063" / "image_disjoint"
OUT = OUTPUTS / "exp067"
HEADS = ["linear_bt", "mlp_bt", "ridge"]
REF = "dinov2_vit_b14"


def main() -> None:
    rows = [json.loads(l) for l in (D / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    R = pd.DataFrame(rows)
    R = R[(R["rep"] == "z") & (R["seed"] == 42) & (R["head"].isin(HEADS))]
    R = R.drop_duplicates(["encoder", "head"], keep="last")
    tp = np.load(D / f"test_pairs_{REF}_z.npz")
    li, ri = tp["li"], tp["ri"]

    def correct(enc: str, head: str) -> np.ndarray:
        e = np.load(D / f"test_pairs_{enc}_z.npz")
        assert (e["li"] == li).all() and (e["ri"] == ri).all() and (e["y"] == tp["y"]).all(), enc
        return np.load(D / "scores" / f"{enc}__z__{head}__s42__correct.npy").astype(float)

    best = {}
    for enc, g in R.groupby("encoder"):
        if set(HEADS) - set(g["head"]):
            log(f"{enc}: faltan cabezas {set(HEADS) - set(g['head'])}; se omite")
            continue
        best[enc] = g.sort_values("val_acc", ascending=False).iloc[0]["head"]
    c_ref_lin, c_ref_best = correct(REF, "linear_bt"), correct(REF, best[REF])
    out = []
    for enc, head in best.items():
        c = correct(enc, head)
        d1, lo1, hi1 = pigeonhole_ci(c, c_ref_lin, li, ri, B=1000)
        d2, lo2, hi2 = pigeonhole_ci(c, c_ref_best, li, ri, B=1000)
        lin = correct(enc, "linear_bt").mean()
        out.append({"encoder": enc, "best_head_by_val": head, "acc_linear_bt": lin, "acc_best": c.mean(),
                    "gain_best_vs_own_linear": c.mean() - lin,
                    "delta_vs_dinov2_linear": d1, "lo1": lo1, "hi1": hi1,
                    "delta_vs_dinov2_best": d2, "lo2": lo2, "hi2": hi2})
    df = pd.DataFrame(out).sort_values("acc_best", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "best_head.csv", index=False)
    with open(OUT / "best_head.md", "w", encoding="utf-8") as f:
        f.write(f"Referencia DINOv2: lineal-BT {c_ref_lin.mean():.4f}; mejor cabeza por val = {best[REF]} "
                f"{c_ref_best.mean():.4f}\n\n| Encoder | Cabeza (val) | Lineal-BT | Mejor | Δ vs DINOv2 lineal [IC] | "
                "Δ vs DINOv2 mejor [IC] |\n|---|---|---|---|---|---|\n")
        for r in df.itertuples():
            f.write(f"| {r.encoder} | {r.best_head_by_val} | {r.acc_linear_bt:.4f} | {r.acc_best:.4f} | "
                    f"{100 * r.delta_vs_dinov2_linear:+.2f} [{100 * r.lo1:+.2f}, {100 * r.hi1:+.2f}] | "
                    f"{100 * r.delta_vs_dinov2_best:+.2f} [{100 * r.lo2:+.2f}, {100 * r.hi2:+.2f}] |\n")
    log((OUT / "best_head.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
