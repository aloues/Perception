"""
EXP-062 — Estimación conjunta de Δ C′ − A′ (percepción por barrio) en Santiago, Atlanta y Berlín (probe LOCO).
Meta-análisis de efecto fijo, de efectos aleatorios (DerSimonian–Laird) y DL con corrección Hartung–Knapp–Sidik–Jonkman
(t con k − 1 gl; con 3 ciudades el IC de DL es optimista). El EE de cada ciudad sale de su IC bootstrap.
Entrada: outputs/exp059/cross_photo_{ciudad}_loco.csv. Salida: outputs/exp062_pooled.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, t

from utils import OUTPUTS, log

CITIES = ("santiago", "atlanta", "berlin")


def pool(y: np.ndarray, se: np.ndarray) -> dict:
    w = 1 / se**2
    fe = (w * y).sum() / w.sum()
    q = (w * (y - fe) ** 2).sum()
    k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    wr = 1 / (se**2 + tau2)
    re = (wr * y).sum() / wr.sum()
    hk_se = ((wr * (y - re) ** 2).sum() / ((k - 1) * wr.sum())) ** 0.5
    return {"fe": fe, "fe_se": w.sum() ** -0.5, "re": re, "re_se": wr.sum() ** -0.5, "hk_se": hk_se, "hk_t": t.ppf(0.975, k - 1), "Q": q,
            "Q_p": chi2.sf(q, k - 1), "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, "tau": tau2**0.5}


def main() -> None:
    rows = []
    for side in (1000, 500):
        d = pd.concat([pd.read_csv(OUTPUTS / "exp059" / f"cross_photo_{c}_loco.csv").assign(city=c) for c in CITIES])
        d = d[(d["side_m"] == side) & (d["quantity"] == "delta_C-A")].set_index("city").loc[list(CITIES)]
        se = ((d["ci_high"] - d["ci_low"]) / (2 * 1.96)).to_numpy()
        for subset, idx in (("3_ciudades", slice(None)), ("solo_replicas", slice(1, None))):
            r = pool(d["value"].to_numpy()[idx], se[idx])
            for kind, val, half in (("fe", r["fe"], 1.96 * r["fe_se"]), ("re", r["re"], 1.96 * r["re_se"]),
                                    ("re_hksj", r["re"], r["hk_t"] * r["hk_se"])):
                rows.append({"side_m": side, "subset": subset, "model": kind, "value": val, "ci_low": val - half,
                             "ci_high": val + half, "Q": r["Q"], "Q_p": r["Q_p"], "I2": r["I2"], "tau": r["tau"]})
            log(f"{side} m {subset}: FE {r['fe']:+.3f} · RE {r['re']:+.3f} ± {1.96 * r['re_se']:.3f} (DL) ± {r['hk_t'] * r['hk_se']:.3f} (HKSJ) · I²={r['I2']:.2f}")
    pd.DataFrame(rows).round(4).to_csv(OUTPUTS / "exp062_pooled.csv", index=False)


if __name__ == "__main__":
    main()
