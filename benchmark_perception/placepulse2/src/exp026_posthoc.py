"""
EXP-026 — diagnósticos POST-HOC (no estaban en el diseño registrado; no cambian ningún criterio).

1. Fiabilidad del residuo por imagen: qué parte de su varianza es ruido binomial del voto
   (con ~6 votos por imagen). Si es casi todo ruido, la autocorrelación del residuo (métrica
   principal de la Parte 2) está muy atenuada y su nulo no es evidencia de ausencia.
2. Solapamiento entre vecindarios: ¿las vecinas por red (NET / NETB) son las MISMAS imágenes que
   las euclídeas con igual número (EUCM)? Si coinciden casi siempre, ninguna métrica puede
   distinguir las dos definiciones con la densidad de Place Pulse.

Uso:  python exp026_posthoc.py      Salidas: outputs/exp026/posthoc_*.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exp026_additive_context import load_images
from exp026_network_residuals import OUT, PILOT, pilot_city
from utils import log, set_seed

SEED = 42


def reliability() -> pd.DataFrame:
    g = pd.read_csv(OUT / "ego0_train_residuals.csv")
    noise = float((g["exp"] * (1 - g["exp"]) / g["n"]).mean())
    total = float(g["resid"].var())
    raw_total = float(g["obs"].var())
    raw_noise = float((g["obs"] * (1 - g["obs"]) / (g["n"] - 1)).mean())
    row = dict(n_images=len(g), median_votes=float(g["n"].median()),
               resid_var=total, resid_binomial_noise_var=noise,
               resid_reliability=1 - noise / total,
               rawscore_var=raw_total, rawscore_binomial_noise_var=raw_noise,
               rawscore_reliability=1 - raw_noise / raw_total)
    log(f"fiabilidad: {row}")
    return pd.DataFrame([row])


def overlap() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    man, _ = load_images("dinov2_vit_b14")
    rows = []
    for city in PILOT:
        g = man[man["city"] == city].reset_index(drop=True)
        V, _, _ = pilot_city(city, g, rng)
        for a, b in (("NET250", "EUCM_NET250"), ("NET400", "EUCM_NET400"),
                     ("NETB400", "EUCM_NETB400"), ("NETB400", "NET400"), ("NET400", "EUC400")):
            both = [(set(x), set(y)) for x, y in zip(V[a], V[b]) if x or y]
            jac = np.array([len(x & y) / len(x | y) for x, y in both])
            rows.append(dict(city=city, a=a, b=b, n_images_with_neighbors=len(both),
                             mean_jaccard=float(jac.mean()),
                             pct_identical_sets=100 * float((jac == 1).mean())))
            log(f"  {city:<9} {a:>7} vs {b:<12} Jaccard medio {jac.mean():.3f}, "
                f"idénticos {100 * (jac == 1).mean():.1f}%")
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(SEED)
    reliability().to_csv(OUT / "posthoc_reliability.csv", index=False, float_format="%.4f")
    overlap().to_csv(OUT / "posthoc_neighbor_overlap.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
