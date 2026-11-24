"""
Comparacion PAREADA entre encoders sobre los MISMOS pares de test (seed 42, corrida FULL).
Para cada protocolo/split: diferencia de pairwise accuracy con IC95% bootstrap sobre pares
(2000 remuestreos, pareado) y test de McNemar exacto. Nota: el bootstrap sobre pares ignora que
las imagenes se repiten entre pares (dependencia) -> IC algo optimistas; se declara.
Salida: outputs/encoder_paired_comparison.{csv,md}
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from utils import OUTPUTS
R = OUTPUTS / "runs"
ENC = {"ours_simclr_vit_b16": "OURS", "imagenet_vit_b16": "ImageNet", "dinov2_vit_b14": "DINOv2"}
CASES = [("paper_protocol", "test"), ("image_disjoint", "test"), ("city_holdout", "test_strict"), ("city_holdout", "test_touch")]
PAIRS = [("dinov2_vit_b14", "imagenet_vit_b16"), ("dinov2_vit_b14", "ours_simclr_vit_b16"), ("imagenet_vit_b16", "ours_simclr_vit_b16")]
rng = np.random.default_rng(42); rows = []
for proto, split in CASES:
    P = {}
    for e in ENC:
        f = R / f"{e}__{proto}__linear__seed42" / "predictions.csv"
        if f.exists():
            d = pd.read_csv(f); d = d[d.split == split]
            P[e] = d.assign(ok=(d.pred == d.label)).set_index(["left", "right"])["ok"]
    for a, b in PAIRS:
        if a not in P or b not in P: continue
        idx = P[a].index.intersection(P[b].index); x, y = P[a][idx].to_numpy(bool), P[b][idx].to_numpy(bool)
        d = x.astype(float) - y.astype(float); n = len(d)
        boots = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
        n01, n10 = int(((~x) & y).sum()), int((x & (~y)).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        rows.append(dict(protocol=proto, split=split, A=ENC[a], B=ENC[b], n_pairs=n, acc_A=round(x.mean(), 4), acc_B=round(y.mean(), 4),
                         delta=round(d.mean(), 4), ci95_low=round(np.percentile(boots, 2.5), 4), ci95_high=round(np.percentile(boots, 97.5), 4),
                         mcnemar_p=float(f"{p:.3g}")))
df = pd.DataFrame(rows); df.to_csv(OUTPUTS / "encoder_paired_comparison.csv", index=False)
L = ["# Comparacion pareada entre encoders (FULL, seed 42, mismos pares de test)", "",
     "| Protocolo | Split | A | B | n pares | acc A | acc B | Δ (A−B) | IC95% | McNemar p |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    L.append(f"| {r.protocol} | {r.split} | {r.A} | {r.B} | {r.n_pairs:,} | {r.acc_A:.4f} | {r.acc_B:.4f} | {r.delta:+.4f} | [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | {r.mcnemar_p:.3g} |")
L += ["", "IC por bootstrap sobre pares (ignora dependencia por imagen repetida): interpretar como optimistas."]
(OUTPUTS / "encoder_paired_comparison.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
