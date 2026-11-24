"""
EXP-055 — ¿Cómo vamos frente al estado del arte? Contexto del 73.5 % de Dubey et al. (2016) con la MISMA métrica.

Asesoría 05/11: "frente a lo que ya existe, cómo vamos". La referencia directa es Dubey et al. 2016 (RSS-CNN VGG,
73.5 % de exactitud por pares en "safe"), con partición ALEATORIA de comparaciones 65/5/30 (= nuestro
paper_protocol, 286,492 / 22,037 / 132,229 pares). En esa partición el 95.7 % de las imágenes de test aparece en
train (EXP-004), así que un modelo afinado de punta a punta puede MEMORIZAR el puntaje de cada imagen.

Mide, sin GPU y sin contenido visual:
  (1) MEM: Bradley–Terry ajustado SOLO con los pares de train (identidad de la imagen, cero píxeles) → test.
  (2) DINOv2 + MEM: logit del probe congelado (run de RQ1) + β·(λ_izq − λ_der), β = 1 FIJADO a priori (ambos son
      log-odds: producto de expertos). Los runs no guardaron logits de val; el barrido de β es solo sensibilidad.
  (3) Techos: (a) acuerdo entre votantes en pares repetidos; (b) oráculo BT ajustado con TODOS los votos.
  (4) Lo mismo en image_disjoint y city_holdout (MEM debería dar 0.5: imágenes de test no vistas).
IC 95 %: bootstrap de pares (1,000). Salida: outputs/exp055/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp038_bt_target import bt_global
from utils import MANIFESTS, OUTPUTS, SPLITS, log

OUT = OUTPUTS / "exp055"
RUNS = OUTPUTS / "runs"
N_BOOT = 1000
RNG = np.random.default_rng(42)


def acc(logit: np.ndarray, y: np.ndarray) -> float:
    """Exactitud por pares; logit exactamente 0 (imagen sin información) cuenta como media moneda."""
    return float(np.mean(np.where(logit == 0, 0.5, (logit > 0) == y)))


def ci(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(y)
    b = [acc(logit[i], y[i]) for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def probe_logits(protocol: str, P: pd.DataFrame, idx: list[int], split: str) -> np.ndarray:
    pr = pd.read_csv(RUNS / f"dinov2_vit_b14__{protocol}__linear__seed42" / "predictions.csv", dtype={"left": str, "right": str})
    pr = pr[pr["split"] == split].reset_index(drop=True)
    sub = P.iloc[idx].reset_index(drop=True)
    assert len(pr) == len(sub) and (pr["left"].values == sub["left"].values).all(), "orden de predicciones ≠ split"
    return pr["logit"].to_numpy()


def voter_ceiling(P: pd.DataFrame) -> dict:
    """Pares no ordenados votados ≥ 2 veces: prob. de que un voto coincida con la mayoría de los DEMÁS (LOO)."""
    a = np.where(P["left"] < P["right"], P["left"], P["right"])
    b = np.where(P["left"] < P["right"], P["right"], P["left"])
    win_a = np.where(P["left"] < P["right"], P["label"], 1 - P["label"])
    g = pd.DataFrame({"a": a, "b": b, "w": win_a}).groupby(["a", "b"])["w"].agg(["sum", "count"])
    g = g[g["count"] >= 2]
    hits, tot = 0.0, 0
    for s, n in zip(g["sum"].to_numpy(), g["count"].to_numpy()):
        for v, k in ((1, s), (0, n - s)):          # k votos con valor v
            others_for_1 = s - (v == 1)
            m = n - 1
            p = 1.0 if 2 * others_for_1 > m else 0.0 if 2 * others_for_1 < m else 0.5
            hits += k * (p if v == 1 else 1 - p)
            tot += k
    return {"n_pairs_repeated": int(len(g)), "n_votes_in_repeated": int(tot),
            "share_votes_in_repeated": float(tot / len(P)), "loo_majority_agreement": float(hits / tot)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(MANIFESTS / "pairs_safe.csv", usecols=["left", "right", "label"], dtype={"left": str, "right": str})
    y_all = P["label"].to_numpy()
    rows = []
    oracle = bt_global(P)                                       # techo optimista: conoce los votos de test
    for protocol, test_key in (("paper_protocol", "test"), ("image_disjoint", "test"), ("city_holdout", "test_strict")):
        S = json.loads((SPLITS / f"{protocol}.json").read_text())
        tr, va, te = S["train"], S["val"], S[test_key]
        lam = bt_global(P.iloc[tr])
        def mem(idx):
            sub = P.iloc[idx]
            return (lam.reindex(sub["left"]).fillna(0).to_numpy() - lam.reindex(sub["right"]).fillna(0).to_numpy())
        m_te = mem(te)
        y_te = y_all[te]
        seen = (P.iloc[te]["left"].isin(lam.index) & P.iloc[te]["right"].isin(lam.index)).to_numpy()
        z_te = probe_logits(protocol, P, te, test_key)
        sens = {f"{b:.2f}": acc(z_te + b * m_te, y_te) for b in (0.25, 0.5, 1.0, 2.0, 4.0)}
        log(f"{protocol} sensibilidad β (post-hoc, solo lectura): {sens}")
        sub = P.iloc[te]
        o_te = oracle.reindex(sub["left"]).to_numpy() - oracle.reindex(sub["right"]).to_numpy()
        for name, lg in (("MEM: BT de train, sin píxeles", m_te),
                         ("DINOv2 congelado + probe lineal (RQ1)", z_te),
                         ("DINOv2 + MEM (β = 1 fijo)", z_te + m_te),
                         ("Oráculo BT con todos los votos (techo optimista)", o_te)):
            lo, hi = ci(lg, y_te)
            rows.append({"protocol": protocol, "model": name, "n_test_pairs": len(te),
                         "share_test_pairs_both_seen": float(seen.mean()),
                         "pairwise_accuracy": acc(lg, y_te), "ci95_low": lo, "ci95_high": hi})
            log(f"{protocol:15s} {name:50s} {rows[-1]['pairwise_accuracy']:.4f} [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(OUT / "sota_protocol.csv", index=False)
    vc = voter_ceiling(P)
    (OUT / "voter_ceiling.json").write_text(json.dumps(vc, indent=2), encoding="utf-8")
    log(f"acuerdo entre votantes (pares repetidos): {vc}")


if __name__ == "__main__":
    main()
