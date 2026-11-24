"""
EXP-050 — Análisis de la encuesta pareada de Lima (pre-registrado en docs/EXPERIMENT_REGISTRY.md).

  analyze   lee outputs/survey/responses/{votes,sessions}.csv (descarga CSV de la hoja de Google), deduplica por
            (sid, n), aplica los filtros de calidad y calcula: (1) Bradley–Terry por estímulo con IC bootstrap por
            participante; (2) split-half por participantes (Spearman–Brown) y consistencia en los pares de control;
            (3) transferencia PP → Lima (ρ con el probe SimCLR y DINOv2); (6) diferencias entre distritos.
  simulate  potencia: con puntuaciones latentes conocidas y el mismo esquema de selección activa, ¿qué fiabilidad
            (split-half y ρ con la verdad) se alcanza con 1,500 / 3,000 / 5,000 / 10,000 votos?

Empates: medio triunfo para cada lado en la verosimilitud BT (declarado). Salidas: outputs/survey/analysis/.
Uso:  python analyze_survey.py --stage analyze|simulate [--responses carpeta]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from utils import OUTPUTS, log

SEED = 42
SURVEY = OUTPUTS / "survey"
OUT = SURVEY / "analysis"
N_BOOT = 1000
MIN_MEDIAN_RT_MS = 700
L2 = 1e-2                       # prior gaussiano débil: identifica la escala con pocos votos


def fit_bt(a: np.ndarray, b: np.ndarray, s: np.ndarray, n_items: int) -> np.ndarray:
    """θ por máxima verosimilitud penalizada; s = 1 gana a, 0 gana b, 0.5 empate. Media 0."""
    def f(th):
        d = th[a] - th[b]
        p = 1.0 / (1.0 + np.exp(-d))
        ll = s * np.log(p + 1e-12) + (1 - s) * np.log(1 - p + 1e-12)
        g = np.bincount(a, s - p, n_items) - np.bincount(b, s - p, n_items)
        return -ll.sum() + L2 * th @ th, -g + 2 * L2 * th
    th = minimize(f, np.zeros(n_items), jac=True, method="L-BFGS-B").x
    return th - th.mean()


def split_half(V: pd.DataFrame, n_items: int, rng, reps: int = 20) -> float:
    """Spearman–Brown de la correlación entre los BT de dos mitades de participantes (media de `reps` particiones)."""
    sids = V["sid"].unique()
    rs = []
    for _ in range(reps):
        h = set(rng.permutation(sids)[: len(sids) // 2])
        m = V["sid"].isin(h).to_numpy()
        t1 = fit_bt(V["ia"].to_numpy()[m], V["ib"].to_numpy()[m], V["s"].to_numpy()[m], n_items)
        t2 = fit_bt(V["ia"].to_numpy()[~m], V["ib"].to_numpy()[~m], V["s"].to_numpy()[~m], n_items)
        r = spearmanr(t1, t2)[0]
        rs.append(2 * r / (1 + r))
    return float(np.mean(rs))


def stage_analyze(responses) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    man = pd.read_csv(SURVEY / "stimuli_manifest.csv")
    idx = {s: i for i, s in enumerate(man["stim_id"])}
    V = pd.read_csv(responses / "votes.csv")
    n_raw = len(V)
    V = V.drop_duplicates(["sid", "n"])
    rt_med = V.groupby("sid")["rt"].median()
    keep = rt_med.index[rt_med >= MIN_MEDIAN_RT_MS]
    ctrl = V[V["rep"].astype(str).str.lower() == "true"]
    # Consistencia: el par de control invertido (a,b) = (b',a') de un voto anterior de la misma sesión.
    cons = []
    for r in ctrl.itertuples():
        prev = V[(V["sid"] == r.sid) & (V["a"] == r.b) & (V["b"] == r.a) & (V["n"] < r.n)]
        if len(prev) and r.w != "tie" and prev.iloc[0]["w"] != "tie":
            cons.append(r.w != prev.iloc[0]["w"])          # lados invertidos: mismo lugar ⇔ letra distinta
    V = V[V["sid"].isin(keep) & (V["rep"].astype(str).str.lower() != "true")].copy()
    V["ia"], V["ib"] = V["a"].map(idx), V["b"].map(idx)
    V["s"] = V["w"].map({"a": 1.0, "b": 0.0, "tie": 0.5})
    n_items = len(man)
    th = fit_bt(V["ia"].to_numpy(), V["ib"].to_numpy(), V["s"].to_numpy(), n_items)
    sids = V["sid"].unique()
    groups = {s: g for s, g in V.groupby("sid")}
    B = np.empty((N_BOOT, n_items))
    for k in range(N_BOOT):
        W = pd.concat([groups[s] for s in rng.choice(sids, len(sids))])
        B[k] = fit_bt(W["ia"].to_numpy(), W["ib"].to_numpy(), W["s"].to_numpy(), n_items)
    man["bt"], man["bt_lo"], man["bt_hi"] = th, np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
    man["n_votes"] = np.bincount(V["ia"], minlength=n_items) + np.bincount(V["ib"], minlength=n_items)
    res = {"votes_raw": n_raw, "votes_used": len(V), "sessions": int(len(sids)),
           "sessions_excluded_rt": int(len(rt_med) - len(keep)),
           "control_consistency": float(np.mean(cons)) if cons else None, "control_pairs": len(cons),
           "split_half_sb": split_half(V, n_items, rng)}
    # (3) Transferencia PP → Lima
    for enc, d in [("simclr", "exp051"), ("dinov2", "exp051_dinov2")]:
        P = pd.read_csv(OUTPUTS / d / "spot_predicted_perception.csv", dtype={"spot_id": str})
        m = man.merge(P[["district", "spot_id", "pct_pp"]], on=["district", "spot_id"], how="left",
                      suffixes=("", f"_{enc}"))
        x = m[f"pct_pp_{enc}"] if f"pct_pp_{enc}" in m else m["pct_pp"]
        res[f"rho_pp_{enc}_all"] = float(spearmanr(x, m["bt"], nan_policy="omit")[0])
        res[f"rho_pp_{enc}_within"] = {k: float(spearmanr(x[g.index], g["bt"], nan_policy="omit")[0])
                                       for k, g in m.groupby("district")}
    # (6) Diferencias entre distritos (media BT, IC bootstrap por participante)
    res["district_bt"] = {k: [float(th[g.index].mean()), float(np.percentile(B[:, g.index].mean(1), 2.5)),
                              float(np.percentile(B[:, g.index].mean(1), 97.5))] for k, g in man.groupby("district")}
    man.to_csv(OUT / "stimulus_bt.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(res, indent=2, ensure_ascii=False))


def simulate_votes(theta, n_votes, rng, per_session=25, tie=0.10):
    """Mismo esquema que la página: peso 1/(1+apariciones)^2, sin pares repetidos dentro de una sesión."""
    n = len(theta)
    cnt = np.zeros(n)
    rows = []
    sid = 0
    while len(rows) < n_votes:
        sid += 1
        seen = set()
        for _ in range(per_session):
            w = 1 / (1 + cnt) ** 2
            for _t in range(50):
                a = rng.choice(n, p=w / w.sum())
                w2 = w.copy(); w2[a] = 0
                b = rng.choice(n, p=w2 / w2.sum())
                if (min(a, b), max(a, b)) not in seen:
                    break
            seen.add((min(a, b), max(a, b)))
            cnt[a] += 1; cnt[b] += 1
            u = rng.random()
            p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
            s = 0.5 if u < tie else float(rng.random() < p)
            rows.append((f"p{sid}", a, b, s))
    return pd.DataFrame(rows[:n_votes], columns=["sid", "ia", "ib", "s"])


def stage_simulate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    # dispersión de la percepción latente entre lugares. 0.5 y 1.0 eran supuestos; 0.73 es la dispersión DENTRO de una
    # ciudad estimada en Place Pulse por EXP-066 (momentos simulados), el caso de Lima (una sola ciudad)
    for sd in (0.5, 0.73, 1.0):
        for nv in (1500, 3000, 5000, 10000):
            for rep in range(3):
                theta = rng.normal(0, sd, 400)
                V = simulate_votes(theta, nv, rng)
                th = fit_bt(V["ia"].to_numpy(), V["ib"].to_numpy(), V["s"].to_numpy(), 400)
                rows.append({"sd": sd, "votes": nv, "rep": rep, "participants": V["sid"].nunique(),
                             "rho_truth": spearmanr(th, theta)[0], "split_half_sb": split_half(V, 400, rng, reps=5)})
                log(rows[-1])
    R = pd.DataFrame(rows).groupby(["sd", "votes"]).mean(numeric_only=True).drop(columns="rep").reset_index()
    R.to_csv(OUT / "power_simulation.csv", index=False, float_format="%.3f")
    log("\n" + R.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["analyze", "simulate"], required=True)
    ap.add_argument("--responses", default=str(SURVEY / "responses"), help="carpeta con votes.csv")
    a = ap.parse_args()
    stage_analyze(Path(a.responses)) if a.stage == "analyze" else stage_simulate()
