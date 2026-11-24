"""
EXP-049a (S7) — Contraste calle–barrio con KL de palabras visuales, sin etiquetas (P-LIMA, 4 distritos).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

Por segmento con spots: histograma BoVW (K = 100, vistas SimCLR, vocabulario del distrito par como en EXP-041) de la
calle frente al de su barrio (spots de los segmentos a 1–2 saltos del grafo dual). KL(calle ‖ barrio) con Dirichlet
α = 0.5, estandarizada contra un nulo de 200 sorteos: mismos nº de spots tomados del barrio, comparados con el resto del
barrio → z. Luego: (i) % de calles con z > 1.96; (ii) ρ(z, |percentil predicho calle − barrio|) con el probe DINOv2 de
EXP-051; (iii) en B/LV/SI, ρ(z, contraste de crimen 2016 por 100 m), con signo y en valor absoluto.

Uso:  python exp049a_kl_contrast.py        Salidas: outputs/exp049a/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest, spearmanr

from exp041_lima_segment_graph import DISTRICTS, K_BOVW, VOCAB_PAIR, lima_spots, load_graph
from exp051_perception_crime_gap import CRIME_DISTRICT, snap_crimes
from lima_views import CRIMES
from segment_graph import dual_adjacency, segments_table
from utils import OUTPUTS, environment_info, log, set_seed
from visual_repr import BoVW, histograms

SEED = 42
OUT = OUTPUTS / "exp049a"
ALPHA = 0.5
N_NULL = 200
MIN_BARRIO = 5
N_BOOT = 2000


def kl(p_counts: np.ndarray, q_counts: np.ndarray) -> np.ndarray:
    """KL(p ‖ q) por fila con suavizado de Dirichlet α (acepta matrices de sorteos)."""
    p = p_counts + ALPHA
    q = q_counts + ALPHA
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    return (p * np.log(p / q)).sum(-1)


def boot_rho(x, y, rng):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(r), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), int(n)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    S, V, E, _ = lima_spots(plima=True)
    E = np.asarray(E)
    snap = pd.read_csv(OUTPUTS / "exp041_plima" / "spot_to_segment.csv", dtype={"spot_id": str})
    S = S.merge(snap[["district", "spot_id", "seg_idx"]], on=["district", "spot_id"], how="left")
    P = pd.read_csv(OUTPUTS / "exp051_dinov2" / "spot_predicted_perception.csv", dtype={"spot_id": str})
    S = S.merge(P[["district", "spot_id", "pct_pp"]], on=["district", "spot_id"], how="left")
    assert S["pct_pp"].notna().all() and S["seg_idx"].notna().all()
    C = pd.read_csv(CRIMES, sep=";")
    seg_rows, summ = [], []
    for d in DISTRICTS:
        Gp, _ = load_graph(d)
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        A1 = ((A + A.T) > 0).astype(np.int32).tocsr()
        R2 = ((A1 + A1 @ A1) > 0).tocsr()
        m = np.flatnonzero((S["district"] == d).to_numpy())
        Vo = V[V["district"] == VOCAB_PAIR[d]]
        bv = BoVW(K_BOVW, seed=0).fit(E[Vo["row"].to_numpy()])
        Vd = V[V["district"] == d]
        local = {s: i for i, s in enumerate(m)}
        H = histograms(bv.assign(E[Vd["row"].to_numpy()]), Vd["sidx"].map(local).to_numpy(), K_BOVW, len(m),
                       norm="none")
        seg_of = S.loc[m, "seg_idx"].to_numpy().astype(int)
        pct = S.loc[m, "pct_pp"].to_numpy()
        crime_dens = None
        if d in CRIME_DISTRICT:
            Cy = C[(C["DISTRITO"] == CRIME_DISTRICT[d]) & (C["YEAR"] == 2016)]
            cnt, _ = snap_crimes(seg, Gp, Cy[["longitude", "latitude"]].to_numpy())
            crime_dens = np.log1p(cnt / (seg["length_m"].to_numpy() / 100.0))
        by_seg = pd.Series(np.arange(len(m))).groupby(seg_of).apply(list).to_dict()
        by_seg.pop(-1, None)
        for s, own in by_seg.items():
            nb = [t for t in R2.indices[R2.indptr[s]:R2.indptr[s + 1]] if t != s]
            bar = [j for t in nb for j in by_seg.get(t, [])]
            if len(bar) < max(MIN_BARRIO, len(own) + 1):        # el nulo necesita más spots que la calle
                continue
            h, hb = H[own].sum(0), H[bar].sum(0)
            k = len(own)
            draws = np.array([rng.choice(len(bar), size=k, replace=False) for _ in range(N_NULL)])
            hd = H[np.asarray(bar)[draws]].sum(1)                       # (N_NULL, K)
            null = kl(hd, hb[None, :] - hd)
            obs = float(kl(h, hb))
            row = dict(district=d, seg_idx=int(s), n_spots=k, n_barrio_spots=len(bar), n_barrio_segments=len(nb),
                       kl=obs, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                       z=float((obs - null.mean()) / max(null.std(ddof=1), 1e-9)),
                       pct_street=float(pct[own].mean()), pct_barrio=float(pct[bar].mean()))
            if crime_dens is not None:
                row["crime_street"] = float(crime_dens[s])
                row["crime_barrio"] = float(crime_dens[nb].mean())
            seg_rows.append(row)
        T = pd.DataFrame([r for r in seg_rows if r["district"] == d])
        T["pct_contrast_abs"] = (T["pct_street"] - T["pct_barrio"]).abs()
        n_dist = int((T["z"] > 1.96).sum())
        bt = binomtest(n_dist, len(T), 0.025)
        ci = bt.proportion_ci(0.95)
        res = dict(district=d, sampling="street_sampling" if d == "lince" else "crime_points", n_segments=len(T),
                   pct_distinct=100 * n_dist / len(T), pct_distinct_ci_low=100 * ci.low,
                   pct_distinct_ci_high=100 * ci.high, z_median=float(T["z"].median()))
        r = boot_rho(T["z"].to_numpy(), T["pct_contrast_abs"].to_numpy(), rng)
        res.update(rho_pct_abs=r[0], rho_pct_abs_ci_low=r[1], rho_pct_abs_ci_high=r[2])
        if crime_dens is not None:
            cc = (T["crime_street"] - T["crime_barrio"]).to_numpy()
            for lab, y in (("crime_signed", cc), ("crime_abs", np.abs(cc))):
                r = boot_rho(T["z"].to_numpy(), y, rng)
                res.update({f"rho_{lab}": r[0], f"rho_{lab}_ci_low": r[1], f"rho_{lab}_ci_high": r[2]})
            # ¿Las calles visualmente distintivas tienen más crimen que su barrio? (media del contraste, z > 1.96 vs resto)
            dist = T["z"].to_numpy() > 1.96
            res.update(crime_contrast_distinct=float(np.nanmean(cc[dist])) if dist.any() else np.nan,
                       crime_contrast_rest=float(np.nanmean(cc[~dist])))
        summ.append(res)
        log(f"{d}: {res}")
    pd.DataFrame(seg_rows).to_csv(OUT / "segment_contrast.csv", index=False, float_format="%.5f")
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    (OUT / "run_info.json").write_text(json.dumps({"alpha": ALPHA, "n_null": N_NULL, "min_barrio": MIN_BARRIO,
                                                   "k_bovw": K_BOVW, "barrio": "1-2 saltos del grafo dual",
                                                   "minutes": round((time.perf_counter() - t0) / 60, 1),
                                                   "env": environment_info()}, indent=2, default=str),
                                       encoding="utf-8")
    log(f"listo en {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
