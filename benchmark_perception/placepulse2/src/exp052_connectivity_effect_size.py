"""
EXP-052 — Tamaño de efecto interpretable de la conectividad (AUC estratificado, d de Cohen) para la similitud
visual (SimCLR, y DINOv2 en SI/Lince) y para la percepción PREDICHA (EXP-051) entre segmentos de calle.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

Uso:  python exp052_connectivity_effect_size.py [--no-corner]   Salidas: outputs/exp052/ (o exp052_nocorner/)
--no-corner: excluye spots a ≤ 15 m de una intersección (sus vistas fotografían la calle transversal), igual que
la variante `no_corner_spots` de EXP-041.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from exp041_lima_segment_graph import DISTRICTS, load_graph, lima_spots
from lima_views import PLIMA_EXPORTS
from segment_graph import dual_adjacency, segment_midpoints, segments_table
from utils import OUTPUTS, log, set_seed
from visual_repr import l2n

SEED = 42
OUT = OUTPUTS / "exp052"
MAX_M, STRATUM_M, N_BOOT = 300.0, 25.0, 2000
DINO = {"barranco": "Inseguros-Barranco-GGZ-2016", "la_victoria": "Inseguros-La_Victoria-GGZ-2016",
        "san_isidro": "Inseguros-San_Isidro-GGZ-2016", "lince": "StreetSample-Lince-s50"}


def dino_spot(folder: str, normalize: bool = True) -> dict:
    z = np.load(PLIMA_EXPORTS / folder / "dinov2.npz")["cls"]
    paths = json.loads((PLIMA_EXPORTS / folder / "dinov2_index.json").read_text(encoding="utf-8"))["paths"]
    spot = pd.Series([p.split("/")[1] for p in paths])
    f = l2n if normalize else (lambda v: v)
    return {s: f(z[idx.to_numpy()].mean(0)) for s, idx in spot.groupby(spot).groups.items()}


def pair_table(seg, A, seg_of: np.ndarray) -> pd.DataFrame:
    F = np.unique(seg_of[seg_of >= 0])
    mid = segment_midpoints(seg)[F]
    i, j = np.array(sorted(cKDTree(mid).query_pairs(MAX_M))).T
    d = np.linalg.norm(mid[i] - mid[j], axis=1)
    conn = np.asarray(A[F[i], F[j]]).ravel() > 0
    return pd.DataFrame({"a": i, "b": j, "dist": d, "stratum": (d // STRATUM_M).astype(int), "connected": conn}), F


def auc_d(P: pd.DataFrame, sim: np.ndarray) -> tuple[float, float]:
    """AUC estratificado (ponderado por nº de pares conectados del estrato) y d de Cohen medio."""
    num = den = dsum = dw = 0.0
    for _, g in P.assign(sim=sim).groupby("stratum"):
        c, n = g.loc[g.connected, "sim"].to_numpy(), g.loc[~g.connected, "sim"].to_numpy()
        if len(c) < 2 or len(n) < 2:
            continue
        r = pd.Series(np.r_[c, n]).rank().to_numpy()[:len(c)]
        auc = (r.sum() - len(c) * (len(c) + 1) / 2) / (len(c) * len(n))
        sd = np.sqrt((c.var(ddof=1) + n.var(ddof=1)) / 2) or np.nan
        num += auc * len(c); den += len(c)
        dsum += (c.mean() - n.mean()) / sd * len(c); dw += len(c)
    return num / den, dsum / dw


def boot(P, sim, nseg, rng):
    est = auc_d(P, sim)
    a, b = P["a"].to_numpy(), P["b"].to_numpy()
    bs = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, nseg, nseg), minlength=nseg)
        rep = np.repeat(np.arange(len(P)), w[a])           # cluster por el segmento a (cada par una vez)
        bs.append(auc_d(P.iloc[rep], sim[rep]))
    bs = np.array(bs)
    return dict(auc=est[0], auc_lo=np.nanpercentile(bs[:, 0], 2.5), auc_hi=np.nanpercentile(bs[:, 0], 97.5),
                d=est[1], d_lo=np.nanpercentile(bs[:, 1], 2.5), d_hi=np.nanpercentile(bs[:, 1], 97.5))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-corner", action="store_true")
    args = ap.parse_args()
    if args.no_corner:
        OUT = OUTPUTS / "exp052_nocorner"
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    S, V, E, Zspot = lima_spots(plima=True)
    snap = pd.read_csv(OUTPUTS / "exp041_plima" / "spot_to_segment.csv", dtype={"spot_id": str})
    perc = pd.read_csv(OUTPUTS / "exp051" / "spot_predicted_perception.csv", dtype={"spot_id": str})
    S = S.reset_index().merge(snap[["district", "spot_id", "seg_idx", "near_intersection"]], on=["district", "spot_id"],
                              how="left")
    if args.no_corner:
        S = S[~S["near_intersection"].fillna(True).astype(bool)]
    S = S.merge(perc[["district", "spot_id", "pct_pp"]], on=["district", "spot_id"], how="left")
    rows = []
    for d in DISTRICTS:
        Gp, _ = load_graph(d)
        seg = segments_table(Gp)
        A = dual_adjacency(seg).tocsr()
        g = S[(S["district"] == d) & (S["seg_idx"] >= 0)]
        P, F = pair_table(seg, A, g["seg_idx"].to_numpy().astype(int))
        pos = {s: k for k, s in enumerate(F)}
        k = g["seg_idx"].astype(int).map(pos).to_numpy()
        feats = {"visual_simclr": Zspot[g["index"].to_numpy()]}
        if d in DINO and (PLIMA_EXPORTS / DINO[d] / "dinov2.npz").exists():
            dz = dino_spot(DINO[d])
            assert all(s in dz for s in g["spot_id"]), f"{d}: spots sin DINOv2"
            feats["visual_dinov2"] = np.stack([dz[s] for s in g["spot_id"]])
        for name, X in feats.items():
            Zs = np.zeros((len(F), X.shape[1]))
            np.add.at(Zs, k, X)
            Zs = l2n(Zs)
            sim = (Zs[P["a"]] * Zs[P["b"]]).sum(1)
            rows.append(dict(district=d, measure=name, n_segments=len(F), n_pairs=len(P),
                             n_connected=int(P.connected.sum()), **boot(P, sim, len(F), rng)))
            log(f"{d} {name}: {rows[-1]}")
        ps = pd.Series(g["pct_pp"].to_numpy()).groupby(k).mean().reindex(range(len(F))).to_numpy()
        sim = -np.abs(ps[P["a"]] - ps[P["b"]])
        rows.append(dict(district=d, measure="predicted_perception", n_segments=len(F), n_pairs=len(P),
                         n_connected=int(P.connected.sum()), **boot(P, sim, len(F), rng)))
        log(f"{d} perception: {rows[-1]}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "effect_sizes.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps({"rows": R.to_dict("records"),
                                                  "seconds": round(time.perf_counter() - t0)}, indent=2), encoding="utf-8")
    log(R[["district", "measure", "auc", "auc_lo", "auc_hi", "d"]].to_string())


if __name__ == "__main__":
    main()
