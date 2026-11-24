"""
EXP-038 — Objetivo humano ajustado por la fuerza del rival (Bradley–Terry) para los mapas por barrio.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-09).

La tasa de victoria de una celda mezcla su percepción con la fuerza de los rivales que le tocaron al azar. Aquí la
percepción de la celda se estima como un parámetro de Bradley–Terry (λ_c) con la fuerza de cada rival fijada por un
ajuste BT global sobre TODOS los votos humanos (el modelo de imágenes no interviene en el objetivo). Se compara con
la tasa de victoria en fiabilidad, fidelidad del modelo (EXP-028) y anchura de los intervalos conformales (EXP-033).

Uso:  python exp038_bt_target.py        Salidas: outputs/exp038/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from exp026_additive_context import load_images
from exp028_neighborhood_scale import demean, rho
from probe_utils import cell_ids, load_pairs, load_split, train_scores
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp038"
PRIOR_VAR = 1.0
MIN_VOTES = 20
N_SPLIT = 200
N_BOOT = 1000
ALPHA = 0.10


def bt_global(pairs: pd.DataFrame) -> pd.Series:
    """BT MAP con prior N(0, PRIOR_VAR) sobre todas las imágenes (L-BFGS)."""
    ids = pd.Index(pd.unique(pd.concat([pairs["left"], pairs["right"]])))
    li, ri = ids.get_indexer(pairs["left"]), ids.get_indexer(pairs["right"])
    y = pairs["label"].to_numpy().astype(float)
    n = len(ids)

    def f(lam):
        d = lam[li] - lam[ri]
        p = expit(d)
        ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) - np.sum(lam ** 2) / (2 * PRIOR_VAR)
        r = y - p
        g = np.bincount(li, r, n) - np.bincount(ri, r, n) - lam / PRIOR_VAR
        return -ll, -g

    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    log(f"BT global: {n:,} imágenes, convergió={res.success}, iter={res.nit}, sd(λ)={res.x.std():.3f}")
    return pd.Series(res.x, index=ids)


def cell_map(unit: np.ndarray, opp: np.ndarray, y: np.ndarray, n_units: int, iters: int = 25):
    """MAP 1-D por unidad con rivales fijos (Newton agrupado). Devuelve λ_c y se de Laplace."""
    lam = np.zeros(n_units)
    for _ in range(iters):
        p = expit(lam[unit] - opp)
        g = np.bincount(unit, y - p, n_units) - lam / PRIOR_VAR
        h = np.bincount(unit, p * (1 - p), n_units) + 1 / PRIOR_VAR
        lam = lam + g / h
    p = expit(lam[unit] - opp)
    se = 1 / np.sqrt(np.bincount(unit, p * (1 - p), n_units) + 1 / PRIOR_VAR)
    return lam, se


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].to_numpy()
    pairs = load_pairs()
    lam_img = bt_global(pairs)

    ch = load_split("city_holdout")
    s, _ = train_scores(Z, list(ids), pairs, ch, "test_strict")
    st = pairs.iloc[ch["test_strict"]]
    held = set(st["city_left"]) | set(st["city_right"])
    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    cell = pd.Series(cell_ids(man, 1000).to_numpy(), index=ids)

    # un registro por voto de imagen retenida: celda, fuerza del rival, resultado
    tt = pairs.iloc[ch["test_touch"]]
    recs = []
    for side, other, win in (("left", "right", tt["label"].to_numpy()), ("right", "left", 1 - tt["label"].to_numpy())):
        keep = tt[side].isin(held_imgs).to_numpy()
        recs.append(pd.DataFrame({"cell": tt[side].to_numpy()[keep].astype(object),
                                  "opp": lam_img.reindex(tt[other].to_numpy()[keep]).to_numpy(), "y": win[keep]}))
    v = pd.concat(recs, ignore_index=True)
    v["cell"] = v["cell"].map(cell)
    cnt = v.groupby("cell").size()
    units = cnt[cnt >= MIN_VOTES].index
    v = v[v["cell"].isin(units)].reset_index(drop=True)
    u = pd.Index(units)
    ui = u.get_indexer(v["cell"])
    lam_c, se_c = cell_map(ui, v["opp"].to_numpy(), v["y"].to_numpy().astype(float), len(u))
    t = pd.DataFrame({"n": np.bincount(ui, minlength=len(u)), "winrate": np.bincount(ui, v["y"], len(u)) / np.bincount(ui, minlength=len(u)),
                      "bt": lam_c, "bt_se": se_c}, index=u)
    t["city"] = t.index.str.split("|").str[0]
    t["model"] = pd.Series(s, index=ids).groupby(cell).mean().reindex(t.index)
    log(f"celdas: {len(t):,} en {t['city'].nunique()} ciudades")

    # fiabilidad split-half (rivales fijos), pooled e intra-ciudad
    rel = {k: {"pooled": [], "within": []} for k in ("winrate", "bt")}
    c = t["city"].to_numpy()
    for _ in range(N_SPLIT):
        half = rng.random(len(v)) < 0.5
        out = {}
        for hname, m in (("a", half), ("b", ~half)):
            uu, yy, oo = ui[m], v["y"].to_numpy()[m].astype(float), v["opp"].to_numpy()[m]
            nn = np.bincount(uu, minlength=len(u))
            out[hname] = {"winrate": np.bincount(uu, yy, len(u)) / np.maximum(nn, 1),
                          "bt": cell_map(uu, oo, yy, len(u))[0], "ok": nn > 0}
        ok = out["a"]["ok"] & out["b"]["ok"]
        for k in ("winrate", "bt"):
            r = rho(out["a"][k][ok], out["b"][k][ok])
            rw = rho(demean(out["a"][k][ok], c[ok]), demean(out["b"][k][ok], c[ok]))
            rel[k]["pooled"].append(2 * r / (1 + r))
            rel[k]["within"].append(2 * rw / (1 + rw))
    rows = []
    by_city = [np.flatnonzero(c == x) for x in np.unique(c)]
    for k in ("winrate", "bt"):
        rw = rho(demean(t[k].to_numpy(), c), demean(t["model"].to_numpy(), c))
        B = [rho(demean(t[k].to_numpy()[ix], c[ix]), demean(t["model"].to_numpy()[ix], c[ix]))
             for ix in (np.concatenate([rng.choice(b, len(b)) for b in by_city]) for _ in range(N_BOOT))]
        rows.append(dict(target=k, reliability_pooled=np.mean(rel[k]["pooled"]), reliability_within=np.mean(rel[k]["within"]),
                         rho_model_within=rw, ci_low=np.percentile(B, 2.5), ci_high=np.percentile(B, 97.5),
                         rho_within_disattenuated=rw / np.sqrt(np.mean(rel[k]["within"]))))
    fid = pd.DataFrame(rows)
    log(fid.round(4).to_string(index=False))
    fid.to_csv(OUT / "reliability_fidelity.csv", index=False, float_format="%.4f")

    # conformal LOCO con ambos objetivos
    crow = []
    for k in ("winrate", "bt"):
        for city in sorted(t["city"].unique()):
            cal, te = t[t["city"] != city], t[t["city"] == city]
            b_, a_ = np.polyfit(cal["model"], cal[k], 1)
            yc, yt = a_ + b_ * cal["model"].to_numpy(), a_ + b_ * te["model"].to_numpy()
            if k == "winrate":
                yc, yt = np.clip(yc, 0.01, 0.99), np.clip(yt, 0.01, 0.99)
                noise_c, noise_t = yc * (1 - yc) / cal["n"].to_numpy(), yt * (1 - yt) / te["n"].to_numpy()
            else:
                noise_c, noise_t = cal["bt_se"].to_numpy() ** 2, te["bt_se"].to_numpy() ** 2
            tau2 = max(np.var(cal[k].to_numpy() - yc) - noise_c.mean(), 0.0)
            r = np.abs(cal[k].to_numpy() - yc) / np.sqrt(noise_c + tau2)
            q = np.quantile(r, min(1.0, np.ceil((len(r) + 1) * (1 - ALPHA)) / len(r)), method="higher")
            half = q * np.sqrt(noise_t + tau2)
            cov = float((np.abs(te[k].to_numpy() - yt) <= half).mean())
            crow.append(dict(target=k, city=city, n_cells=len(te), coverage=cov, mean_width=float(np.mean(2 * half)),
                             sd_target_city=float(te[k].std()), relative_width=float(np.mean(2 * half) / te[k].std())))
    conf = pd.DataFrame(crow)
    conf.to_csv(OUT / "conformal_by_city.csv", index=False, float_format="%.4f")
    agg = conf.groupby("target").agg(mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"),
                                     mean_relative_width=("relative_width", "mean"))
    log(agg.round(4).to_string())
    agg.to_csv(OUT / "conformal_summary.csv", float_format="%.4f")
    t.to_csv(OUT / "cells_bt.csv", float_format="%.5f")
    (OUT / "summary.json").write_text(json.dumps({"n_cells": len(t), "prior_var": PRIOR_VAR, "n_split": N_SPLIT,
                                                  "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
