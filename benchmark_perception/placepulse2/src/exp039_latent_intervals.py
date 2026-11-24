"""
EXP-039 — Validación del modelo de ruido del voto e intervalos para la percepción LATENTE del barrio.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-09).

Observado_celda = latente + ruido del voto (var ≈ p(1−p)/n). El conformal normalizado de EXP-033 usa
σ = √(p(1−p)/n + τ²). Si ese modelo de ruido es correcto, la cobertura no depende de n (nº de votos) y τ² es
estable entre ciudades; entonces el intervalo de la percepción LATENTE (lo que importa para una ciudad sin votos)
es ŷ ± q·τ.

Uso:  python exp039_latent_intervals.py        Salidas: outputs/exp039/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from exp026_additive_context import load_images
from exp033_conformal_maps import ALPHA, cells_table
from exp035_representation_v2 import fit, load_feature_sets, train_frames
from exp037_pipeline_v2 import score_all
from probe_utils import load_pairs, load_split, votes
from utils import MANIFESTS, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp039"


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cfg = json.loads((OUTPUTS / "exp035" / "selected_config.json").read_text(encoding="utf-8"))
    ids, sets = load_feature_sets("final")
    man = load_images("dinov2_vit_b14")[0]
    row = pd.Series(np.arange(len(ids)), index=ids)
    pairs, ties = load_pairs(), pd.read_csv(MANIFESTS / "pairs_safe_ties.csv", dtype=str)
    ch = load_split("city_holdout")
    va = pairs.iloc[ch["val"]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
    h = fit(sets[cfg["features"]], row, train_frames(pairs, ties, ch, bool(cfg["ties"])), va, cfg["head"], 1e-3, float(cfg["wd"]))
    s = score_all(h, sets[cfg["features"]])
    st = pairs.iloc[ch["test_strict"]]
    held = sorted(set(st["city_left"]) | set(st["city_right"]))
    v = votes(pairs, ch["test_touch"], keep=set(man.loc[man["city"].isin(held), "location_id"]))
    t = cells_table(man, s, v)
    t = t[t["city"].isin(held)]

    cells, folds = [], []
    for city in held:
        cal, te = t[t["city"] != city], t[t["city"] == city]
        b, a = np.polyfit(cal["model"], cal["human"], 1)
        yc = np.clip(a + b * cal["model"].to_numpy(), 0.01, 0.99)
        yt = np.clip(a + b * te["model"].to_numpy(), 0.01, 0.99)
        tau2 = max(np.var(cal["human"] - yc) - np.mean(yc * (1 - yc) / cal["n"]), 0.0)
        r = np.abs(cal["human"].to_numpy() - yc) / np.sqrt(yc * (1 - yc) / cal["n"].to_numpy() + tau2)
        q = np.quantile(r, min(1.0, np.ceil((len(r) + 1) * (1 - ALPHA)) / len(r)), method="higher")
        sig_t = np.sqrt(yt * (1 - yt) / te["n"].to_numpy() + tau2)
        cells.append(te.assign(pred=yt, covered=np.abs(te["human"].to_numpy() - yt) <= q * sig_t,
                               width_obs=2 * q * sig_t, width_latent=2 * q * np.sqrt(tau2)))
        folds.append(dict(city=city, tau2=tau2, q=q, sd_human_city=float(te["human"].std())))
    c = pd.concat(cells)
    f = pd.DataFrame(folds)
    c["votes_tercile"] = pd.qcut(c["n"], 3, labels=["pocos", "medios", "muchos"])
    terc = c.groupby("votes_tercile", observed=True).agg(n_cells=("covered", "size"), coverage=("covered", "mean"),
                                                         median_votes=("n", "median"), mean_width_obs=("width_obs", "mean"))
    cv_tau2 = float(f["tau2"].std() / f["tau2"].mean())
    rel = pd.DataFrame({"city": f["city"], "width_latent": 2 * f["q"] * np.sqrt(f["tau2"]),
                        "width_observed": c.groupby("city")["width_obs"].mean().reindex(f["city"]).to_numpy(),
                        "sd_between_cells": f["sd_human_city"]})
    rel["relative_latent"] = rel["width_latent"] / rel["sd_between_cells"]
    rel["relative_observed"] = rel["width_observed"] / rel["sd_between_cells"]
    log(terc.round(4).to_string())
    log(f"CV(τ²) entre pliegues = {cv_tau2:.3f}; τ medio = {np.sqrt(f['tau2']).mean():.4f}")
    log(rel.round(3).to_string(index=False))
    terc.to_csv(OUT / "coverage_by_votes_tercile.csv", float_format="%.4f")
    f.to_csv(OUT / "tau2_by_fold.csv", index=False, float_format="%.5f")
    rel.to_csv(OUT / "latent_vs_observed_width.csv", index=False, float_format="%.4f")
    passed = bool(terc["coverage"].between(0.85, 0.95).all() and cv_tau2 < 0.20)
    (OUT / "summary.json").write_text(json.dumps({
        "config": cfg, "coverage_by_tercile": terc["coverage"].round(4).to_dict(), "cv_tau2": cv_tau2,
        "mean_relative_width_latent": float(rel["relative_latent"].mean()),
        "mean_relative_width_observed": float(rel["relative_observed"].mean()),
        "criterion_passed": passed, "seconds": round(time.perf_counter() - t0, 1)}, indent=2, default=str), encoding="utf-8")
    log(f"criterio cumplido: {passed}")


if __name__ == "__main__":
    main()
