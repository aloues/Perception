"""
EXP-063 Track D — ¿El orden de las cabezas se mantiene a escala de LUGAR? (celdas de 1 km en las 10 ciudades retenidas
de city_holdout; ninguna imagen ni voto de esas ciudades se usó para entrenar ni elegir hiperparámetros).

Objetivo humano de la celda: media de λ_test (Bradley–Terry ajustado SOLO con los pares test_strict, ambos extremos en
ciudades retenidas) de sus imágenes; celdas con ≥ MIN_IMG imágenes con votos. Predicción: media de g(z) de las mismas
imágenes. Métrica: ρ de Spearman DENTRO de cada ciudad, promediada (y global). IC 95 % por bootstrap de celdas
estratificado por ciudad; Δρ vs lineal-BT pareado (mismas celdas remuestreadas).

Uso: python src/exp063_place.py        Salida: outputs/exp063/place_cells_city_holdout.csv (+ .md)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp063_heads import bt_fit  # noqa: E402
from probe_utils import cell_ids  # noqa: E402
from utils import FEATURES, MANIFESTS, OUTPUTS, log  # noqa: E402

OD = OUTPUTS / "exp063" / "city_holdout"
MIN_IMG = 5
SIDE = 1000
B = 1000
COORD_RE = re.compile(r"images/(-?\d+\.\d+)_(-?\d+\.\d+)_")


def main() -> None:
    enc, rep = "dinov2_vit_b14", "z"
    rows = np.load(OD / f"rows_{enc}_{rep}.npy")
    ids = json.loads((FEATURES / enc / "index.json").read_text())["location_ids"]
    gid = np.array([ids[r] for r in rows])
    tp = np.load(OD / f"test_pairs_{enc}_{rep}.npz")
    trows, tlam = bt_fit(tp["li"], tp["ri"], tp["y"])
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).set_index("location_id")
    sub = man.loc[gid[trows]].copy()
    ll = sub["image_path"].str.extract(COORD_RE).astype(float)
    sub["lat"], sub["lon"] = ll[0].to_numpy(), ll[1].to_numpy()
    sub["cell"] = cell_ids(sub.reset_index(), SIDE).to_numpy()
    sub["lam"] = tlam
    sub["k"] = trows
    cnt = sub.groupby("cell")["lam"].transform("size")
    sub = sub[cnt >= MIN_IMG]
    cells = sub.groupby("cell").agg(city=("city", "first"), lam=("lam", "mean"), n=("lam", "size")).reset_index()
    log(f"{len(cells)} celdas (≥{MIN_IMG} imágenes) en {cells.city.nunique()} ciudades, {len(sub)} imágenes")
    recs = [json.loads(l) for l in (OD / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    heads = [(r["head"], r["track"]) for r in recs if "error" not in r and r["seed"] == 42 and r["rep"] == rep]
    cell_idx = pd.Index(cells["cell"])
    ci = cell_idx.get_indexer(sub["cell"])
    preds = {}
    for h, _ in heads:
        g = np.load(OD / "scores" / f"{enc}__{rep}__{h}__s42.npy").astype(float)
        preds[h] = np.bincount(ci, g[sub["k"].to_numpy()], len(cells)) / np.bincount(ci, None, len(cells))
    city_of = cells["city"].to_numpy()
    cities = sorted(set(city_of))
    by_city = {c: np.where(city_of == c)[0] for c in cities}
    rng = np.random.default_rng(42)
    boots = [{c: rng.choice(ix, len(ix)) for c, ix in by_city.items()} for _ in range(B)]

    def intra(pred, sel):
        return float(np.nanmean([spearmanr(pred[ix], cells["lam"].to_numpy()[ix]).correlation
                                 for ix in sel.values() if len(ix) >= 8]))

    base_b = np.array([intra(preds["linear_bt"], s) for s in boots])
    out = []
    for h, t in heads:
        rb = np.array([intra(preds[h], s) for s in boots])
        r = {"head": h, "track": t, "n_cells": len(cells), "n_cities": len(cities),
             "rho_intra_city": intra(preds[h], by_city), "rho_lo": np.percentile(rb, 2.5),
             "rho_hi": np.percentile(rb, 97.5),
             "rho_pooled": float(spearmanr(preds[h], cells["lam"]).correlation)}
        if h != "linear_bt":
            dlt = rb - base_b
            r.update({"delta_vs_linear": r["rho_intra_city"] - intra(preds["linear_bt"], by_city),
                      "delta_lo": np.percentile(dlt, 2.5), "delta_hi": np.percentile(dlt, 97.5)})
        out.append(r)
    df = pd.DataFrame(out).sort_values("rho_intra_city", ascending=False)
    df.to_csv(OUTPUTS / "exp063" / "place_cells_city_holdout.csv", index=False)
    with open(OUTPUTS / "exp063" / "place_cells_city_holdout.md", "w", encoding="utf-8") as f:
        f.write(f"Celdas de {SIDE} m con ≥ {MIN_IMG} imágenes con votos de test; {len(cells)} celdas, {len(cities)} "
                f"ciudades retenidas. Objetivo = media de λ_BT (solo votos test_strict).\n\n")
        f.write("| Cabeza | Track | ρ intra-ciudad [IC 95 %] | ρ global | Δρ vs lineal [IC 95 %] |\n|---|---|---|---|---|\n")
        for r in df.itertuples():
            dd = "—" if pd.isna(getattr(r, "delta_vs_linear", np.nan)) else \
                f"{r.delta_vs_linear:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
            f.write(f"| {r.head} | {r.track} | {r.rho_intra_city:.3f} [{r.rho_lo:.3f}, {r.rho_hi:.3f}] | "
                    f"{r.rho_pooled:.3f} | {dd} |\n")
    log(df.to_string())


if __name__ == "__main__":
    main()
