"""
EXP-028 — ¿Es fiable la predicción de percepción a escala de BARRIO aunque sea ruidosa a escala de imagen?

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa EXP-028..031, 2026-10-09).

Idea (R1 del asesor): no afirmar "esta esquina es peligrosa", sino "esta zona podría percibirse como insegura".
Para que esa afirmación sea defendible hay que medir, con imágenes y ciudades NO vistas, qué tan bien el
modelo reproduce el juicio humano agregado por área, comparado con lo máximo alcanzable dado el ruido de los
votos (techo split-half), y controlar que no sea solo un efecto "ciudad".

Uso:  python exp028_neighborhood_scale.py        Salidas: outputs/exp028/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp026_additive_context import load_images
from probe_utils import cell_ids, load_pairs, load_split, train_scores, votes
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp028"
SIDES = [250, 500, 1000, 2000]
MIN_VOTES_CELL = 20
MIN_VOTES_IMG = 3
N_SPLIT = 200
N_BOOT = 1000
MIN_CELLS_CITY = 10


def rho(a, b) -> float:
    return float(spearmanr(a, b)[0]) if len(a) > 2 else np.nan


def demean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return (s - s.groupby(groups).transform("mean")).to_numpy()


def unit_table(v: pd.DataFrame, unit_of: pd.Series, score: pd.Series, city_of: pd.Series,
               min_votes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega votos y puntajes del modelo por unidad (imagen o celda)."""
    vv = v.assign(unit=v["id"].map(unit_of))
    g = vv.groupby("unit").agg(n=("win", "size"), human=("win", "mean"))
    g = g[g["n"] >= min_votes]
    ids_in = pd.Series(unit_of.index, index=unit_of.index)
    model = score.groupby(unit_of).mean()
    g["model"] = model.reindex(g.index)
    g["city"] = city_of.groupby(unit_of).first().reindex(g.index)
    return g.dropna(), vv[vv["unit"].isin(g.index)]


def split_half(vv: pd.DataFrame, units: pd.DataFrame, rng, within_city: bool) -> float:
    """Fiabilidad split-half (Spearman-Brown) del puntaje humano por unidad."""
    rs = []
    order = units.index
    for _ in range(N_SPLIT):
        half = rng.random(len(vv)) < 0.5
        a = vv[half].groupby("unit")["win"].mean().reindex(order)
        b = vv[~half].groupby("unit")["win"].mean().reindex(order)
        ok = a.notna() & b.notna()
        x, y = a[ok].to_numpy(), b[ok].to_numpy()
        if within_city:
            c = units.loc[ok[ok].index, "city"].to_numpy()
            x, y = demean(x, c), demean(y, c)
        r = rho(x, y)
        rs.append(2 * r / (1 + r) if r > -1 else np.nan)
    return float(np.nanmean(rs))


def boot(units: pd.DataFrame, rng, within_city: bool) -> tuple[float, float]:
    vals = []
    by_city = [g.index.to_numpy() for _, g in units.groupby("city")]
    for _ in range(N_BOOT):
        ix = np.concatenate([rng.choice(c, len(c)) for c in by_city])   # estratificado por ciudad
        u = units.loc[ix]
        if within_city:
            vals.append(rho(demean(u["human"].to_numpy(), u["city"].to_numpy()),
                            demean(u["model"].to_numpy(), u["city"].to_numpy())))
        else:
            vals.append(rho(u["human"], u["model"]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def analyze(protocol: str, man, s: np.ndarray, v: pd.DataFrame, rng) -> list[dict]:
    score = pd.Series(s, index=man["location_id"].to_numpy())
    city_of = pd.Series(man["city"].to_numpy(), index=man["location_id"].to_numpy())
    rows = []
    levels = [("imagen", pd.Series(man["location_id"].to_numpy(), index=man["location_id"].to_numpy()),
               MIN_VOTES_IMG)]
    levels += [(f"celda_{L}m", pd.Series(cell_ids(man, L).to_numpy(), index=man["location_id"].to_numpy()),
                MIN_VOTES_CELL) for L in SIDES]
    for name, unit_of, mv in levels:
        units, vv = unit_table(v, unit_of, score, city_of, mv)
        r_all = rho(units["human"], units["model"])
        r_wc = rho(demean(units["human"].to_numpy(), units["city"].to_numpy()),
                   demean(units["model"].to_numpy(), units["city"].to_numpy()))
        per_city = [rho(g["human"], g["model"]) for _, g in units.groupby("city") if len(g) >= MIN_CELLS_CITY]
        rel_all = split_half(vv, units, rng, within_city=False)
        rel_wc = split_half(vv, units, rng, within_city=True)
        lo, hi = boot(units, rng, False)
        lo_w, hi_w = boot(units, rng, True)
        row = dict(protocol=protocol, level=name, n_units=len(units), n_cities=units["city"].nunique(),
                   median_votes=float(units["n"].median()),
                   rho=r_all, rho_ci_low=lo, rho_ci_high=hi, reliability=rel_all,
                   rho_disattenuated=r_all / np.sqrt(rel_all) if rel_all > 0 else np.nan,
                   rho_within_city=r_wc, rho_within_ci_low=lo_w, rho_within_ci_high=hi_w,
                   reliability_within_city=rel_wc,
                   rho_within_disattenuated=r_wc / np.sqrt(rel_wc) if rel_wc > 0 else np.nan,
                   mean_rho_per_city=float(np.nanmean(per_city)) if per_city else np.nan,
                   n_cities_rho=len(per_city))
        rows.append(row)
        log(f"  {protocol:<14} {name:<11} n={len(units):6d}  ρ={r_all:.3f} [{lo:.3f},{hi:.3f}] "
            f"fiab={rel_all:.3f} ρ*={row['rho_disattenuated']:.3f} | intra-ciudad ρ={r_wc:.3f} "
            f"[{lo_w:.3f},{hi_w:.3f}] fiab={rel_wc:.3f} ρ*={row['rho_within_disattenuated']:.3f}")
    return rows


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].tolist()
    pairs = load_pairs()
    rows, sanity = [], {}

    sp = load_split("image_disjoint")
    s, info = train_scores(Z, ids, pairs, sp, "test")
    sanity["image_disjoint"] = info["metrics"]
    log(f"probe image_disjoint: test acc {info['metrics']['pairwise_accuracy']:.4f}")
    test_imgs = set(votes(pairs, sp["test"])["id"])
    v = votes(pairs, sp["test"], keep=test_imgs)
    rows += analyze("image_disjoint", man, s, v, rng)

    ch = load_split("city_holdout")
    s2, info2 = train_scores(Z, ids, pairs, ch, "test_strict")
    sanity["city_holdout"] = info2["metrics"]
    log(f"probe city_holdout: test_strict acc {info2['metrics']['pairwise_accuracy']:.4f}")
    st = pairs.iloc[ch["test_strict"]]
    held = set(st["city_left"]) | set(st["city_right"])
    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    v2 = votes(pairs, ch["test_touch"], keep=held_imgs)
    rows += analyze("city_holdout", man, s2, v2, rng)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "neighborhood_scale.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps({
        "seed": SEED, "sides_m": SIDES, "min_votes_cell": MIN_VOTES_CELL, "min_votes_img": MIN_VOTES_IMG,
        "n_split_half": N_SPLIT, "n_boot": N_BOOT, "held_out_cities": sorted(held),
        "probe_sanity": sanity, "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
