"""
EXP-033 — Mapa de percepción por barrio con incertidumbre calibrada en ciudades NO vistas (conformal).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-09).

Para entregar un mapa por barrio a una ciudad sin votos (p. ej. Lima) no basta con una buena correlación: cada
celda necesita un intervalo cuya cobertura se cumpla en una ciudad NUEVA. Se usa conformal split normalizado,
calibrado dejando fuera una ciudad cada vez entre las 10 ciudades retenidas de `city_holdout`.

Uso:  python exp033_conformal_maps.py        Salidas: outputs/exp033/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from exp026_additive_context import load_images
from probe_utils import cell_ids, load_pairs, load_split, local_xy, train_scores, votes
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp033"
ALPHA = 0.10
MIN_VOTES = 20
SIDE = 1000


def cells_table(man, s, v) -> pd.DataFrame:
    xy = local_xy(man)
    base = pd.DataFrame({"id": man["location_id"], "city": man["city"], "cell": cell_ids(man, SIDE),
                         "s": s, "x": xy[:, 0], "y": xy[:, 1]})
    agg = base.groupby("cell").agg(city=("city", "first"), model=("s", "mean"), x=("x", "mean"), y=("y", "mean"))
    vv = v.assign(cell=v["id"].map(base.set_index("id")["cell"]))
    h = vv.groupby("cell")["win"].agg(n="size", human="mean")
    t = agg.join(h, how="inner")
    return t[t["n"] >= MIN_VOTES]


def conformal_loco(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cells = [], []
    for city in sorted(t["city"].unique()):
        cal, te = t[t["city"] != city], t[t["city"] == city]
        b, a = np.polyfit(cal["model"], cal["human"], 1)
        fit = lambda d: np.clip(a + b * d["model"].to_numpy(), 0.01, 0.99)
        yc = fit(cal)
        tau2 = max(np.var(cal["human"] - yc) - np.mean(yc * (1 - yc) / cal["n"]), 0.0)
        sig = lambda yh, n: np.sqrt(yh * (1 - yh) / n + tau2)
        r = np.abs(cal["human"].to_numpy() - yc) / sig(yc, cal["n"].to_numpy())
        m = len(r)
        q = np.quantile(r, min(1.0, np.ceil((m + 1) * (1 - ALPHA)) / m), method="higher")
        yt = fit(te)
        half = q * sig(yt, te["n"].to_numpy())
        inside = np.abs(te["human"].to_numpy() - yt) <= half
        rows.append(dict(city=city, n_cells=len(te), coverage=inside.mean(), mean_width=float(np.mean(2 * half)),
                         tau2=tau2, q=q, slope=b, intercept=a,
                         rho_model_human=float(pd.Series(yt).corr(te["human"].reset_index(drop=True), method="spearman"))))
        cells.append(te.assign(pred=yt, lo=yt - half, hi=yt + half, covered=inside))
        log(f"  {city:<15} celdas={len(te):4d} cobertura={inside.mean():.3f} ancho medio={np.mean(2 * half):.3f}")
    return pd.DataFrame(rows), pd.concat(cells)


def maps(cells: pd.DataFrame, cities: list[str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cities), 3, figsize=(12.5, 4.1 * len(cities)), dpi=150)
    for r, city in enumerate(cities):
        c = cells[cells["city"] == city]
        vmin, vmax = np.percentile(np.r_[c["pred"], c["human"]], [5, 95])
        for k, (col, title, cmap, lim) in enumerate([
                ("pred", "Predicción del modelo (ciudad no vista)", "Blues", (vmin, vmax)),
                ("human", "Juicio humano (votos Place Pulse)", "Blues", (vmin, vmax)),
                ("width", "Ancho del intervalo conformal 90 %", "Greys", None)]):
            ax = axes[r, k]
            vals = c["hi"] - c["lo"] if col == "width" else c[col]
            sc = ax.scatter(c["x"] / 1000, c["y"] / 1000, c=vals, cmap=cmap, s=26, marker="s",
                            vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, linewidths=0)
            miss = c[~c["covered"]]
            if col == "human" and len(miss):
                ax.scatter(miss["x"] / 1000, miss["y"] / 1000, s=40, marker="s", facecolors="none",
                           edgecolors="#e34948", linewidths=0.9, label="fuera del intervalo")
                ax.legend(loc="lower left", fontsize=7, frameon=False)
            ax.set_title(f"{city} · {title}", fontsize=9, loc="left")
            ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=7)
    fig.suptitle("EXP-033 · Percepción de seguridad por celda de 1 km en ciudades NO vistas en entrenamiento "
                 "(más oscuro = percibido más seguro)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    man, Z = load_images("dinov2_vit_b14")
    pairs = load_pairs()
    ch = load_split("city_holdout")
    s, info = train_scores(Z, man["location_id"].tolist(), pairs, ch, "test_strict")
    log(f"probe city_holdout: test_strict acc {info['metrics']['pairwise_accuracy']:.4f}")
    st = pairs.iloc[ch["test_strict"]]
    held = sorted(set(st["city_left"]) | set(st["city_right"]))
    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    v = votes(pairs, ch["test_touch"], keep=held_imgs)
    t = cells_table(man, s, v)
    t = t[t["city"].isin(held)]
    res, cells = conformal_loco(t)
    res.to_csv(OUT / "conformal_by_city.csv", index=False, float_format="%.4f")
    cells.to_csv(OUT / "cells_with_intervals.csv", float_format="%.5f")
    maps(cells, ["Santiago", "Rio De Janeiro"], OUT / "exp033_maps.png")
    summ = dict(alpha=ALPHA, side_m=SIDE, min_votes=MIN_VOTES, held_out_cities=held,
                mean_coverage=float(res["coverage"].mean()),
                cities_coverage_ge_080=int((res["coverage"] >= 0.80).sum()),
                weighted_coverage=float(np.average(res["coverage"], weights=res["n_cells"])),
                mean_width=float(res["mean_width"].mean()), seconds=round(time.perf_counter() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
