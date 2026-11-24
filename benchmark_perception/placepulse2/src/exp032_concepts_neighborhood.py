"""
EXP-032 — ¿Qué conceptos explican las diferencias de percepción ENTRE BARRIOS? (une RQ2 y RQ3)

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-09). Requiere EXP-031
(outputs/exp031/concept_scores_std.csv.gz). Ajuste en celdas de 1 km de las ciudades de TRAIN de
`city_holdout`; evaluación en las 10 ciudades retenidas (mismas celdas que EXP-028/033).

Uso:  python exp032_concepts_neighborhood.py        Salidas: outputs/exp032/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV

from exp026_additive_context import load_images
from exp028_neighborhood_scale import demean, rho
from exp031_concept_bottleneck import EXPECTED
from probe_utils import cell_ids, load_pairs, load_split, train_scores, votes
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp032"
MIN_VOTES = 20
N_BOOT = 1000
N_BOOT_COEF = 300
ALPHAS = np.logspace(-3, 3, 13)


def cell_table(v: pd.DataFrame, cell: pd.Series) -> pd.DataFrame:
    t = v.assign(cell=v["id"].map(cell)).groupby("cell")["win"].agg(n="size", human="mean")
    t = t[t["n"] >= MIN_VOTES]
    t["city"] = t.index.str.split("|").str[0]
    return t


def fit(X, y, groups):
    gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=GroupKFold(5), scoring="neg_mean_squared_error")
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def wc(h, p, c):
    return rho(demean(np.asarray(h), np.asarray(c)), demean(np.asarray(p), np.asarray(c)))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=str(OUTPUTS / "exp031" / "concept_scores_std.csv.gz"))
    ap.add_argument("--out", default="exp032")
    args = ap.parse_args()
    OUT = OUTPUTS / args.out
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].to_numpy()
    F = pd.read_csv(Path(args.concepts)).set_index("location_id").reindex(ids)
    names = [c for c in F.columns]
    pairs = load_pairs()
    ch = load_split("city_holdout")
    s, _ = train_scores(Z, list(ids), pairs, ch, "test_strict")
    st = pairs.iloc[ch["test_strict"]]
    held = set(st["city_left"]) | set(st["city_right"])
    cell = pd.Series(cell_ids(man, 1000).to_numpy(), index=ids)

    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    train_imgs = set(man.loc[~man["city"].isin(held), "location_id"])
    tr = cell_table(votes(pairs, ch["train"], keep=train_imgs), cell)
    te = cell_table(votes(pairs, ch["test_touch"], keep=held_imgs), cell)
    Fc = F.groupby(cell).mean()
    P = pd.Series(s, index=ids).groupby(cell).mean()
    log(f"celdas de ajuste {len(tr):,} ({tr['city'].nunique()} ciudades) | celdas de test {len(te):,} "
        f"({te['city'].nunique()} ciudades)")

    Xtr, Xte = Fc.reindex(tr.index).to_numpy(), Fc.reindex(te.index).to_numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    m_c = fit(Xtr, tr["human"].to_numpy(), tr["city"].to_numpy())
    Ptr, Pte = P.reindex(tr.index).to_numpy()[:, None], P.reindex(te.index).to_numpy()[:, None]
    pm, ps = Ptr.mean(), Ptr.std()
    m_cp = fit(np.hstack([Xtr, (Ptr - pm) / ps]), tr["human"].to_numpy(), tr["city"].to_numpy())
    # sensibilidad declarada: ajuste con objetivo y features centrados por ciudad (asociación intra-ciudad)
    Xtr_d = Xtr - pd.DataFrame(Xtr).groupby(tr["city"].to_numpy()).transform("mean").to_numpy()
    m_cw = fit(Xtr_d, demean(tr["human"].to_numpy(), tr["city"].to_numpy()), tr["city"].to_numpy())

    preds = {"C_conceptos": m_c.predict(Xte), "P_probe": Pte.ravel(),
             "CP_conceptos+probe": m_cp.predict(np.hstack([Xte, (Pte - pm) / ps])),
             "C_conceptos_intraciudad": m_cw.predict(Xte)}
    h, c = te["human"].to_numpy(), te["city"].to_numpy()
    pd.DataFrame({"human": te["human"], "city": te["city"], **{k: v for k, v in preds.items()}}, index=te.index).to_csv(OUT / "cell_predictions.csv", float_format="%.6f")
    point = {k: wc(h, p, c) for k, p in preds.items()}
    by_city = [np.flatnonzero(c == x) for x in np.unique(c)]
    B = []
    for _ in range(N_BOOT):
        ix = np.concatenate([rng.choice(b, len(b)) for b in by_city])
        B.append({k: wc(h[ix], p[ix], c[ix]) for k, p in preds.items()})
    B = pd.DataFrame(B)
    rows = [dict(model=k, rho_within_city=point[k], ci_low=np.percentile(B[k], 2.5),
                 ci_high=np.percentile(B[k], 97.5)) for k in preds]
    for a, b in (("C_conceptos", "P_probe"), ("CP_conceptos+probe", "P_probe")):
        d = B[a] - B[b]
        rows.append(dict(model=f"{a} - {b}", rho_within_city=point[a] - point[b],
                         ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    res = pd.DataFrame(rows)
    res["ratio_C_over_P"] = point["C_conceptos"] / point["P_probe"]
    log(res.round(4).to_string(index=False))
    res.to_csv(OUT / "concepts_neighborhood.csv", index=False, float_format="%.4f")

    # coeficientes estandarizados con IC por bootstrap de ciudades (clusters)
    cities = tr["city"].unique()
    coefs = []
    for _ in range(N_BOOT_COEF):
        pick = rng.choice(cities, len(cities))
        ix = np.concatenate([np.flatnonzero(tr["city"].to_numpy() == x) for x in pick])
        coefs.append(Ridge(alpha=m_c.alpha).fit(Xtr[ix], tr["human"].to_numpy()[ix]).coef_)
    coefs = np.array(coefs)
    cw = pd.DataFrame({"concept": names, "coef": m_c.coef_, "ci_low": np.percentile(coefs, 2.5, 0),
                       "ci_high": np.percentile(coefs, 97.5, 0), "coef_intraciudad": m_cw.coef_,
                       "expected_sign": [EXPECTED.get(n, 0) for n in names]})
    d = cw[cw["expected_sign"] != 0]
    ok = int((np.sign(d["coef"]) == d["expected_sign"]).sum())
    ok_w = int((np.sign(d["coef_intraciudad"]) == d["expected_sign"]).sum())
    cw.to_csv(OUT / "concept_coefficients.csv", index=False, float_format="%.4f")
    log(cw.round(4).to_string(index=False))

    # explicación por celda (Santiago, Río): 2 conceptos que más alejan la celda de la media de su ciudad
    expl = []
    for city in ("Santiago", "Rio De Janeiro"):
        sub = te[te["city"] == city]
        Xs = (Fc.reindex(sub.index).to_numpy() - mu) / sd
        contrib = (Xs - Xs.mean(0)) * m_c.coef_
        for k, cid in enumerate(sub.index):
            order = np.argsort(contrib[k])
            expl.append(dict(city=city, cell=cid, human=sub.loc[cid, "human"], pred=preds["C_conceptos"][te.index.get_loc(cid)],
                             most_negative=f"{names[order[0]]} ({contrib[k][order[0]]:+.3f})",
                             second_negative=f"{names[order[1]]} ({contrib[k][order[1]]:+.3f})",
                             most_positive=f"{names[order[-1]]} ({contrib[k][order[-1]]:+.3f})"))
    pd.DataFrame(expl).to_csv(OUT / "cell_explanations_santiago_rio.csv", index=False, float_format="%.4f")

    summ = dict(n_train_cells=len(tr), n_test_cells=len(te), alpha_C=float(m_c.alpha),
                rho_C=point["C_conceptos"], rho_P=point["P_probe"], ratio=point["C_conceptos"] / point["P_probe"],
                signs_ok=ok, signs_total=int(len(d)), sign_test_p=float(binomtest(ok, len(d), 0.5, alternative="greater").pvalue),
                signs_ok_intraciudad=ok_w, seconds=round(time.perf_counter() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
