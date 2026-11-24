"""
EXP-034 — ¿La ventaja del encoder (RQ1) se mantiene a escala de BARRIO en ciudades no vistas?

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-09). Mismas celdas, votos y probe
que EXP-028 (`city_holdout`, celdas de 1 km con >= 20 votos); solo cambia el encoder.

Uso:  python exp034_encoders_neighborhood.py        Salidas: outputs/exp034/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from exp026_additive_context import load_images
from exp028_neighborhood_scale import demean, rho
from probe_utils import cell_ids, load_pairs, load_split, train_scores, votes
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp034"
ENCODERS = ["dinov2_vit_b14", "imagenet_vit_b16", "ours_simclr_vit_b16"]
N_BOOT = 1000


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()
    ch = load_split("city_holdout")
    st = pairs.iloc[ch["test_strict"]]
    held = set(st["city_left"]) | set(st["city_right"])
    scores, acc, man0 = {}, {}, None
    for enc in ENCODERS:
        man, Z = load_images(enc)
        if man0 is None:
            man0 = man
        assert (man["location_id"].to_numpy() == man0["location_id"].to_numpy()).all()
        s, info = train_scores(Z, man["location_id"].tolist(), pairs, ch, "test_strict")
        scores[enc], acc[enc] = s, info["metrics"]["pairwise_accuracy"]
        log(f"  {enc:<22} test_strict acc {acc[enc]:.4f}")
    man = man0
    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    v = votes(pairs, ch["test_touch"], keep=held_imgs)
    cell = pd.Series(cell_ids(man, 1000).to_numpy(), index=man["location_id"].to_numpy())
    vv = v.assign(cell=v["id"].map(cell))
    t = vv.groupby("cell")["win"].agg(n="size", human="mean")
    t = t[t["n"] >= 20]
    t["city"] = t.index.str.split("|").str[0]
    for enc in ENCODERS:
        t[enc] = pd.Series(scores[enc], index=man["location_id"].to_numpy()).groupby(cell).mean().reindex(t.index)
    t = t.dropna()

    def wc(d, enc):
        return rho(demean(d["human"].to_numpy(), d["city"].to_numpy()), demean(d[enc].to_numpy(), d["city"].to_numpy()))

    point = {enc: wc(t, enc) for enc in ENCODERS}
    by_city = [g.index.to_numpy() for _, g in t.groupby("city")]
    B = []
    for _ in range(N_BOOT):
        d = t.loc[np.concatenate([rng.choice(c, len(c)) for c in by_city])]
        B.append({enc: wc(d, enc) for enc in ENCODERS})
    B = pd.DataFrame(B)
    rows = [dict(encoder=e, pair_acc_test_strict=acc[e], rho_within_city=point[e],
                 ci_low=np.percentile(B[e], 2.5), ci_high=np.percentile(B[e], 97.5)) for e in ENCODERS]
    for a, b in (("dinov2_vit_b14", "ours_simclr_vit_b16"), ("dinov2_vit_b14", "imagenet_vit_b16"),
                 ("imagenet_vit_b16", "ours_simclr_vit_b16")):
        d = B[a] - B[b]
        rows.append(dict(encoder=f"{a} - {b}", rho_within_city=point[a] - point[b],
                         ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    res = pd.DataFrame(rows)
    log(res.round(4).to_string(index=False))
    res.to_csv(OUT / "encoders_neighborhood.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps({"n_cells": len(t), "n_cities": int(t["city"].nunique()),
                                                  "seconds": round(time.perf_counter() - t0, 1)}, indent=2),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
