"""
EXP-037 — Pipeline v2 a escala de barrio: propaga la representación elegida en validación (EXP-035,
outputs/exp035/selected_config.json) a EXP-028 (ρ intra-ciudad por celda de 1 km, ciudades no vistas), EXP-033
(conformal LOCO) y EXP-029 (kNN entre ciudades). Compara contra v1 (DINOv2 CLS + probe lineal fijo) en pareado.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa v2, 2026-10-09).

Uso:  python exp037_pipeline_v2.py        Salidas: outputs/exp037/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from exp028_neighborhood_scale import demean, rho
from exp029_retrieval import knn_scores, pair_eval, paired_delta
from exp033_conformal_maps import cells_table, conformal_loco
from exp035_representation_v2 import fit, load_feature_sets, train_frames
from probe_utils import load_pairs, load_split, votes
from utils import MANIFESTS, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp037"
N_BOOT = 1000


@torch.no_grad()
def score_all(h, X):
    return np.concatenate([h.score_images(torch.from_numpy(X[a:a + 8192])).numpy() for a in range(0, len(X), 8192)])


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    cfg = json.loads((OUTPUTS / "exp035" / "selected_config.json").read_text(encoding="utf-8"))
    log(f"configuración v2 (elegida en val): {cfg}")
    ids, sets = load_feature_sets("final")
    from exp026_additive_context import load_images
    man = load_images("dinov2_vit_b14")[0]
    assert man["location_id"].tolist() == ids
    row = pd.Series(np.arange(len(ids)), index=ids)
    pairs, ties = load_pairs(), pd.read_csv(MANIFESTS / "pairs_safe_ties.csv", dtype=str)
    ch = load_split("city_holdout")
    va = pairs.iloc[ch["val"]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
    variants = {"v1": ("dinov2", "linear", 1e-4, False),
                "v2": (cfg["features"], cfg["head"], float(cfg["wd"]), bool(cfg["ties"]))}
    scores = {}
    for k, (f, hd, wd, tie) in variants.items():
        h = fit(sets[f], row, train_frames(pairs, ties, ch, tie), va, hd, 1e-3, wd)
        scores[k] = score_all(h, sets[f])
        log(f"  {k}: {f} {hd} wd={wd} empates={tie} entrenado ({time.perf_counter() - t0:.0f}s)")

    # ── barrio: ρ intra-ciudad + conformal ─────────────────────────────────────────────────
    st = pairs.iloc[ch["test_strict"]]
    held = sorted(set(st["city_left"]) | set(st["city_right"]))
    held_imgs = set(man.loc[man["city"].isin(held), "location_id"])
    v = votes(pairs, ch["test_touch"], keep=held_imgs)
    tabs = {k: cells_table(man, s, v) for k, s in scores.items()}
    t1 = tabs["v1"]
    t2 = tabs["v2"].reindex(t1.index)
    h_, c_ = t1["human"].to_numpy(), t1["city"].to_numpy()
    wc = lambda p, ix: rho(demean(h_[ix], c_[ix]), demean(p[ix], c_[ix]))
    allix = np.arange(len(h_))
    by = [np.flatnonzero(c_ == x) for x in np.unique(c_)]
    B = [(wc(t1["model"].to_numpy(), ix), wc(t2["model"].to_numpy(), ix))
         for ix in (np.concatenate([rng.choice(b, len(b)) for b in by]) for _ in range(N_BOOT))]
    B = np.array(B)
    hood = dict(rho_within_v1=wc(t1["model"].to_numpy(), allix), rho_within_v2=wc(t2["model"].to_numpy(), allix),
                delta=wc(t2["model"].to_numpy(), allix) - wc(t1["model"].to_numpy(), allix),
                ci_low=float(np.percentile(B[:, 1] - B[:, 0], 2.5)), ci_high=float(np.percentile(B[:, 1] - B[:, 0], 97.5)))
    log(f"  barrio 1 km intra-ciudad: {hood}")
    conf = {}
    for k, t in tabs.items():
        res, _ = conformal_loco(t)
        res.to_csv(OUT / f"conformal_{k}.csv", index=False, float_format="%.4f")
        conf[k] = dict(mean_coverage=float(res["coverage"].mean()), min_coverage=float(res["coverage"].min()),
                       mean_width=float(res["mean_width"].mean()), mean_rho_city=float(res["rho_model_human"].mean()))
    log(f"  conformal: {conf}")

    # ── kNN entre ciudades con el embedding v2 (si la v2 es una fusión de features) ──────────
    sp = load_split("image_disjoint")
    vt = votes(pairs, sp["train"]).groupby("id")["win"].agg(["size", "mean"]).query("size >= 3")
    mrows = row.reindex(vt.index).to_numpy()
    te = pairs.iloc[sp["test"]]
    q_ids = pd.unique(pd.concat([te["left"], te["right"]]))
    qrows = row.reindex(q_ids).to_numpy()
    city = man["city"].to_numpy()
    knn = {}
    for k, f in (("v1", "dinov2"), ("v2", cfg["features"])):
        X = sets[f] / np.linalg.norm(sets[f], axis=1, keepdims=True)
        p, _ = knn_scores(X[qrows], city[qrows], X[mrows], city[mrows], vt["mean"].to_numpy(), mode="cross")
        knn[k] = pair_eval(te, pd.Series(p, index=q_ids))
    kd = paired_delta(knn["v2"], knn["v1"], rng)
    knn_res = dict(acc_v1=float(knn["v1"].mean()), acc_v2=float(knn["v2"].mean()), **{k: float(v) for k, v in kd.items()})
    log(f"  kNN entre ciudades: {knn_res}")

    (OUT / "summary.json").write_text(json.dumps({"config_v2": cfg, "neighborhood_within_city": hood,
                                                  "conformal": conf, "knn_cross_city": knn_res,
                                                  "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
