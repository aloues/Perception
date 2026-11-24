"""
EXP-026 (Parte 1) — contexto ADITIVO vs SUSTITUTIVO sobre Place Pulse 2.0, 56 ciudades.

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-026, 2026-10-09).

POR QUE
-------
EXP-021 construyó ego-1 = [mean(z_i ∪ vecinas), std]: el embedding propio z_i queda diluido con
peso 1/(n+1). La etiqueta de Place Pulse (voto pareado) es de la IMAGEN, así que ese diseño mide
si el promedio del barrio puede REEMPLAZAR a la imagen, no si el barrio AÑADE información. Además,
el 97.8 % de los pares `safe` son entre ciudades distintas: un contexto puede ayudar solo por
codificar la ciudad. Este script separa las tres cosas:

    A     ego0          [z_i, 0]
    S     sust250       [mean(z_i ∪ N250), std]       (reproduce EXP-021)
    SR    sust_rand250  [mean(z_i ∪ R250), std]       R = aleatorias de la misma ciudad, |R| = |N|
    L     add250        [z_i, mean(N250)]
    LR    add_rand250   [z_i, mean(R250)]
    LC    add_city      [z_i, mean(ciudad \\ i)]
    L400  add400        [z_i, mean(N400)]
    LR400 add_rand400   [z_i, mean(R400)]
    CL    city_local    [z_i, mean(ciudad \\ i), mean(N250)]   (2304-d)
    CR    city_rand     [z_i, mean(ciudad \\ i), mean(R250)]   (2304-d)
    C0    city_zero     [z_i, mean(ciudad \\ i), 0]            (2304-d)

Sin vecinas -> bloque de contexto local = 0 (y R vacío para mantener |R| = |N|). Tope 20 vecinas
más cercanas (paridad con EXP-021). Vecinas por distancia euclídea con la MISMA proyección local de
EXP-021. Las features se escriben en el formato de train_probe.py, que se usa SIN cambios.

Uso:
    python exp026_additive_context.py               # construye features, entrena y analiza
    python exp026_additive_context.py --analyze-only
Salidas: outputs/exp026/  (features/*.npy no versionadas; runs en outputs/runs/exp026__*)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import binomtest

from utils import FEATURES, MANIFESTS, OUTPUTS, RUNS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp026"
FEAT = OUT / "features"
COORD_RE = re.compile(r"images/(-?\d+\.\d+)_(-?\d+\.\d+)_")
NEIGH_MAX = 20
N_BOOT = 2000

VARIANTS_ID = ["ego0", "sust250", "sust_rand250", "add250", "add_rand250", "add_city",
               "add400", "add_rand400", "city_local", "city_rand", "city_zero"]
VARIANTS_CH = ["ego0", "add250", "add_rand250", "add_city"]


def load_images(encoder: str):
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).dropna(subset=["image_path"])
    ll = man["image_path"].str.extract(COORD_RE).astype(float)
    man["lat"], man["lon"] = ll[0], ll[1]
    man = man.dropna(subset=["lat", "lon"])
    d = FEATURES / encoder
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    shards = sorted(d.glob("shard_*.npy"))
    Z = np.concatenate([np.load(s) for s in shards], axis=0) if shards else np.load(d / "features.npy")
    row_of = {lid: i for i, lid in enumerate(index["location_ids"][: Z.shape[0]])}
    man = man[man["location_id"].isin(row_of)].reset_index(drop=True)
    Z = Z[man["location_id"].map(row_of).to_numpy()].astype(np.float32)   # Z alineado con man
    return man, Z


def neighbor_sets(man: pd.DataFrame, radius_m: float, rng: np.random.Generator):
    """Listas (índices globales) de vecinas euclídeas <= radio y de aleatorias count-matched."""
    near, rand = [None] * len(man), [None] * len(man)
    for city, g in man.groupby("city"):
        idx = g.index.to_numpy()
        lat0 = g["lat"].mean()
        # misma proyección equirectangular local que EXP-021
        xy = np.c_[g["lon"].to_numpy() * 111320 * np.cos(np.radians(lat0)), g["lat"].to_numpy() * 110540]
        tree = cKDTree(xy)
        for j in range(len(g)):
            nb = [k for k in tree.query_ball_point(xy[j], radius_m) if k != j]
            if len(nb) > NEIGH_MAX:
                dd = np.linalg.norm(xy[nb] - xy[j], axis=1)
                nb = [nb[k] for k in np.argsort(dd, kind="stable")[:NEIGH_MAX]]
            near[idx[j]] = idx[nb]
            if nb:
                pool = np.setdiff1d(np.arange(len(g)), np.r_[j, nb])
                rand[idx[j]] = idx[rng.choice(pool, size=min(len(nb), len(pool)), replace=False)]
            else:
                rand[idx[j]] = idx[[]]
    return near, rand


def mean_matrix(lists, n: int, include_self: bool) -> sp.csr_matrix:
    """Matriz fila-normalizada W tal que (W @ Z)[i] = media de Z sobre la lista (± la propia)."""
    rows, cols, vals = [], [], []
    for i, L in enumerate(lists):
        members = np.r_[i, L] if include_self else np.asarray(L)
        if len(members) == 0:
            continue
        rows.extend([i] * len(members))
        cols.extend(members.tolist())
        vals.extend([1.0 / len(members)] * len(members))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


def city_loo_mean(man: pd.DataFrame, Z: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Z)
    for _, g in man.groupby("city"):
        idx = g.index.to_numpy()
        tot = Z[idx].astype(np.float64).sum(axis=0)
        out[idx] = ((tot[None, :] - Z[idx]) / (len(idx) - 1)).astype(np.float32)
    return out


def save_variant(name: str, X: np.ndarray, ids: list[str], meta: dict) -> None:
    d = FEAT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "features.npy", X.astype(np.float32))
    (d / "index.json").write_text(json.dumps({"variant": name, "embed_dim": int(X.shape[1]),
                                              "n_images": len(ids), "location_ids": ids, **meta},
                                             indent=2), encoding="utf-8")
    log(f"  -> {name} {X.shape}")


def build(encoder: str) -> dict:
    rng = np.random.default_rng(SEED)
    man, Z = load_images(encoder)
    n, dim = Z.shape
    ids = man["location_id"].tolist()
    log(f"{n:,} imágenes, {man['city'].nunique()} ciudades, Z {Z.shape}")
    N250, R250 = neighbor_sets(man, 250.0, rng)
    N400, R400 = neighbor_sets(man, 400.0, rng)
    cnt = {k: np.array([len(x) for x in v]) for k, v in
           dict(N250=N250, R250=R250, N400=N400, R400=R400).items()}
    assert (cnt["N250"] == cnt["R250"]).mean() > 0.999 and (cnt["N400"] == cnt["R400"]).mean() > 0.999
    city = city_loo_mean(man, Z)
    zeros = np.zeros_like(Z)
    meta = {"encoder_base": encoder, "neigh_max": NEIGH_MAX, "seed": SEED}

    def ctx(lists):
        return (mean_matrix(lists, n, include_self=False) @ Z).astype(np.float32)

    def subst(lists):
        W = mean_matrix(lists, n, include_self=True)
        m = W @ Z
        var = np.maximum(W @ (Z.astype(np.float64) ** 2) - m ** 2, 0.0)
        return np.hstack([m, np.sqrt(var)]).astype(np.float32)

    save_variant("ego0", np.hstack([Z, zeros]), ids, meta)
    save_variant("sust250", subst(N250), ids, {**meta, "radius_m": 250})
    save_variant("sust_rand250", subst(R250), ids, {**meta, "radius_m": 250, "random_same_city": True})
    c250, r250 = ctx(N250), ctx(R250)
    save_variant("add250", np.hstack([Z, c250]), ids, {**meta, "radius_m": 250})
    save_variant("add_rand250", np.hstack([Z, r250]), ids, {**meta, "radius_m": 250, "random_same_city": True})
    save_variant("add_city", np.hstack([Z, city]), ids, {**meta, "city_loo_mean": True})
    save_variant("city_local", np.hstack([Z, city, c250]), ids, {**meta, "radius_m": 250})
    save_variant("city_rand", np.hstack([Z, city, r250]), ids, {**meta, "radius_m": 250, "random_same_city": True})
    save_variant("city_zero", np.hstack([Z, city, zeros]), ids, meta)
    del c250, r250
    save_variant("add400", np.hstack([Z, ctx(N400)]), ids, {**meta, "radius_m": 400})
    save_variant("add_rand400", np.hstack([Z, ctx(R400)]), ids, {**meta, "radius_m": 400, "random_same_city": True})

    cov = (man.assign(n250=cnt["N250"], n400=cnt["N400"]).groupby("city")
           .agg(n_images=("location_id", "size"), mean_n250=("n250", "mean"),
                pct_zero_250=("n250", lambda s: 100 * (s == 0).mean()),
                mean_n400=("n400", "mean"), pct_zero_400=("n400", lambda s: 100 * (s == 0).mean()))
           .round(2).reset_index())
    cov.to_csv(OUT / "coverage_by_city.csv", index=False)
    summ = {"n_images": n, "n_cities": int(man["city"].nunique()),
            "mean_neighbors_250": float(cnt["N250"].mean()), "pct_zero_250": float(100 * (cnt["N250"] == 0).mean()),
            "mean_neighbors_400": float(cnt["N400"].mean()), "pct_zero_400": float(100 * (cnt["N400"] == 0).mean())}
    log(f"cobertura: {summ}")
    return summ


def train(variants: list[str], protocol: str) -> None:
    here = Path(__file__).resolve().parent
    for v in variants:
        run_id = f"exp026__{v}__{protocol}__seed{SEED}"
        if (RUNS / run_id / "metrics.json").exists():
            log(f"  {run_id}: ya existe, no se re-entrena")
            continue
        t = time.perf_counter()
        subprocess.run([sys.executable, str(here / "train_probe.py"), "--encoder", v,
                        "--protocol", protocol, "--seed", str(SEED), "--run-id", run_id,
                        "--features-root", str(FEAT)], check=True,
                       stdout=subprocess.DEVNULL)
        log(f"  {run_id}: {time.perf_counter() - t:.0f}s")


def _logloss(logit, y):
    p = np.clip(1 / (1 + np.exp(-np.clip(logit, -50, 50))), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(base: pd.DataFrame, other: pd.DataFrame, rng) -> dict:
    assert (base["left"].to_numpy() == other["left"].to_numpy()).all()
    assert (base["right"].to_numpy() == other["right"].to_numpy()).all()
    y = base["label"].to_numpy()
    c0 = (base["pred"] == base["label"]).to_numpy()
    c1 = (other["pred"] == other["label"]).to_numpy()
    l0, l1 = _logloss(base["logit"].to_numpy(), y), _logloss(other["logit"].to_numpy(), y)
    n = len(y)
    da, dl = np.empty(N_BOOT), np.empty(N_BOOT)
    for t in range(N_BOOT):
        ix = rng.integers(0, n, n)
        da[t] = c1[ix].mean() - c0[ix].mean()
        dl[t] = l1[ix].mean() - l0[ix].mean()
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(n=n, acc_base=c0.mean(), acc=c1.mean(), delta_acc=c1.mean() - c0.mean(),
                delta_acc_ci_low=np.percentile(da, 2.5), delta_acc_ci_high=np.percentile(da, 97.5),
                mcnemar_p=binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0,
                logloss_base=l0.mean(), logloss=l1.mean(), delta_logloss=l1.mean() - l0.mean(),
                delta_logloss_ci_low=np.percentile(dl, 2.5), delta_logloss_ci_high=np.percentile(dl, 97.5))


# comparaciones fijadas en el diseño (otra, base)
COMPARISONS_ID = [("sust250", "ego0"), ("sust_rand250", "ego0"), ("sust_rand250", "sust250"),
                  ("add250", "ego0"), ("add_rand250", "ego0"), ("add_city", "ego0"),
                  ("add250", "add_rand250"), ("add400", "ego0"), ("add_rand400", "ego0"),
                  ("add400", "add_rand400"), ("city_zero", "ego0"), ("city_local", "city_rand"),
                  ("city_local", "city_zero"), ("city_rand", "city_zero"), ("add_city", "add250")]
COMPARISONS_CH = [("add250", "ego0"), ("add_rand250", "ego0"), ("add_city", "ego0"),
                  ("add250", "add_rand250")]


def analyze() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for protocol, comps in (("image_disjoint", COMPARISONS_ID), ("city_holdout", COMPARISONS_CH)):
        for other, base in comps:
            P = {v: pd.read_csv(RUNS / f"exp026__{v}__{protocol}__seed{SEED}" / "predictions.csv")
                 for v in (other, base)}
            for split in sorted(P[base]["split"].unique()):
                r = paired(P[base][P[base]["split"] == split].reset_index(drop=True),
                           P[other][P[other]["split"] == split].reset_index(drop=True), rng)
                rows.append(dict(protocol=protocol, split=split, comparison=f"{other} - {base}", **r))
                log(f"  {protocol:<14} {split:<11} {other:>13} - {base:<12} Δacc={r['delta_acc']:+.4f} "
                    f"[{r['delta_acc_ci_low']:+.4f},{r['delta_acc_ci_high']:+.4f}] p={r['mcnemar_p']:.2g}  "
                    f"Δll={r['delta_logloss']:+.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "paired_comparisons.csv", index=False, float_format="%.5f")
    acc = []
    for protocol, vs in (("image_disjoint", VARIANTS_ID), ("city_holdout", VARIANTS_CH)):
        for v in vs:
            m = json.loads((RUNS / f"exp026__{v}__{protocol}__seed{SEED}" / "metrics.json").read_text())
            for split, t in m["test"].items():
                acc.append(dict(protocol=protocol, variant=v, split=split, n=t["n"],
                                accuracy=t["pairwise_accuracy"], ci95_low=t["ci95_low"],
                                ci95_high=t["ci95_high"], log_loss=t["log_loss"]))
    pd.DataFrame(acc).to_csv(OUT / "accuracy_by_variant.csv", index=False, float_format="%.5f")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_vit_b14")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summ = {}
    if not args.analyze_only:
        if not (FEAT / "add_rand400" / "features.npy").exists():
            summ = build(args.encoder)
            (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        train(VARIANTS_ID, "image_disjoint")
        train(VARIANTS_CH, "city_holdout")
    analyze()
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
