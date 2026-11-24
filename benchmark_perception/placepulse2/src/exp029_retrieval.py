"""
EXP-029 — Explicación por ejemplos: ¿a qué escenas de OTRAS ciudades se parece esta escena? (R5 del asesor)

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa EXP-028..031, 2026-10-09).

Memoria = imágenes de train (image_disjoint) con >= 3 votos de train; su puntaje es el juicio humano (tasa de
victoria). Para cada imagen nueva se recuperan sus k = 50 vecinas más parecidas en DINOv2 EXCLUYENDO su propia
ciudad, y su puntaje es la media ponderada (softmax de similitud, τ = 0.05) del juicio humano de esas vecinas.
Si funciona, la predicción se explica sola: "se parece a estas escenas de otras ciudades, que la gente juzgó así".
Inspiración: Deep k-NN (Papernot & McDaniel 2018), "This looks like that" (Chen et al. 2019).

Uso:  python exp029_retrieval.py        Salidas: outputs/exp029/ (incl. figura de ejemplos)
"""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from exp026_additive_context import load_images
from probe_utils import cell_ids, load_pairs, load_split, train_scores, votes
from utils import OUTPUTS, ZIP_PATH, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp029"
K, TAU = 50, 0.05
MIN_VOTES = 3
N_BOOT = 2000
CHUNK = 1500


def knn_scores(Zq, city_q, Zm, city_m, score_m, k=K, tau=TAU, mode="cross", rng=None):
    """Puntaje por kNN. mode: cross (excluye su ciudad), any, random (k aleatorias de otras ciudades)."""
    out = np.empty(len(Zq), dtype=np.float64)
    top_idx = np.empty((len(Zq), 5), dtype=np.int64)
    cm = pd.Categorical(city_m)
    cq = pd.Categorical(city_q, categories=cm.categories).codes
    cmc = cm.codes
    for a in range(0, len(Zq), CHUNK):
        b = min(a + CHUNK, len(Zq))
        if mode == "random":
            for i in range(a, b):
                pool = np.flatnonzero(cmc != cq[i])
                j = rng.choice(pool, k, replace=False)
                out[i] = score_m[j].mean()
            continue
        S = Zq[a:b] @ Zm.T
        if mode == "cross":
            S[cq[a:b, None] == cmc[None, :]] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        w = np.exp((sims - sims.max(axis=1, keepdims=True)) / tau)
        w /= w.sum(axis=1, keepdims=True)
        out[a:b] = (w * score_m[idx]).sum(axis=1)
        order = np.argsort(-sims, axis=1)[:, :5]
        top_idx[a:b] = np.take_along_axis(idx, order, axis=1)
    return out, top_idx


def pair_eval(df: pd.DataFrame, pred: pd.Series) -> np.ndarray:
    d = pred.reindex(df["left"]).to_numpy() - pred.reindex(df["right"]).to_numpy()
    return ((d > 0).astype(int) == df["label"].to_numpy())


def paired_delta(c1, c0, rng):
    n = len(c1)
    bs = np.array([c1[ix].mean() - c0[ix].mean() for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(delta=c1.mean() - c0.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                mcnemar_p=binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


def figure(man, rows_q, top, mem_ids, mem_score, q_pred, q_human, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    path_of = dict(zip(man["location_id"], man["image_path"]))
    city_of = dict(zip(man["location_id"], man["city"]))
    fig, axes = plt.subplots(len(rows_q), 6, figsize=(13, 2.4 * len(rows_q)), dpi=130)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for r, (lid, tops, p, h) in enumerate(zip(rows_q, top, q_pred, q_human)):
            cells = [(lid, f"CONSULTA · {city_of[lid]}\npred {p:.2f} · humano {h:.2f}")]
            cells += [(mem_ids[j], f"{city_of[mem_ids[j]]}\njuicio humano {mem_score[j]:.2f}") for j in tops]
            for c, (x, title) in enumerate(cells):
                ax = axes[r, c]
                with zf.open(path_of[x]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(title, fontsize=7.5, color="#0b0b0b" if c else "#1F4E79")
                ax.axis("off")
    fig.suptitle("EXP-029 · Explicación por ejemplos: escenas más parecidas de OTRAS ciudades "
                 "(puntaje = tasa de victoria en votos 'más seguro')", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].to_numpy()
    pairs = load_pairs()
    sp = load_split("image_disjoint")

    # memoria: train con >= 3 votos de train
    vt = votes(pairs, sp["train"]).groupby("id")["win"].agg(["size", "mean"])
    vt = vt[vt["size"] >= MIN_VOTES]
    row = pd.Series(np.arange(len(ids)), index=ids)
    mrows = row.reindex(vt.index).to_numpy()
    Zm, city_m, score_m, mem_ids = Z[mrows], man["city"].to_numpy()[mrows], vt["mean"].to_numpy(), vt.index.to_numpy()
    log(f"memoria: {len(mrows):,} imágenes de train con >= {MIN_VOTES} votos")

    # consultas: imágenes de val y test
    q_ids = pd.unique(pd.concat([votes(pairs, sp["val"])["id"], votes(pairs, sp["test"])["id"]]))
    qrows = row.reindex(q_ids).to_numpy()
    Zq, city_q = Z[qrows], man["city"].to_numpy()[qrows]
    log(f"consultas: {len(qrows):,} imágenes (val + test)")

    s_probe, info = train_scores(Z, list(ids), pairs, sp, "test")
    probe = pd.Series(s_probe, index=ids)
    preds = {"probe": probe}
    tops = None
    for mode in ("cross", "any", "random"):
        p, t = knn_scores(Zq, city_q, Zm, city_m, score_m, mode=mode, rng=rng)
        preds[f"knn_{mode}"] = pd.Series(p, index=q_ids)
        if mode == "cross":
            tops = t
        log(f"  kNN {mode}: listo ({time.perf_counter() - t0:.0f}s)")
    # sensibilidad (declarada, no criterio): k = 10 y 200
    for k in (10, 200):
        p, _ = knn_scores(Zq, city_q, Zm, city_m, score_m, k=k, mode="cross")
        preds[f"knn_cross_k{k}"] = pd.Series(p, index=q_ids)

    # ensamble: z-scores, peso elegido en val
    val = pairs.iloc[sp["val"]]
    zp = (probe - probe.reindex(q_ids).mean()) / probe.reindex(q_ids).std()
    zk = (preds["knn_cross"] - preds["knn_cross"].mean()) / preds["knn_cross"].std()
    best = max(np.linspace(0, 1, 21), key=lambda a: pair_eval(val, a * zp.reindex(q_ids) + (1 - a) * zk).mean())
    preds["ensemble"] = best * zp.reindex(q_ids) + (1 - best) * zk
    log(f"  ensamble: peso probe = {best:.2f} (elegido en val)")

    test = pairs.iloc[sp["test"]]
    correct = {k: pair_eval(test, v) for k, v in preds.items()}
    acc = pd.DataFrame([dict(variant=k, n=len(c), accuracy=c.mean()) for k, c in correct.items()])
    comps = [("knn_cross", "probe"), ("knn_cross", "knn_random"), ("knn_any", "knn_cross"), ("ensemble", "probe")]
    delta = pd.DataFrame([dict(comparison=f"{a} - {b}", **paired_delta(correct[a], correct[b], rng)) for a, b in comps])
    for _, r in acc.iterrows():
        log(f"  {r['variant']:<16} acc={r['accuracy']:.4f}")
    for _, r in delta.iterrows():
        log(f"  {r['comparison']:<22} Δ={r['delta']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['mcnemar_p']:.2g}")
    acc.to_csv(OUT / "retrieval_accuracy.csv", index=False, float_format="%.5f")
    delta.to_csv(OUT / "retrieval_paired.csv", index=False, float_format="%.5f")

    # ── nivel barrio (exploratorio): celdas de 1 km ────────────────────────────────────────────
    cell = pd.Series(cell_ids(man, 1000).to_numpy(), index=ids)
    city_of = pd.Series(man["city"].to_numpy(), index=ids)
    tr_votes = votes(pairs, sp["train"]).assign(cell=lambda d: d["id"].map(cell))
    te_imgs = set(votes(pairs, sp["test"])["id"])
    te_votes = votes(pairs, sp["test"], keep=te_imgs).assign(cell=lambda d: d["id"].map(cell))
    mem_c = tr_votes.groupby("cell")["win"].agg(["size", "mean"]).query("size >= 20")
    tst_c = te_votes.groupby("cell")["win"].agg(["size", "mean"]).query("size >= 20")
    tr_imgs = set(tr_votes["id"])
    emb = lambda keep, cells: np.vstack([Z[row.reindex([i for i in ids_c if i in keep]).to_numpy()].mean(0)
                                        for ids_c in (cell[cell == c].index for c in cells)])
    Em, Et = emb(tr_imgs, mem_c.index), emb(te_imgs, tst_c.index)
    Em /= np.linalg.norm(Em, axis=1, keepdims=True)
    Et /= np.linalg.norm(Et, axis=1, keepdims=True)
    cc_m = mem_c.index.str.split("|").str[0].to_numpy()
    cc_t = tst_c.index.str.split("|").str[0].to_numpy()
    pc, _ = knn_scores(Et, cc_t, Em, cc_m, mem_c["mean"].to_numpy(), k=20, mode="cross")
    pr, _ = knn_scores(Et, cc_t, Em, cc_m, mem_c["mean"].to_numpy(), k=20, mode="random", rng=rng)
    probe_cell = probe.reindex([i for i in ids if i in te_imgs]).groupby(cell).mean().reindex(tst_c.index)
    hood = dict(n_test_cells=len(tst_c), n_memory_cells=len(mem_c),
                rho_retrieval_cells=float(spearmanr(pc, tst_c["mean"])[0]),
                rho_random_cells=float(spearmanr(pr, tst_c["mean"])[0]),
                rho_probe_cells=float(spearmanr(probe_cell, tst_c["mean"])[0]))
    log(f"  barrio 1 km: {hood}")

    # figura de ejemplos: 3 consultas de test (alta, media, baja según kNN)
    te_list = [i for i in q_ids if i in te_imgs]
    pos = pd.Series(np.arange(len(q_ids)), index=q_ids)
    pk = preds["knn_cross"].reindex(te_list).sort_values()
    hum = votes(pairs, sp["test"], keep=te_imgs).groupby("id")["win"].agg(["size", "mean"])
    pk = pk[pk.index.isin(hum[hum["size"] >= 4].index)]
    pick = [pk.index[-1], pk.index[len(pk) // 2], pk.index[0]]
    figure(man, pick, [tops[pos[x]] for x in pick], mem_ids, score_m,
           [preds["knn_cross"][x] for x in pick], [hum.loc[x, "mean"] for x in pick],
           OUT / "exp029_examples.png")

    (OUT / "summary.json").write_text(json.dumps({
        "seed": SEED, "k": K, "tau": TAU, "min_votes": MIN_VOTES, "n_memory": int(len(mrows)),
        "n_queries": int(len(qrows)), "ensemble_probe_weight": float(best),
        "probe_test_acc": info["metrics"]["pairwise_accuracy"], "neighborhood_1km": hood,
        "seconds": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"listo en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
