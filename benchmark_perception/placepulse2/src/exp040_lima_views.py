"""
EXP-040 — Lima: redundancia de las 12 vistas por spot y representaciones V1/V2/V3/V4, SIN etiquetas.

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-040, 2026-10-23).

POR QUE
-------
El asesor (23/10/2026) observa que las 12 vistas de un spot son redundantes y propone comparar la
media de las 12 (idea del profesor) con 4 vistas aleatorias y con un Visual Bag of Words. En Lima no
hay etiquetas de percepción, así que aquí solo se mide lo que se puede medir sin ellas:
  (1) cuánta redundancia hay (similitud por separación angular, rango efectivo);
  (2) cuánta información de V1 conserva V2 (fidelidad y estructura spot–spot);
  (3) cuánta "firma del lugar" aporta cada vista adicional (re-identificación split-half);
  (4) si el vocabulario BoVW es estable y transfiere entre distritos;
  (5) cuánto cambia una predicción de un probe de Place Pulse al usar 4 vistas (proxy, NO verdad).

Uso:
    python exp040_lima_views.py            # todo
    python exp040_lima_views.py --skip-proxy
Salidas: outputs/exp040/
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import figstyle
from lima_views import attach_coordinates, load_views, spots_table
from utils import OUTPUTS, environment_info, log, set_seed
from visual_repr import (BoVW, angular_separation, effective_rank, fuse, geometric_overlap, histograms,
                         js_divergence, l2n, select_random, select_stratified)

SEED = 42
OUT = OUTPUTS / "exp040"
H12 = np.arange(0, 360, 30)
HALF_A, HALF_B = H12[H12 < 180], H12[H12 >= 180]
N_SEEDS = 10
KS = (100, 120, 500)


def build_tensor() -> tuple[pd.DataFrame, np.ndarray]:
    """Spots 12/12 → (spots, E[spot, heading_idx, 768]) con las vistas ordenadas por heading."""
    V, E = load_views()
    V = attach_coordinates(V)
    S = spots_table(V)
    S = S[S["complete_12"]].reset_index(drop=True)
    V = V.merge(S[["district", "spot_id"]].reset_index().rename(columns={"index": "sidx"}),
                on=["district", "spot_id"])
    T = np.zeros((len(S), 12, E.shape[1]), dtype=np.float32)
    T[V["sidx"].to_numpy(), (V["heading"].to_numpy() // 30)] = np.asarray(E)[V["row"].to_numpy()]
    assert np.all(np.linalg.norm(T, axis=2) > 0.99)
    return S, T


# ─── (1) redundancia ──────────────────────────────────────────────────────────

def redundancy(S: pd.DataFrame, T: np.ndarray, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    C = np.einsum("nid,njd->nij", T, T)
    iu = np.triu_indices(12, 1)
    sep = angular_separation(H12[iu[0]], H12[iu[1]])
    rows = []
    for d, g in S.groupby("district"):
        idx = g.index.to_numpy()
        cs = C[idx][:, iu[0], iu[1]]
        for s in np.unique(sep):
            v = cs[:, sep == s].ravel()
            rows.append(dict(district=d, kind="same_spot", separation_deg=int(s), mean_cos=v.mean(),
                             sd_cos=v.std(), pct_cos_gt_095=100 * (v > 0.95).mean(), n_pairs=len(v),
                             geometric_overlap_fov90=geometric_overlap(90, s)))
        # nulo: vistas de spots DISTINTOS del mismo distrito (aleatorias) y mismo heading
        a, b = rng.choice(idx, 20000), rng.choice(idx, 20000)
        keep = a != b
        ha, hb = rng.integers(0, 12, 20000), rng.integers(0, 12, 20000)
        v = np.einsum("nd,nd->n", T[a[keep], ha[keep]], T[b[keep], hb[keep]])
        rows.append(dict(district=d, kind="null_other_spot_any_heading", separation_deg=-1, mean_cos=v.mean(),
                         sd_cos=v.std(), pct_cos_gt_095=100 * (v > 0.95).mean(), n_pairs=len(v)))
        v = np.einsum("nd,nd->n", T[a[keep], ha[keep]], T[b[keep], ha[keep]])
        rows.append(dict(district=d, kind="null_other_spot_same_heading", separation_deg=0, mean_cos=v.mean(),
                         sd_cos=v.std(), pct_cos_gt_095=100 * (v > 0.95).mean(), n_pairs=len(v)))
    er = pd.DataFrame({"district": S["district"], "spot_id": S["spot_id"],
                       "effective_rank": [effective_rank(T[i]) for i in range(len(S))]})
    return pd.DataFrame(rows), er


# ─── (2) fidelidad de V2 ──────────────────────────────────────────────────────

def view_subsets(n_spots: int, n: int, mode: str, seed: int) -> np.ndarray:
    """Máscara [spot, 12] con las vistas elegidas (una sorteo independiente por spot)."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_spots, 12), bool)
    for i in range(n_spots):
        h = select_random(H12, n, rng) if mode == "random" else select_stratified(H12, n, rng)
        M[i, h // 30] = True
    return M


def masked_mean(T: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (T * M[:, :, None]).sum(axis=1) / M.sum(axis=1, keepdims=True)


def fidelity(S: pd.DataFrame, T: np.ndarray, rng) -> pd.DataFrame:
    V1 = l2n(T.mean(axis=1))
    rows = []
    for d, g in S.groupby("district"):
        idx = g.index.to_numpy()
        a, b = rng.choice(len(idx), 50000), rng.choice(len(idx), 50000)
        keep = a != b
        a, b = idx[a[keep]], idx[b[keep]]
        s1 = np.einsum("nd,nd->n", V1[a], V1[b])
        for mode in ("random", "stratified"):
            for seed in range(N_SEEDS):
                V2 = l2n(masked_mean(T, view_subsets(len(S), 4, mode, seed)))
                cos = np.einsum("nd,nd->n", V2[idx], V1[idx])
                rho = spearmanr(s1, np.einsum("nd,nd->n", V2[a], V2[b])).statistic
                rows.append(dict(district=d, variant=f"V2_4{'R' if mode == 'random' else 'S'}", seed=seed,
                                 cos_to_V1_mean=cos.mean(), cos_to_V1_p05=np.percentile(cos, 5),
                                 spearman_structure=rho))
    return pd.DataFrame(rows)


# ─── (3) re-identificación split-half ─────────────────────────────────────────

def rank_of_true(Q: np.ndarray, G: np.ndarray, similarity: str = "cos") -> np.ndarray:
    """Rango (1 = mejor) del spot verdadero para cada consulta i (galería i es la correcta)."""
    if similarity == "cos":
        Sim = l2n(Q) @ l2n(G).T
    else:                                              # -JS entre histogramas
        Sim = -np.stack([js_divergence(np.repeat(Q[i:i + 1], len(G), 0), G) for i in range(len(Q))])
    true = np.diag(Sim)
    return 1 + (Sim > true[:, None]).sum(axis=1)


def reid(S: pd.DataFrame, T: np.ndarray) -> pd.DataFrame:
    rows = []
    for d, g in S.groupby("district"):
        idx = g.index.to_numpy()
        Tb = T[idx][:, HALF_B // 30].mean(axis=1)
        for n in (1, 2, 3, 4, 6):
            for mode in ("random", "stratified") if n == 2 else ("random",):
                for seed in range(N_SEEDS if n < 6 else 1):
                    rng = np.random.default_rng(seed)
                    M = np.zeros((len(idx), 12), bool)
                    for i in range(len(idx)):
                        h = select_random(HALF_A, n, rng) if mode == "random" else \
                            np.array([0, 90]) + 30 * int(rng.integers(0, 3))
                        M[i, h // 30] = True
                    r = rank_of_true(masked_mean(T[idx], M), Tb)
                    rows.append(dict(district=d, n_views_query=n, mode=mode, seed=seed, n_spots=len(idx),
                                     top1=(r == 1).mean(), top5=(r <= 5).mean(), mrr=(1 / r).mean()))
    return pd.DataFrame(rows)


# ─── (4) BoVW ─────────────────────────────────────────────────────────────────

def bovw(S: pd.DataFrame, T: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows, vocab = [], {}
    districts = sorted(S["district"].unique())
    for train_d in districts:
        test_d = [d for d in districts if d != train_d][0]
        tr = S.index[S["district"] == train_d].to_numpy()
        te = S.index[S["district"] == test_d].to_numpy()
        Xtr = T[tr].reshape(-1, T.shape[2])
        Xte = T[te].reshape(-1, T.shape[2])
        for K in KS:
            words = {}
            for seed in (0, 1):
                t0 = time.perf_counter()
                bv = BoVW(K, seed=seed).fit(Xtr)
                words[seed] = bv.assign(Xte)
                occ_tr = bv.train_occupancy()
                vocab[(train_d, K, seed)] = bv
                log(f"  BoVW train={train_d} K={K} seed={seed}: {time.perf_counter() - t0:.0f}s")
                p = np.bincount(words[seed], minlength=K) / len(words[seed])
                ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(K)
                w = words[seed].reshape(len(te), 12)
                spot_g = np.repeat(np.arange(len(te)), 6)
                HA = histograms(w[:, HALF_A // 30].ravel(), spot_g, K, len(te))
                HB = histograms(w[:, HALF_B // 30].ravel(), spot_g, K, len(te))
                r_bovw = rank_of_true(HA, HB, "js")
                EA = T[te][:, HALF_A // 30].mean(axis=1)
                EB = T[te][:, HALF_B // 30].mean(axis=1)
                r_fus = rank_of_true(fuse(EA, HA), fuse(EB, HB))
                Hspot = histograms(w.ravel(), np.repeat(np.arange(len(te)), 12), K, len(te))
                rows.append(dict(vocab_district=train_d, eval_district=test_d, K=K, seed=seed,
                                 n_train_descriptors=len(Xtr), train_empty_clusters=int((occ_tr == 0).sum()),
                                 test_bins_used_pct=100 * (p > 0).mean(), test_norm_entropy=ent,
                                 words_per_spot_distinct_mean=(Hspot > 0).sum(axis=1).mean(),
                                 reid_top1_bovw_js=(r_bovw == 1).mean(), reid_mrr_bovw_js=(1 / r_bovw).mean(),
                                 reid_top1_fusion=(r_fus == 1).mean(), reid_mrr_fusion=(1 / r_fus).mean()))
            ari = adjusted_rand_score(words[0], words[1])
            for r in rows[-2:]:
                r["ari_seed0_vs_seed1_on_eval"] = ari
    return pd.DataFrame(rows), vocab


# ─── (5) proxy de sensibilidad ────────────────────────────────────────────────

def pp_proxy(S: pd.DataFrame, T: np.ndarray) -> pd.DataFrame:
    """Probe SimCLR de Place Pulse (mismo checkpoint epoch_50 y preprocesado) puntúa vistas de Lima.

    NO es percepción medida en Lima (prohibido por M01/PROJECT_STATE §2): solo mide cuánto cambia la
    predicción de un spot al usar 4 vistas en vez de 12.
    """
    from probe_utils import load_pairs, load_split, train_scores
    from train_probe import load_features
    X, row_of = load_features("ours_simclr_vit_b16")
    ids = sorted(row_of, key=row_of.get)
    _, info = train_scores(X, ids, load_pairs(), load_split("image_disjoint"))
    w = info["weights"]
    sv = T @ w                                            # puntaje por vista (lineal)
    v1 = sv.mean(axis=1)
    rows = []
    for d, g in S.groupby("district"):
        idx = g.index.to_numpy()
        for mode in ("random", "stratified"):
            draws = np.stack([(sv * view_subsets(len(S), 4, mode, s)).sum(1) / 4 for s in range(N_SEEDS)])[:, idx]
            rows.append(dict(district=d, variant=f"V2_4{'R' if mode == 'random' else 'S'}",
                             probe_test_acc_pp=info["metrics"]["pairwise_accuracy"],
                             pearson_V2_vs_V1_mean=np.mean([np.corrcoef(x, v1[idx])[0, 1] for x in draws]),
                             sd_between_spots_V1=v1[idx].std(), sd_across_seeds_mean=draws.std(axis=0).mean(),
                             ratio_seed_sd_to_between_sd=draws.std(axis=0).mean() / v1[idx].std(),
                             sd_across_views_within_spot=sv[idx].std(axis=1).mean()))
    return pd.DataFrame(rows)


# ─── figuras ──────────────────────────────────────────────────────────────────

def figures(red: pd.DataFrame, rid: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    figstyle.setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    for c, (d, g) in zip(figstyle.SERIES, red.groupby("district")):
        s = g[g["kind"] == "same_spot"]
        ax.plot(s["separation_deg"], s["mean_cos"], "-o", color=c, label=d.replace("_", " ").title())
        nul = g[g["kind"] == "null_other_spot_any_heading"]["mean_cos"].iloc[0]
        ax.axhline(nul, color=c, ls=":", lw=1.5)
    ax.text(182, ax.get_ylim()[0] + 0.01, "punteado: spots distintos", fontsize=7, color=figstyle.INK2, ha="right")
    ax.set_xlabel("separación angular entre vistas (°)")
    ax.set_ylabel("coseno medio (SimCLR)")
    ax.set_title("Similitud entre vistas del mismo spot")
    ax.set_xticks(range(30, 181, 30))
    ax.legend()
    ax = axes[1]
    r = rid[rid["mode"] == "random"].groupby(["district", "n_views_query"])["top1"].agg(["mean", "std"]).reset_index()
    for c, (d, g) in zip(figstyle.SERIES, r.groupby("district")):
        ax.errorbar(g["n_views_query"], g["mean"], yerr=g["std"].fillna(0), fmt="-o", color=c, capsize=3,
                    label=d.replace("_", " ").title())
    ax.set_xlabel("nº de vistas de la mitad A en la consulta")
    ax.set_ylabel("top-1 re-identificación")
    ax.set_title("Re-identificación split-half (galería: mitad B)")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "exp040_redundancy_reid.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-proxy", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    S, T = build_tensor()
    log(f"spots 12/12: {S['district'].value_counts().to_dict()}  T {T.shape}")
    S.to_csv(OUT / "spots_used.csv", index=False)
    red, er = redundancy(S, T, rng)
    red.to_csv(OUT / "redundancy_by_separation.csv", index=False, float_format="%.5f")
    er.to_csv(OUT / "effective_rank_by_spot.csv", index=False, float_format="%.4f")
    fid = fidelity(S, T, rng)
    fid.to_csv(OUT / "v2_fidelity.csv", index=False, float_format="%.5f")
    rid = reid(S, T)
    rid.to_csv(OUT / "reid_split_half.csv", index=False, float_format="%.5f")
    bv, _ = bovw(S, T)
    bv.to_csv(OUT / "bovw_district_holdout.csv", index=False, float_format="%.5f")
    proxy = None if args.skip_proxy else pp_proxy(S, T)
    if proxy is not None:
        proxy.to_csv(OUT / "pp_probe_proxy_sensitivity.csv", index=False, float_format="%.5f")
    figures(red, rid)

    same = red[red["kind"] == "same_spot"]
    summ = {"n_spots_12": S["district"].value_counts().to_dict(),
            "embedding": "SimCLR ViT-B/16 epoch_50, 768-d, L2 (imágenes no locales)",
            "cos_30": same[same.separation_deg == 30].set_index("district")["mean_cos"].round(4).to_dict(),
            "cos_180": same[same.separation_deg == 180].set_index("district")["mean_cos"].round(4).to_dict(),
            "cos_null": red[red.kind == "null_other_spot_any_heading"].set_index("district")["mean_cos"].round(4).to_dict(),
            "effective_rank_mean": er.groupby("district")["effective_rank"].mean().round(3).to_dict(),
            "v2_fidelity": fid.groupby(["district", "variant"])[["cos_to_V1_mean", "spearman_structure"]]
                              .mean().round(4).reset_index().to_dict("records"),
            "reid_top1": rid.groupby(["district", "n_views_query", "mode"])["top1"].mean().round(4)
                            .reset_index().to_dict("records"),
            "seconds": round(time.perf_counter() - t0),
            "environment": environment_info()}
    c = summ
    r1 = all(c["cos_30"][d] > c["cos_180"][d] > c["cos_null"][d] and c["effective_rank_mean"][d] < 6 for d in c["cos_30"])
    f2 = fid[fid.variant == "V2_4R"].groupby("district")[["cos_to_V1_mean", "spearman_structure"]].mean()
    r2 = bool(((f2["cos_to_V1_mean"] >= 0.95) & (f2["spearman_structure"] >= 0.90)).all())
    t = rid[rid["mode"] == "random"].groupby(["district", "n_views_query"])["top1"].mean().unstack()
    r3 = bool((t[3] >= 0.9 * t[6]).all())
    summ["criteria"] = {"R1_redundancy": bool(r1), "R2_four_views_suffice_for_fidelity": r2,
                        "R3_reid_saturates_by_3_of_6": r3}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    log(json.dumps(summ["criteria"]))
    log(f"listo en {summ['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
