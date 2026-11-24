"""
EXP-053 — ¿Qué se ve en Barranco que lo hace parecer más seguro que La Victoria? Descomposición de la brecha de
percepción PREDICHA entre distritos en palabras visuales (parches DINOv2, vocabulario K = 100 de EXP-044).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

Ridge (estandarizado) de la puntuación BT de PP sobre histogramas L1 de palabras; en Lima, histograma medio de las 12
vistas por spot. Por linealidad, Δ(A − B) = Σ_k c_k con c_k = (coef_k / σ_k)·(x̄_k^A − x̄_k^B). Bootstrap de spots.

Uso:  python exp053_bias_words.py        Salidas: outputs/exp053/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp044_patch_bovw import ALPHAS, OUT as EXP044, features, load_blocks, prototype_mosaic
from exp052_connectivity_effect_size import DINO, PLIMA_EXPORTS
from lima_views import COLLECTOR_DATA
from utils import OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp053"
VN = "K100_s0"
N_BOOT = 2000
TOP = 8
PAIRS = [("barranco", "la_victoria"), ("san_isidro", "la_victoria"), ("barranco", "san_isidro"),
         ("lince", "la_victoria")]
NAMES = {"barranco": "Barranco", "la_victoria": "La Victoria", "san_isidro": "San Isidro", "lince": "Lince"}


def fit_pp():
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from exp038_bt_target import bt_global
    from probe_utils import load_pairs
    ids, F = load_blocks()
    y = bt_global(load_pairs()).reindex(ids).to_numpy()
    ok = np.flatnonzero(np.isfinite(y))
    X = features(f"bovw_l1:{VN}", {k: v[ok] for k, v in F.items() if not k.startswith("proto_")}, ok)
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X), y[ok])
    w = m.coef_ / sc.scale_                                   # coeficiente por unidad de frecuencia
    b = m.intercept_ - (sc.mean_ * w).sum()
    log(f"ridge PP: {len(ok):,} imágenes, α = {m.alpha_:.0f}, R² in-sample = {m.score(sc.transform(X), y[ok]):.3f}")
    return w, b, ids, F


def lima_spot_hists() -> tuple[pd.DataFrame, np.ndarray, dict]:
    S = pd.read_csv(OUTPUTS / "exp051_dinov2" / "spot_predicted_perception.csv", dtype={"spot_id": str})
    H = np.zeros((len(S), 100))
    views = {}
    for d, folder in DINO.items():
        z = np.load(PLIMA_EXPORTS / folder / "dinov2.npz")[f"hist_{VN}"].astype(np.float64)
        paths = json.loads((PLIMA_EXPORTS / folder / "dinov2_index.json").read_text(encoding="utf-8"))["paths"]
        P = z / z.sum(1, keepdims=True)
        spot = np.array([p.split("/")[1] for p in paths])
        views[d] = (paths, P)
        for i in np.flatnonzero((S["district"] == d).to_numpy()):
            m = spot == S.at[i, "spot_id"]
            H[i] = P[m].mean(0)
    assert (H.sum(1) > 0.99).all()
    return S, H, views


def lima_overlays(words: list[int], favored: list[str], views: dict, out) -> None:
    """Vistas de Lima con mayor frecuencia de cada palabra; se marcan los parches asignados a esa palabra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    from encoders import load_dinov2
    from visual_repr import l2n
    spec = load_dinov2("cpu")
    C = np.load(EXP044 / f"vocab_{VN}.npy")
    fig, axes = plt.subplots(len(words), 3, figsize=(10.5, 2.25 * len(words)), squeeze=False)
    for r, (k, d) in enumerate(zip(words, favored)):
        paths, P = views[d]
        for c, j in enumerate(np.argsort(-P[:, k])[:3]):
            with Image.open(COLLECTOR_DATA / paths[j]) as im:
                im = im.convert("RGB")
                with torch.no_grad():
                    f = spec.model.forward_features(spec.transform(im)[None]).float().numpy()[0]
                tok = l2n(f[spec.model.num_prefix_tokens:])
                hit = ((1.0 - 2.0 * tok @ C.T + (C ** 2).sum(1)[None]).argmin(1) == k).reshape(16, 16)
                ax = axes[r, c]
                ax.imshow(im)
                W, Hh = im.size
                ax.imshow(np.ma.masked_where(~hit, hit), extent=(0, W, Hh, 0), alpha=0.45, cmap="autumn",
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"w{k}\n{NAMES[d]}", fontsize=8)
    fig.suptitle("EXP-053 · vistas de Lima con más parches de cada palabra (parches marcados)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    w, b, ids_pp, F_pp = fit_pp()
    S, H, views = lima_spot_hists()
    S["bovw_score"] = H @ w + b
    val = {}
    for d, g in S.groupby("district"):
        val[d] = float(spearmanr(g["bovw_score"], g["score"]).statistic)
    log(f"validez ρ(BoVW, probe CLS) por spot: {val}")
    dist_mean = S.groupby("district")[["bovw_score", "score"]].mean()
    log(f"medias por distrito:\n{dist_mean}")

    rows, contrib = [], []
    for a, c in PAIRS:
        ia, ic = np.flatnonzero((S["district"] == a).to_numpy()), np.flatnonzero((S["district"] == c).to_numpy())
        dx = H[ia].mean(0) - H[ic].mean(0)
        ck = w * dx
        B = np.empty((N_BOOT, 100))
        for t in range(N_BOOT):
            B[t] = w * (H[rng.choice(ia, len(ia))].mean(0) - H[rng.choice(ic, len(ic))].mean(0))
        tot, totb = ck.sum(), B.sum(1)
        order = np.argsort(-np.abs(ck))
        rows.append(dict(pair=f"{a} - {c}", delta_bovw=tot, ci_low=np.percentile(totb, 2.5),
                         ci_high=np.percentile(totb, 97.5),
                         delta_cls_probe=float(S.loc[ia, "score"].mean() - S.loc[ic, "score"].mean()),
                         pct_explained_top5=100 * float(ck[order[:5]].sum() / tot) if tot else np.nan,
                         top5_words=",".join(map(str, order[:5]))))
        for rank, k in enumerate(order[:TOP], 1):
            contrib.append(dict(pair=f"{a} - {c}", rank=rank, word=int(k), contribution=ck[k],
                                ci_low=np.percentile(B[:, k], 2.5), ci_high=np.percentile(B[:, k], 97.5),
                                pct_of_gap=100 * ck[k] / tot, coef_per_freq=w[k],
                                freq_a=H[ia, k].mean(), freq_b=H[ic, k].mean()))
        log(f"{a} − {c}: {rows[-1]}")
    R = pd.DataFrame(rows)
    Ct = pd.DataFrame(contrib)
    R.to_csv(OUT / "gap_decomposition.csv", index=False, float_format="%.5f")
    Ct.to_csv(OUT / "top_word_contributions.csv", index=False, float_format="%.5f")
    S[["district", "spot_id", "seg_idx", "score", "pct_pp", "bovw_score"]].to_csv(
        OUT / "spot_bovw_score.csv", index=False, float_format="%.5f")

    main_pair = Ct[Ct["pair"] == "barranco - la_victoria"]
    words = main_pair["word"].tolist()
    favored = ["barranco" if fa > fb else "la_victoria" for fa, fb in zip(main_pair["freq_a"], main_pair["freq_b"])]
    # mosaico PP con el signo del coeficiente estandarizado (+ = más seguro)
    prototype_mosaic(F_pp, ids_pp, VN, words, (w[words] / np.abs(w).max()).tolist(), OUT / "pp_prototypes_bias_words.png")
    lima_overlays(words, favored, views, OUT / "lima_views_bias_words.png")
    summ = {"validity_rho_by_district": val,
            "validity_met": bool(min(val.values()) >= 0.5 and
                                 np.sign(R.iloc[0]["delta_bovw"]) == np.sign(R.iloc[0]["delta_cls_probe"])),
            "pairs": R.to_dict("records"), "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=float), encoding="utf-8")
    log(f"criterio de validez: {summ['validity_met']} · listo en {summ['minutes']} min")


if __name__ == "__main__":
    main()
