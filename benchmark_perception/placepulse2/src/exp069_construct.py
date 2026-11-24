"""
EXP-069 — Validez de constructo: ¿"seguro" es lo mismo que "rico" o "bonito"? Pre-registro: docs/EXPERIMENT_REGISTRY.md,
"Programa V3 (14/11/2026)".

Para cada uno de los 6 estudios de Place Pulse 2.0 se entrena la MISMA cabeza lineal-BT (receta train_probe.py) sobre
las MISMAS features DINOv2-B congeladas, con la MISMA partición de imágenes de image_disjoint. Luego:
  H-069a  exactitud de la cabeza 'wealthier' en los pares de test de 'safe';
  H-069b  exactitud de la cabeza safe tras quitarle el subespacio de las otras 5 cabezas;
  H-069c  combinación lineal (ajustada en train de safe) de los 5 puntajes ajenos vs cabeza safe;
  control humano: correlación de tasas de victoria por imagen entre atributos, desatenuada por fiabilidad split-half.

Uso: python src/exp069_construct.py [--boot 1000]      Salida: outputs/exp069/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, environment_info, log  # noqa: E402
from v3_common import (ENCODER, REG_ACC, SEED, assign_rows, fit_linear_bt, image_partition, load_pairs,  # noqa: E402
                       load_split, load_Z, pair_idx, pigeonhole_ci, sigmoid, verify_partition)

OUT = OUTPUTS / "exp069"
VOTES = Path(__file__).resolve().parents[3] / "external_assets" / "placepulse2" / "extracted" / "votes.tsv"
STUDIES = {"50a68a51fdc9f05596000002": "safe", "50f62c41a84ea7c5fdd2e454": "lively",
           "50f62c68a84ea7c5fdd2e456": "boring", "50f62cb7a84ea7c5fdd2e458": "wealthy",
           "50f62ccfa84ea7c5fdd2e459": "depressing", "5217c351ad93a7d3e7b07a64": "beautiful"}
ES = {"safe": "seguro", "lively": "animado", "boring": "aburrido", "wealthy": "rico", "depressing": "deprimente",
      "beautiful": "bonito"}
OTHERS = ["wealthy", "beautiful", "lively", "boring", "depressing"]


def load_votes(row: dict) -> pd.DataFrame:
    v = pd.read_csv(VOTES, sep="\t", dtype=str, usecols=["choice", "left", "right", "study_id"])
    v = v[v["study_id"].isin(STUDIES) & v["choice"].isin(["left", "right"])].copy()
    v["attr"] = v["study_id"].map(STUDIES)
    v["label"] = (v["choice"] == "left").astype(int)
    v = v[v["left"].isin(row) & v["right"].isin(row)]
    return v.reset_index(drop=True)


def human_control(v: pd.DataFrame, rng, min_app: int = 3, min_rel: int = 6, n_split: int = 20) -> pd.DataFrame:
    """Tasa de victoria por imagen y atributo; corr. de Spearman entre atributos (imágenes con ≥ min_app apariciones en
    ambos), y fiabilidad split-half (imágenes con ≥ min_rel apariciones; Spearman–Brown) para desatenuar."""
    from scipy.stats import spearmanr
    long = pd.concat([pd.DataFrame({"img": v["left"], "attr": v["attr"], "win": v["label"]}),
                      pd.DataFrame({"img": v["right"], "attr": v["attr"], "win": 1 - v["label"]})], ignore_index=True)
    agg = long.groupby(["img", "attr"])["win"].agg(["mean", "size"]).reset_index()
    rate = agg.pivot(index="img", columns="attr", values="mean")
    cnt = agg.pivot(index="img", columns="attr", values="size").fillna(0)
    rel = {}
    for a in STUDIES.values():
        L = long[long["attr"] == a]
        ok = cnt.index[cnt[a] >= min_rel]
        L = L[L["img"].isin(ok)].copy()
        rs = []
        for _ in range(n_split):
            L["h"] = rng.integers(0, 2, len(L))
            h = L.groupby(["img", "h"])["win"].mean().unstack().dropna()
            r = spearmanr(h[0], h[1]).statistic
            rs.append(2 * r / (1 + r))
        rel[a] = float(np.mean(rs))
    rows = []
    for a in OTHERS:
        m = (cnt["safe"] >= min_app) & (cnt[a] >= min_app)
        r = spearmanr(rate.loc[m, "safe"], rate.loc[m, a]).statistic
        # fiabilidad a la cantidad de apariciones usada en la correlación (Spearman–Brown desde la de min_rel)
        k_s = cnt.loc[m, "safe"].mean() / (min_rel * 1.0)
        k_a = cnt.loc[m, a].mean() / (min_rel * 1.0)
        rel_s = k_s * rel["safe"] / (1 + (k_s - 1) * rel["safe"])
        rel_a = k_a * rel[a] / (1 + (k_a - 1) * rel[a])
        rows.append({"attr": a, "n_images": int(m.sum()), "rho_winrate": float(r),
                     "rel_safe": float(rel_s), "rel_attr": float(rel_a),
                     "rho_disattenuated": float(r / np.sqrt(max(rel_s * rel_a, 1e-6)))})
    out = pd.DataFrame(rows)
    out.attrs["split_half_reliability_at_min_rel"] = rel
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, row = load_Z(ENCODER)
    df = load_pairs()
    split = load_split("image_disjoint")
    assign = image_partition(df)
    verify_partition(df, assign, split)
    v = load_votes(row)
    v["part"] = assign_rows(v["left"], v["right"], assign)
    log(f"votos decisivos por atributo (con features): {v['attr'].value_counts().to_dict()}")

    G, info, counts = {}, {}, {}
    for a in STUDIES.values():
        if a == "safe":  # misma fuente que el resto del benchmark (pairs_safe.csv) para reproducir 0.6785
            tr, va = df.loc[split["train"]], df.loc[split["val"]]
        else:
            tr, va = v[(v["attr"] == a) & (v["part"] == "train")], v[(v["attr"] == a) & (v["part"] == "val")]
        li, ri, k = pair_idx(tr["left"], tr["right"], row)
        vli, vri, vk = pair_idx(va["left"], va["right"], row)
        G[a], info[a] = fit_linear_bt(X, li, ri, tr["label"].to_numpy()[k], vli, vri, va["label"].to_numpy()[vk])
        counts[a] = {"train": int(k.sum()), "val": int(vk.sum()),
                     "test": int(((v["attr"] == a) & (v["part"] == "test")).sum()) if a != "safe" else len(split["test"])}
        log(f"  cabeza {a}: train {counts[a]['train']:,} val {counts[a]['val']:,} ép. {info[a]['epochs_best']}")

    # pares de test de safe
    T = df.loc[split["test"]]
    tli, tri, _ = pair_idx(T["left"], T["right"], row)
    ty = T["label"].to_numpy()
    Tr = df.loc[split["train"]]
    rli, rri, _ = pair_idx(Tr["left"], Tr["right"], row)
    ry = Tr["label"].to_numpy()

    def correct(g, li=tli, ri=tri, y=ty):
        return (((g[li] - g[ri]) > 0) == (y == 1)).astype(float)
    c_safe = correct(G["safe"])
    acc_safe = c_safe.mean()
    log(f"safe test acc {acc_safe:.4f} (registrada {REG_ACC['image_disjoint']})")
    if abs(acc_safe - REG_ACC["image_disjoint"]) > 0.002:
        raise RuntimeError("no se reproduce la cabeza safe")
    res = {"n_votes_by_attr": counts, "safe_acc_test": float(acc_safe)}

    # exactitud de cada cabeza ajena en test de safe (signo elegido en TRAIN de safe)
    rows = []
    test_imgs = np.unique(np.r_[tli, tri])
    for a in OTHERS:
        sgn = 1.0 if correct(G[a], rli, rri, ry).mean() >= 0.5 else -1.0
        c = correct(sgn * G[a])
        e, lo, hi = pigeonhole_ci(c, None, tli, tri, B=args.boot)
        d, dlo, dhi = pigeonhole_ci(c, c_safe, tli, tri, B=args.boot)
        r_img = float(np.corrcoef(G["safe"][test_imgs], G[a][test_imgs])[0, 1])
        own = v[(v["attr"] == a) & (v["part"] == "test")]
        oli, ori, ok = pair_idx(own["left"], own["right"], row)
        own_acc = float(correct(G[a], oli, ori, own["label"].to_numpy()[ok]).mean())
        rows.append({"attr": a, "attr_es": ES[a], "sign": sgn, "acc_on_safe_test": e, "lo": lo, "hi": hi,
                     "delta_vs_safe_head": d, "dlo": dlo, "dhi": dhi, "pearson_score_test_images_vs_safe": r_img,
                     "own_attr_test_acc": own_acc, "n_own_test": int(ok.sum())})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "cross_attribute.csv", index=False)
    res["cross_attribute"] = tab.to_dict(orient="records")

    # H-069b: residualización del vector de pesos (recuperado exacto por mínimos cuadrados: g = X w)
    Xd = X.astype(np.float64)
    W = {a: np.linalg.lstsq(Xd, G[a], rcond=None)[0] for a in STUDIES.values()}
    recon = {a: float(np.corrcoef(Xd @ W[a], G[a])[0, 1]) for a in W}
    Wo = np.stack([W[a] for a in OTHERS], axis=1)
    w = W["safe"]
    w_perp = w - Wo @ np.linalg.lstsq(Wo, w, rcond=None)[0]
    g_perp = Xd @ w_perp
    c_perp = correct(g_perp)
    e, lo, hi = pigeonhole_ci(c_perp, None, tli, tri, B=args.boot)
    # la geometría relevante es la de los puntajes, no la del espacio de pesos: también se residualiza el PUNTAJE
    # de safe sobre los 5 puntajes ajenos (regresión por mínimos cuadrados en imágenes de TRAIN)
    tr_imgs = np.unique(np.r_[rli, rri])
    Go = np.stack([G[a] for a in OTHERS], axis=1)
    beta = np.linalg.lstsq(np.c_[Go[tr_imgs], np.ones(len(tr_imgs))], G["safe"][tr_imgs], rcond=None)[0]
    g_res = G["safe"] - np.c_[Go, np.ones(len(Go))] @ beta
    c_res = correct(g_res)
    e2, lo2, hi2 = pigeonhole_ci(c_res, None, tli, tri, B=args.boot)
    r2 = 1 - np.var(g_res[tr_imgs]) / np.var(G["safe"][tr_imgs])
    res["residual"] = {"weight_space_projection": {"acc": e, "ci": [lo, hi],
                                                   "norm_retained": float(np.linalg.norm(w_perp) / np.linalg.norm(w))},
                       "score_space_residual": {"acc": e2, "ci": [lo2, hi2], "R2_safe_on_others_train_imgs": float(r2),
                                                "beta": dict(zip(OTHERS + ["const"], map(float, beta)))},
                       "weight_recovery_corr": recon}

    # H-069c: logística sobre diferencias de los 5 puntajes ajenos, ajustada en train de safe
    from sklearn.linear_model import LogisticRegression
    Dtr = Go[rli] - Go[rri]
    Dte = Go[tli] - Go[tri]
    lr = LogisticRegression(C=1e6, fit_intercept=False, max_iter=2000).fit(Dtr, ry)
    c_comb = ((lr.decision_function(Dte) > 0) == (ty == 1)).astype(float)
    d, dlo, dhi = pigeonhole_ci(c_comb, c_safe, tli, tri, B=args.boot)
    res["combination"] = {"acc": float(c_comb.mean()), "delta_vs_safe_head": [d, dlo, dhi],
                          "coef": dict(zip(OTHERS, map(float, lr.coef_[0])))}
    # correlaciones entre puntajes de los 6 atributos en imágenes de test
    S = pd.DataFrame({ES[a]: G[a][test_imgs] for a in STUDIES.values()})
    S.corr().round(4).to_csv(OUT / "score_correlations_test_images.csv")

    # control humano (sin modelo)
    hc = human_control(v.assign(), np.random.default_rng(SEED))
    hc.to_csv(OUT / "human_control.csv", index=False)
    res["human_control"] = {"table": hc.to_dict(orient="records"),
                            "split_half_reliability": hc.attrs["split_half_reliability_at_min_rel"]}

    # figura
    import matplotlib.pyplot as plt
    from figstyle import INK2, NEUTRAL, SERIES, setup
    setup()
    (OUT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = [f"cabeza '{ES[a]}'" + (" (signo −)" if s < 0 else "") for a, s in zip(tab["attr"], tab["sign"])]
    labels += ["combinación de las 5", "'seguro' sin lo común", "cabeza 'seguro'"]
    vals = list(tab["acc_on_safe_test"]) + [c_comb.mean(), e2, acc_safe]
    los = list(tab["lo"]) + [c_comb.mean(), lo2, acc_safe]
    his = list(tab["hi"]) + [c_comb.mean(), hi2, acc_safe]
    yy = np.arange(len(vals))
    cols = [SERIES[0]] * len(tab) + [SERIES[3], SERIES[1], SERIES[2]]
    ax.barh(yy, np.array(vals) * 100 - 50, left=50, color=cols, height=0.6)
    ax.errorbar(np.array(vals) * 100, yy, xerr=[(np.array(vals) - los) * 100, (np.array(his) - vals) * 100],
                fmt="none", ecolor=INK2, capsize=2)
    ax.set_yticks(yy, labels)
    ax.set_xlim(48, 70)
    ax.axvline(50, color=NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("Exactitud en los votos de 'más seguro' (fotos nuevas, %)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "construct_validity_es.png", dpi=200)

    res["meta"] = {"encoder": ENCODER, "seed": SEED, "boot": args.boot, "runtime_s": round(time.time() - t0),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                   "prereg": "docs/EXPERIMENT_REGISTRY.md · Programa V3 (14/11/2026)"}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"args": vars(args), "env": environment_info()}, indent=2,
                                                ensure_ascii=False), encoding="utf-8")
    log(json.dumps({k: res[k] for k in ("safe_acc_test", "residual", "combination")}, indent=1, ensure_ascii=False))
    log(tab.round(4).to_string())
    log(hc.round(3).to_string())
    log(f"listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
