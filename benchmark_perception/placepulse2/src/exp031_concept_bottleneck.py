"""
EXP-031 — Conceptos interpretables SIN anotadores: concept bottleneck con CLIP sobre el codebook (RQ2).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa EXP-028..031, 2026-10-09).

Cada concepto del codebook se mide zero-shot con CLIP como P(prompt positivo | par contrastivo), usando los
pares YA definidos en run_poc_concepts.py (EXP-005/006) + `sightline_obstruction`. Con esos 17 números por
imagen se entrena el mismo probe BT lineal. Preguntas: (a) cuánta señal perceptual conservan 17 conceptos con
nombre, frente a CLIP completo (512-d) y frente a 17 "conceptos" aleatorios sin relación urbana; (b) si el
signo de cada peso coincide con la dirección que el codebook atribuye a la literatura.
Inspiración: Label-free CBM (Oikarinen et al., ICLR 2023); Concept Bottleneck Models (Koh et al., ICML 2020).

LÍMITE: que un concepto prediga no prueba que CLIP lo MIDA bien (validez de medición, nivel 2 de M10).

Uso:  python exp031_concept_bottleneck.py      (requiere extract_clip.py)   Salidas: outputs/exp031/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression

from exp026_additive_context import load_images
from probe_utils import load_pairs, load_split, train_scores, votes
from run_poc_concepts import CLIP_PROMPTS
from utils import FEATURES, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp031"
PROMPTS = dict(CLIP_PROMPTS)
PROMPTS["sightline_obstruction"] = ("a street where the view ahead is blocked by walls, hedges or large vehicles",
                                    "a street with a clear and unobstructed view ahead")
# predicciones direccionales del codebook (literature_expectation = generally positive/negative)
EXPECTED = {"visual_openness": +1, "sightline_obstruction": -1, "graffiti": -1, "litter": -1,
            "facade_condition": -1, "vacant_lot": -1, "sidewalk": +1, "commercial_activity": +1}
NOUNS = ["banana", "guitar", "elephant", "teapot", "violin", "penguin", "pineapple", "dinosaur", "umbrella",
         "saxophone", "giraffe", "cupcake", "telescope", "octopus", "kangaroo", "harmonica", "strawberry",
         "zebra", "accordion", "lobster", "flamingo", "pretzel", "trombone", "koala", "pumpkin", "walrus",
         "croissant", "xylophone", "hedgehog", "avocado", "camel", "tuba", "jellyfish", "mango", "parrot",
         "cello", "squirrel", "donut", "rhinoceros", "ukulele"]
N_RANDOM = 10
N_BOOT_W = 200


def text_embed(pairs_txt: list[tuple[str, str]]):
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok([s for p in pairs_txt for s in p])).float()
        t = torch.nn.functional.normalize(t, dim=-1).numpy()
        scale = float(model.logit_scale.exp())
    return t.reshape(len(pairs_txt), 2, -1), scale


def concept_scores(C: np.ndarray, T: np.ndarray, scale: float) -> np.ndarray:
    """P(prompt positivo) por imagen y concepto: softmax de 2 vías con la escala de CLIP."""
    lp, ln = scale * (C @ T[:, 0].T), scale * (C @ T[:, 1].T)
    return 1.0 / (1.0 + np.exp(-(lp - ln)))


def standardize(F: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    mu, sd = F[train_rows].mean(0), F[train_rows].std(0) + 1e-8
    return ((F - mu) / sd).astype(np.float32)


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, _ = load_images("dinov2_vit_b14")
    ids = man["location_id"].tolist()
    cidx = json.loads((FEATURES / "clip_vit_b32" / "index.json").read_text(encoding="utf-8"))
    Cfull = np.load(FEATURES / "clip_vit_b32" / "features.npy")
    row_of = {lid: i for i, lid in enumerate(cidx["location_ids"])}
    C = Cfull[[row_of[i] for i in ids]]
    pairs = load_pairs()
    sp = load_split("image_disjoint")
    ch = load_split("city_holdout")
    tr_ids = set(votes(pairs, sp["train"])["id"])
    train_rows = np.array([i in tr_ids for i in ids])

    names = list(PROMPTS)
    T, scale = text_embed([PROMPTS[c] for c in names])
    F = standardize(concept_scores(C, T, scale), train_rows)
    pd.DataFrame(F, columns=names).assign(location_id=ids).to_csv(OUT / "concept_scores_std.csv.gz",
                                                                  index=False, float_format="%.4f")

    res = []

    def run(tag, X, protocol, split, key):
        _, info = train_scores(X, ids, pairs, split, key)
        m = info["metrics"]
        res.append(dict(variant=tag, protocol=protocol, split=key, dim=X.shape[1], n=m["n"],
                        accuracy=m["pairwise_accuracy"], ci95_low=m["ci95_low"], ci95_high=m["ci95_high"]))
        log(f"  {tag:<16} {protocol:<14} {key:<11} d={X.shape[1]:4d} acc={m['pairwise_accuracy']:.4f} "
            f"[{m['ci95_low']:.4f},{m['ci95_high']:.4f}]")
        return info

    info_cbm = run("concepts_17", F, "image_disjoint", sp, "test")
    run("clip_full", C, "image_disjoint", sp, "test")
    run("concepts_17", F, "city_holdout", ch, "test_strict")
    run("clip_full", C, "city_holdout", ch, "test_strict")
    rand_acc = []
    for r in range(N_RANDOM):
        nn = rng.choice(NOUNS, size=(len(names), 2), replace=True)
        Tr, _ = text_embed([(f"a photo of a {a}", f"a photo of a {b}") for a, b in nn])
        Fr = standardize(concept_scores(C, Tr, scale), train_rows)
        rand_acc.append(run(f"random_17_{r}", Fr, "image_disjoint", sp, "test")["metrics"]["pairwise_accuracy"])
    pd.DataFrame(res).to_csv(OUT / "cbm_accuracy.csv", index=False, float_format="%.5f")

    # pesos y signos: regresión logística sin intercepto sobre Δx (equivalente al probe BT lineal)
    tr = pairs.iloc[sp["train"]]
    pos = {lid: i for i, lid in enumerate(ids)}
    D = F[tr["left"].map(pos).to_numpy()] - F[tr["right"].map(pos).to_numpy()]
    y = tr["label"].to_numpy()
    lr = LogisticRegression(fit_intercept=False, C=1e4, max_iter=2000).fit(D, y)
    W = []
    for _ in range(N_BOOT_W):
        ix = rng.integers(0, len(y), len(y))
        W.append(LogisticRegression(fit_intercept=False, C=1e4, max_iter=2000).fit(D[ix], y[ix]).coef_.ravel())
    W = np.array(W)
    hum = votes(pairs, sp["train"]).groupby("id")["win"].agg(["size", "mean"]).query("size >= 3")["mean"]
    fi = pd.DataFrame(F, index=ids, columns=names).reindex(hum.index)
    rows = []
    for j, c in enumerate(names):
        lo, hi = np.percentile(W[:, j], [2.5, 97.5])
        rows.append(dict(concept=c, weight=lr.coef_[0, j], ci95_low=lo, ci95_high=hi,
                         probe_weight=float(info_cbm["weights"][j]),
                         marginal_rho_with_human=float(spearmanr(fi[c], hum)[0]),
                         expected_sign=EXPECTED.get(c, 0)))
    w = pd.DataFrame(rows)
    w["sign_ok"] = np.where(w["expected_sign"] == 0, np.nan, np.sign(w["weight"]) == w["expected_sign"])
    w["sign_ok_ci"] = np.where(w["expected_sign"] == 0, np.nan,
                               ((w["expected_sign"] > 0) & (w["ci95_low"] > 0)) |
                               ((w["expected_sign"] < 0) & (w["ci95_high"] < 0)))
    w["marginal_sign_ok"] = np.where(w["expected_sign"] == 0, np.nan,
                                     np.sign(w["marginal_rho_with_human"]) == w["expected_sign"])
    w.to_csv(OUT / "concept_weights.csv", index=False, float_format="%.4f")
    d = w[w["expected_sign"] != 0]
    k_ok = int(d["sign_ok"].sum())
    sign_p = binomtest(k_ok, len(d), 0.5, alternative="greater").pvalue
    log(w.round(3).to_string(index=False))

    acc_cbm = [r for r in res if r["variant"] == "concepts_17" and r["protocol"] == "image_disjoint"][0]["accuracy"]
    acc_clip = [r for r in res if r["variant"] == "clip_full" and r["protocol"] == "image_disjoint"][0]["accuracy"]
    summary = {"seed": SEED, "n_random_draws": N_RANDOM, "acc_concepts": acc_cbm, "acc_clip_full": acc_clip,
               "retention": (acc_cbm - 0.5) / (acc_clip - 0.5), "acc_random_mean": float(np.mean(rand_acc)),
               "acc_random_max": float(np.max(rand_acc)), "random_accs": rand_acc,
               "signs_ok": k_ok, "signs_total": int(len(d)), "sign_test_p": float(sign_p),
               "signs_ok_ci": int(d["sign_ok_ci"].sum()), "marginal_signs_ok": int(d["marginal_sign_ok"].sum()),
               "seconds": round(time.perf_counter() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
