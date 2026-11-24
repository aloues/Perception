"""
EXP-060 — Fidelidad de las palabras visuales (RQ2): ¿las regiones que una palabra señala son las que el modelo usa?

Para las 10 palabras con mayor |coef| (K100_s0, EXP-044): imágenes de TEST de `image_disjoint` de las 3 ciudades piloto
(el probe no las vio) con ≥ MIN_PATCHES parches de la palabra. Por imagen: puntaje del probe DINOv2 (CLS L2 · w, el de
RQ1) en la foto original, con los parches de la palabra tapados (valor 0 tras normalizar = color medio de ImageNet) y con
el MISMO número de parches tapados al azar fuera de la palabra (2 sorteos). Fidelidad por imagen:
    F = −signo(coef) · [(s_palabra − s_orig) − (s_azar − s_orig)]
F > 0: tapar una palabra "segura" baja la seguridad más que tapar al azar (o una "insegura" la sube).
Criterio (registro): media de F sobre las 10 palabras > 0 con IC 95 % (bootstrap de imágenes) que excluye 0 y F > 0 en ≥ 7/10.

Uso:  python exp060_word_faithfulness.py        Salidas: outputs/exp060/
"""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from encoders import load_dinov2
from exp044_patch_bovw import OUT as OUT44, load_blocks, manifest
from exp051_perception_crime_gap import pp_probe
from probe_utils import load_pairs, load_split
from utils import OUTPUTS, ZIP_PATH, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp060"
VOCAB = "K100_s0"
MIN_PATCHES = 10
PER_WORD = 60
N_RAND = 2
GRID, PATCH = 16, 14


def assign(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Palabra (centro k-means más cercano en euclídeo) de cada parche L2-normalizado."""
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    return np.argmin((C ** 2).sum(1)[None] - 2 * P @ C.T, axis=1)


def masked(x: torch.Tensor, patches: np.ndarray) -> torch.Tensor:
    y = x.clone()
    for p in patches:
        r, c = divmod(int(p), GRID)
        y[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 0.0
    return y


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    w, _, acc = pp_probe("dinov2_vit_b14")
    w = torch.from_numpy(np.asarray(w, np.float32))
    log(f"probe DINOv2 (image_disjoint) exactitud test {acc:.4f}")
    C = np.load(OUT44 / f"vocab_{VOCAB}.npy")
    top = pd.read_csv(OUT44 / f"top_words_{VOCAB}.csv")
    ids, B = load_blocks()
    H = B[f"hist_{VOCAB}"]
    pairs = load_pairs()
    te = pairs.iloc[load_split("image_disjoint")["test"]]
    test_imgs = set(te["left"]) | set(te["right"])
    path_of = dict(zip(manifest()["location_id"], manifest()["image_path"]))
    idx = np.array([i for i, lid in enumerate(ids) if lid in test_imgs])
    log(f"imágenes piloto de test: {len(idx):,}")
    spec = load_dinov2("cpu")
    torch.set_num_threads(10)

    def forward(xs):
        with torch.no_grad():
            f = spec.model.forward_features(torch.stack(xs)).float()
        cls = torch.nn.functional.normalize(f[:, 0], dim=-1)
        return (cls @ w).numpy(), f[:, 1:].numpy()

    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for wd, coef in zip(top["word"], top["coef_std"]):
            cand = idx[H[idx, wd] >= MIN_PATCHES]
            pick = rng.choice(cand, min(PER_WORD, len(cand)), replace=False)
            for i in pick:
                lid = ids[i]
                with zf.open(path_of[lid]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    x = spec.transform(im.convert("RGB"))
                s0, pt = forward([x])
                words = assign(pt[0], C)
                wp = np.flatnonzero(words == wd)
                if len(wp) < MIN_PATCHES // 2:
                    continue
                others = np.flatnonzero(words != wd)
                xs = [masked(x, wp)] + [masked(x, rng.choice(others, len(wp), replace=False)) for _ in range(N_RAND)]
                s, _ = forward(xs)
                d_word, d_rand = s[0] - s0[0], s[1:].mean() - s0[0]
                rows.append(dict(word=int(wd), coef=float(coef), image=lid, n_patches=len(wp), s_orig=float(s0[0]),
                                 d_word=float(d_word), d_rand=float(d_rand),
                                 F=float(-np.sign(coef) * (d_word - d_rand))))
            log(f"palabra w{wd} (coef {coef:+.3f}): {sum(r['word'] == wd for r in rows)} imágenes · "
                f"{(time.perf_counter() - t0) / 60:.1f} min")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "per_image.csv", index=False, float_format="%.5f")
    summ_rows = []
    for wd, g in R.groupby("word", sort=False):
        b = [g["F"].to_numpy()[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
        summ_rows.append(dict(word=int(wd), coef=float(g["coef"].iloc[0]), n=len(g), patches_median=float(g["n_patches"].median()),
                              mean_d_word=g["d_word"].mean(), mean_d_rand=g["d_rand"].mean(), F=g["F"].mean(),
                              ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5)))
    S = pd.DataFrame(summ_rows)
    per_word = [R[R["word"] == k]["F"].to_numpy() for k in S["word"]]
    boot = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in per_word]) for _ in range(2000)]
    overall = dict(F_mean_over_words=float(S["F"].mean()), ci_low=float(np.percentile(boot, 2.5)),
                   ci_high=float(np.percentile(boot, 97.5)), words_F_positive=int((S["F"] > 0).sum()), n_words=len(S),
                   words_ci_above_0=int((S["ci_low"] > 0).sum()), minutes=(time.perf_counter() - t0) / 60)
    S.to_csv(OUT / "per_word.csv", index=False, float_format="%.5f")
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    log(S.round(4).to_string(index=False))
    log(overall)


if __name__ == "__main__":
    main()
