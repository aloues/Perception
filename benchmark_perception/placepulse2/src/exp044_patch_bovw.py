"""
EXP-044 — BoVW de PARCHES de DINOv2 en Place Pulse con votos humanos (D5 del prompt maestro 2026-10-29).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29, con la enmienda A1).

EXP-040 probó un BoVW débil (1 descriptor global por vista, ≤ 6 palabras por spot). Aquí cada imagen aporta sus 256
tokens de parche (ViT-B/14 a 224 px), que es el BoVW clásico que describió el asesor. Tres fases:

  vocab    2,000 imágenes de ciudades NO piloto (sin etiquetas usadas) → 100 parches L2 por imagen (200k) →
           MiniBatchKMeans K ∈ {100, 120, 500} × semillas {0, 1}. Ninguna imagen piloto entra al vocabulario.
  extract  imágenes de las 3 ciudades piloto en bloques (subproceso por bloque, escritura atómica, M03): CLS,
           media de parches, histograma de conteos por vocabulario y candidatos a prototipo por palabra. Los tokens
           NO se guardan (12 GB libres en disco).
  analyze  objetivo = puntuación latente BT por imagen (EXP-038, todos los votos) · ridge con CV espacial por
           bloques de 1 km (5 pliegues) y holdout de ciudad · Spearman con IC por bootstrap de bloques · techo
           split-half · ARI/ocupación · mosaico de prototipos (RQ2).

Uso:  python exp044_patch_bovw.py --stage vocab|extract|analyze        Salidas: outputs/exp044/
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import MANIFESTS, OUTPUTS, ZIP_PATH, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp044"
STAGE = OUT / "blocks"
CITIES = ["Atlanta", "Berlin", "Santiago"]
KS, SEEDS = (100, 120, 500), (0, 1)
VOCAB_IMAGES, PATCHES_PER_IMAGE = 2000, 100
BLOCK = 2048
N_PROTO = 12
CELL_M, N_FOLDS, N_BOOT = 1000.0, 5, 2000
ALPHAS = np.logspace(-2, 5, 15)


def vocab_names() -> list[str]:
    return [f"K{k}_s{s}" for k in KS for s in SEEDS]


def manifest() -> pd.DataFrame:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    return man[man["image_in_zip"] == "True"].reset_index(drop=True)


def pilot_ids() -> tuple[list[str], dict]:
    man = manifest()
    p = man[man["city"].isin(CITIES)]
    return p["location_id"].tolist(), dict(zip(man["location_id"], man["image_path"]))


# ─── DINOv2: tokens de parche ─────────────────────────────────────────────────

def _model(threads: int):
    import torch

    from encoders import load_dinov2
    torch.set_num_threads(threads)
    return load_dinov2("cpu")


def _batches(ids, path_of, spec, batch_size):
    """Genera (ids_lote, f) con f = forward_features [B, prefijo + 256, 768] (después de la LayerNorm final)."""
    import torch
    from PIL import Image
    with zipfile.ZipFile(ZIP_PATH) as zf, torch.no_grad():
        for a in range(0, len(ids), batch_size):
            chunk = ids[a:a + batch_size]
            tens = []
            for lid in chunk:
                with zf.open(path_of[lid]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    tens.append(spec.transform(im.convert("RGB")))
            yield chunk, spec.model.forward_features(torch.stack(tens)).float().numpy()
            if (a // batch_size) % 50 == 0:
                gc.collect()


def l2n(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12, None)


def stage_vocab(threads: int, batch_size: int) -> None:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    rng = np.random.default_rng(SEED)
    pool = man[~man["city"].isin(CITIES)]["location_id"].to_numpy()
    ids = sorted(rng.choice(pool, VOCAB_IMAGES, replace=False).tolist())
    path_of = dict(zip(man["location_id"], man["image_path"]))
    spec = _model(threads)
    npre = spec.model.num_prefix_tokens
    P, t0 = [], time.perf_counter()
    for chunk, f in _batches(ids, path_of, spec, batch_size):
        tok = l2n(f[:, npre:])
        for b in range(len(chunk)):
            P.append(tok[b, rng.choice(tok.shape[1], PATCHES_PER_IMAGE, replace=False)])
    P = np.concatenate(P).astype(np.float32)
    log(f"vocab: {P.shape} parches de {len(ids)} imágenes de otras ciudades en {(time.perf_counter() - t0) / 60:.1f} min")
    labels, info = {}, {}
    for k in KS:
        for s in SEEDS:
            tt = time.perf_counter()
            km = MiniBatchKMeans(n_clusters=k, random_state=s, batch_size=8192, n_init=3, max_iter=200).fit(P)
            name = f"K{k}_s{s}"
            np.save(OUT / f"vocab_{name}.npy", km.cluster_centers_.astype(np.float32))
            labels[name] = km.labels_
            occ = np.bincount(km.labels_, minlength=k)
            info[name] = {"inertia": float(km.inertia_), "empty_bins": int((occ == 0).sum()),
                          "seconds": round(time.perf_counter() - tt)}
            log(f"  {name}: {info[name]}")
    ari = {f"K{k}": float(adjusted_rand_score(labels[f"K{k}_s0"], labels[f"K{k}_s1"])) for k in KS}
    (OUT / "vocab.json").write_text(json.dumps({
        "vocab_images": ids, "cities_excluded": CITIES, "patches_per_image": PATCHES_PER_IMAGE,
        "n_patches": int(len(P)), "patch_norm": "L2", "kmeans": info,
        "ari_between_seeds_on_fit_sample": ari}, indent=2), encoding="utf-8")
    log(f"ARI entre semillas: {ari}")


def worker(block: int, threads: int, batch_size: int) -> None:
    ids, path_of = pilot_ids()
    ids = ids[block * BLOCK:(block + 1) * BLOCK]
    spec = _model(threads)
    npre = spec.model.num_prefix_tokens
    C = {n: np.load(OUT / f"vocab_{n}.npy") for n in vocab_names()}
    C2 = {n: (c ** 2).sum(1) for n, c in C.items()}
    cls, avg, t0 = [], [], time.perf_counter()
    H = {n: [] for n in C}
    best = {n: [] for n in C}                      # por imagen: (dist mín, parche) a cada palabra
    for chunk, f in _batches(ids, path_of, spec, batch_size):
        cls.append(f[:, 0])
        avg.append(f[:, npre:].mean(1))
        tok = l2n(f[:, npre:])                     # [B, 256, 768]
        B, T, D = tok.shape
        flat = tok.reshape(-1, D)
        for n, c in C.items():
            d = 1.0 - 2.0 * flat @ c.T + C2[n][None]        # ‖p‖ = 1
            w = d.argmin(1)
            H[n].append(np.stack([np.bincount(w[b * T:(b + 1) * T], minlength=len(c)) for b in range(B)]))
            dm = np.full((B, len(c)), np.inf, np.float32)
            pi = np.zeros((B, len(c)), np.int16)
            dd = d[np.arange(len(w)), w].reshape(B, T)
            ww = w.reshape(B, T)
            for b in range(B):
                order = np.argsort(dd[b])[::-1]          # el último escrito (menor distancia) gana
                dm[b, ww[b, order]] = dd[b, order]
                pi[b, ww[b, order]] = order
            best[n].append((dm, pi))
    out = {"cls": np.concatenate(cls).astype(np.float32), "avg": np.concatenate(avg).astype(np.float32)}
    for n in C:
        out[f"hist_{n}"] = np.concatenate(H[n]).astype(np.uint16)
        dm = np.concatenate([b[0] for b in best[n]])
        pi = np.concatenate([b[1] for b in best[n]])
        top = np.argsort(dm, axis=0)[:N_PROTO]                       # [N_PROTO, K] filas de imagen
        out[f"proto_dist_{n}"] = np.take_along_axis(dm, top, 0)
        out[f"proto_row_{n}"] = (top + block * BLOCK).astype(np.int32)
        out[f"proto_patch_{n}"] = np.take_along_axis(pi, top, 0)
    assert out["cls"].shape == (len(ids), 768) and np.isfinite(out["cls"]).all()
    tmp = STAGE / f"block_{block:03d}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, STAGE / f"block_{block:03d}.npz")
    log(f"bloque {block}: {len(ids)} imágenes en {(time.perf_counter() - t0) / 60:.1f} min")


def stage_extract(threads: int, batch_size: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    ids, _ = pilot_ids()
    nb = (len(ids) + BLOCK - 1) // BLOCK
    log(f"{len(ids):,} imágenes piloto, {nb} bloques, {len(list(STAGE.glob('block_*[0-9].npz')))} hechos")
    for b in range(nb):
        if not (STAGE / f"block_{b:03d}.npz").exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(b),
                            "--threads", str(threads), "--batch-size", str(batch_size)], check=True)
    log("extracción completa")


# ─── análisis ─────────────────────────────────────────────────────────────────

def load_blocks() -> tuple[list[str], dict]:
    ids, _ = pilot_ids()
    nb = (len(ids) + BLOCK - 1) // BLOCK
    parts = [dict(np.load(STAGE / f"block_{b:03d}.npz")) for b in range(nb)]
    F = {"cls": np.concatenate([p["cls"] for p in parts]), "avg": np.concatenate([p["avg"] for p in parts])}
    for n in vocab_names():
        F[f"hist_{n}"] = np.concatenate([p[f"hist_{n}"] for p in parts]).astype(np.float64)
        d = np.concatenate([p[f"proto_dist_{n}"] for p in parts])
        r = np.concatenate([p[f"proto_row_{n}"] for p in parts])
        q = np.concatenate([p[f"proto_patch_{n}"] for p in parts])
        top = np.argsort(d, axis=0)[:N_PROTO]
        F[f"proto_{n}"] = (np.take_along_axis(d, top, 0), np.take_along_axis(r, top, 0), np.take_along_axis(q, top, 0))
    assert len(F["cls"]) == len(ids)
    return ids, F


def idf_from(Hc: np.ndarray) -> np.ndarray:
    return np.log((1 + len(Hc)) / (1 + (Hc > 0).sum(0))) + 1


def features(name: str, F: dict, tr: np.ndarray) -> np.ndarray:
    """Representación `name`; todo lo aprendido (idf) sale SOLO de las filas de entrenamiento `tr`."""
    cls = l2n(F["cls"])
    if name == "E0_cls":
        return cls
    if name == "patch_mean":
        return l2n(F["avg"])
    if name == "cls+patch_mean":
        return np.hstack([cls, l2n(F["avg"])])
    kind, vn = name.split(":")
    Hc = F[f"hist_{vn}"]
    if kind == "bovw_l1":
        return Hc / Hc.sum(1, keepdims=True)
    if kind == "bovw_tfidf":
        return l2n(Hc / Hc.sum(1, keepdims=True) * idf_from(Hc[tr]))
    if kind == "cls+bovw":
        return np.hstack([cls, l2n(Hc / Hc.sum(1, keepdims=True))])
    raise ValueError(name)


def ridge_oof(name: str, F: dict, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    pred, alphas = np.full(len(y), np.nan), []
    for tr, te in folds:
        X = features(name, F, tr)
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
        alphas.append(float(m.alpha_))
    return pred, alphas


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def boot_cells(y, preds: dict, cells: np.ndarray, base: str, rng) -> pd.DataFrame:
    """ρ por representación y Δρ vs `base` con bootstrap de bloques espaciales (mismo remuestreo para todas)."""
    from scipy.stats import rankdata
    uc, inv = np.unique(cells, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uc))]
    names = list(preds)
    B = np.zeros((N_BOOT, len(names)))
    for b in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uc), len(uc))])
        ry = rankdata(y[idx])
        for j, n in enumerate(names):
            B[b, j] = np.corrcoef(ry, rankdata(preds[n][idx]))[0, 1]
    j0 = names.index(base)
    rows = []
    for j, n in enumerate(names):
        d = B[:, j] - B[:, j0]
        rows.append(dict(representation=n, rho=spearman(y, preds[n]), rho_ci_low=np.percentile(B[:, j], 2.5),
                         rho_ci_high=np.percentile(B[:, j], 97.5), delta_vs_base=spearman(y, preds[n]) - spearman(y, preds[base]),
                         delta_ci_low=np.percentile(d, 2.5), delta_ci_high=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def split_half_ceiling(pairs: pd.DataFrame, keep: list[str], rng, reps: int = 2) -> dict:
    from exp038_bt_target import bt_global
    rs = []
    for _ in range(reps):
        m = rng.random(len(pairs)) < 0.5
        a, b = bt_global(pairs[m]), bt_global(pairs[~m])
        common = [i for i in keep if i in a.index and i in b.index]
        rs.append(spearman(a.loc[common], b.loc[common]))
    r = float(np.mean(rs))
    return {"split_half_rho": r, "spearman_brown": 2 * r / (1 + r), "reps": reps}


def prototype_mosaic(F: dict, ids: list[str], vn: str, words: list[int], coefs: list[float], out: Path) -> None:
    """Para cada palabra, los N_PROTO parches más cercanos al centroide (imágenes distintas), con contexto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    _, path_of = pilot_ids()
    d, r, q = F[f"proto_{vn}"]
    fig, axes = plt.subplots(len(words), N_PROTO, figsize=(1.1 * N_PROTO, 1.25 * len(words)), squeeze=False)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i, (w, c) in enumerate(zip(words, coefs)):
            for j in range(N_PROTO):
                ax = axes[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                if not np.isfinite(d[j, w]):                 # menos de N_PROTO imágenes con esta palabra
                    continue
                lid = ids[int(r[j, w])]
                with zf.open(path_of[lid]) as fh:
                    im = Image.open(io.BytesIO(fh.read())).convert("RGB").resize((224, 224))
                py, px = divmod(int(q[j, w]), 16)
                x0, y0 = max(0, px * 14 - 21), max(0, py * 14 - 21)      # parche de 14 px + 21 px de contexto
                ax.imshow(im.crop((x0, y0, min(224, px * 14 + 35), min(224, py * 14 + 35))))
                if j == 0:
                    ax.set_ylabel(f"w{w}\n{c:+.2f}", fontsize=7)
    fig.suptitle(f"EXP-044 · prototipos de las 10 palabras con mayor |coef| ({vn}); signo + = más seguro", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def stage_analyze() -> None:
    from sklearn.model_selection import GroupKFold

    from exp026_additive_context import COORD_RE
    from exp038_bt_target import bt_global
    from probe_utils import cell_ids, load_pairs
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    ids, F = load_blocks()
    man = manifest().set_index("location_id").loc[ids].reset_index()
    ll = man["image_path"].str.extract(COORD_RE).astype(float)
    man["lat"], man["lon"] = ll[0], ll[1]
    pairs = load_pairs()
    lam = bt_global(pairs)
    nvotes = pd.concat([pairs["left"], pairs["right"]]).value_counts()
    y = lam.reindex(ids).to_numpy()
    ok = np.isfinite(y)
    log(f"{ok.sum():,}/{len(ids):,} imágenes piloto con puntuación BT; votos por imagen mediana "
        f"{nvotes.reindex(ids).median():.0f}")
    # sanidad: CLS de este pase = embedding DINOv2 del benchmark
    from exp026_additive_context import load_images
    man_all, Z = load_images("dinov2_vit_b14")
    pos = {lid: i for i, lid in enumerate(man_all["location_id"])}
    smp = [i for i in range(0, len(ids), 97) if ids[i] in pos]
    cls_check = float(np.abs(l2n(F["cls"][smp]) - Z[[pos[ids[i]] for i in smp]]).max())
    log(f"verificación CLS vs embedding del benchmark: max |Δ| = {cls_check:.2e}")
    # solo imágenes con objetivo (las que no tienen votos de seguridad no tienen λ); los prototipos de RQ2 usan todas
    ids_all, F_all = ids, F
    keep = np.flatnonzero(ok)
    F = {k: (v if k.startswith("proto_") else v[keep]) for k, v in F.items()}
    ids, y, man = [ids[i] for i in keep], y[keep], man.iloc[keep].reset_index(drop=True)

    cells = cell_ids(man, CELL_M).to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(np.zeros(len(ids)), groups=cells))
    reps = ["E0_cls", "patch_mean", "cls+patch_mean"] + [f"{k}:{v}" for v in vocab_names()
                                                         for k in ("bovw_l1", "bovw_tfidf", "cls+bovw")]
    preds, alpha_info = {}, {}
    for r in reps:
        preds[r], alpha_info[r] = ridge_oof(r, F, y, folds)
        log(f"  CV espacial {r:<22} ρ = {spearman(y, preds[r]):.4f}  α = {alpha_info[r]}")
    cv = boot_cells(y, preds, cells, "E0_cls", rng)
    cv.to_csv(OUT / "cv_spatial_spearman.csv", index=False, float_format="%.5f")
    city_rows = []
    for c in CITIES:
        m = (man["city"] == c).to_numpy()
        for r in reps:
            city_rows.append(dict(city=c, representation=r, rho=spearman(y[m], preds[r][m])))
    pd.DataFrame(city_rows).to_csv(OUT / "cv_spatial_by_city.csv", index=False, float_format="%.5f")

    # holdout de ciudad: entrena en 2, evalúa en la tercera
    hold = []
    for c in CITIES:
        te = np.flatnonzero((man["city"] == c).to_numpy())
        tr = np.flatnonzero((man["city"] != c).to_numpy())
        hp = {}
        for r in reps:
            p, _ = ridge_oof(r, F, y, [(tr, te)])
            hp[r] = p[te]
        b = boot_cells(y[te], hp, cells[te], "E0_cls", rng)
        b.insert(0, "held_out_city", c)
        hold.append(b)
    hold = pd.concat(hold)
    hold.to_csv(OUT / "city_holdout_spearman.csv", index=False, float_format="%.5f")

    ceiling = split_half_ceiling(pairs, ids, rng)
    log(f"techo split-half: {ceiling}")

    # ocupación y entropía de los histogramas
    occ = []
    vj = json.loads((OUT / "vocab.json").read_text(encoding="utf-8"))
    for vn in vocab_names():
        Hc = F[f"hist_{vn}"]
        P = Hc / Hc.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(P > 0, P * np.log(P), 0), 1)
        occ.append(dict(vocab=vn, bins_used_pilot_pct=100 * float((Hc.sum(0) > 0).mean()),
                        words_per_image_median=float(np.median((Hc > 0).sum(1))),
                        entropy_nats_median=float(np.median(ent)),
                        ari_between_seeds=vj["ari_between_seeds_on_fit_sample"][vn.split("_")[0]]))
    pd.DataFrame(occ).to_csv(OUT / "vocab_occupancy.csv", index=False, float_format="%.4f")

    # RQ2: palabras con mayor |coef| (K = 100, semilla 0; ajuste en todas las piloto, solo para interpretar)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    vn = "K100_s0"
    X = features(f"bovw_l1:{vn}", F, np.arange(len(ids)))
    m = RidgeCV(alphas=ALPHAS).fit(StandardScaler().fit_transform(X), y)
    top = np.argsort(-np.abs(m.coef_))[:10]
    prototype_mosaic(F_all, ids_all, vn, top.tolist(), m.coef_[top].tolist(), OUT / "prototypes_top10_K100_s0.png")
    pd.DataFrame({"word": top, "coef_std": m.coef_[top],
                  "mean_freq": X[:, top].mean(0), "semantic_label": "candidato (sin inspección)"}
                 ).to_csv(OUT / "top_words_K100_s0.csv", index=False, float_format="%.5f")

    # criterios registrados
    e0 = cv.set_index("representation").loc["E0_cls"]
    best = {}
    for k in ("bovw_l1", "bovw_tfidf", "cls+bovw"):
        sub = cv[cv["representation"].str.startswith(k + ":")]
        best[k] = sub.sort_values("rho", ascending=False).iloc[0].to_dict()
    pm = cv.set_index("representation").loc["cls+patch_mean"]
    summary = {
        "n_images": int(len(ids_all)), "n_with_target": int(ok.sum()), "cls_check_max_abs_diff": cls_check,
        "ceiling": ceiling, "E0_rho": float(e0["rho"]),
        "criterion_adds": {"rep": best["cls+bovw"]["representation"], "delta": best["cls+bovw"]["delta_vs_base"],
                           "ci": [best["cls+bovw"]["delta_ci_low"], best["cls+bovw"]["delta_ci_high"]],
                           "met": bool(best["cls+bovw"]["delta_ci_low"] > 0),
                           "note": "mejor K/semilla de cls+bovw: elegido post-hoc entre 6; ver tabla completa"},
        "criterion_interpretable_competitive": {
            "rep": max((best["bovw_l1"], best["bovw_tfidf"]), key=lambda r: r["rho"])["representation"],
            "ratio_vs_E0": max(best["bovw_l1"]["rho"], best["bovw_tfidf"]["rho"]) / float(e0["rho"]),
            "met": bool(max(best["bovw_l1"]["rho"], best["bovw_tfidf"]["rho"]) >= 0.9 * float(e0["rho"]))},
        "control_pooling": {"cls+patch_mean_delta": float(pm["delta_vs_base"]),
                            "ci": [float(pm["delta_ci_low"]), float(pm["delta_ci_high"])]},
        "ridge_alphas": alpha_info, "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(json.dumps(summary["criterion_adds"], default=float))
    log(json.dumps(summary["criterion_interpretable_competitive"], default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vocab", "extract", "analyze", "all"], default="all")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.threads, args.batch_size)
        return
    if args.stage in ("vocab", "all") and not (OUT / "vocab.json").exists():
        stage_vocab(args.threads, args.batch_size)
    if args.stage in ("extract", "all"):
        stage_extract(args.threads, args.batch_size)
    if args.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
