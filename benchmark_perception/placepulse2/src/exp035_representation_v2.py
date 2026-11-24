"""
EXP-035 — Representación v2: fusión DINOv2⊕CLIP, protocolo de linear probing del paper de DINOv2 (CLS + promedio
de parches + capas intermedias), empates como objetivo 0.5 y regularización/cabeza elegidas en VALIDACIÓN.

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa v2, 2026-10-09).

Uso:
    python exp035_representation_v2.py --phase A     # features disponibles: dinov2 (CLS), clip, fusión
    python exp035_representation_v2.py --phase B     # + tokens DINOv2 (requiere extract_dinov2_tokens.py)
    python exp035_representation_v2.py --final       # entrena lo elegido en val en los 3 protocolos, test 1 vez
Salidas: outputs/exp035/
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest

from evaluate import evaluate_all
from exp026_additive_context import load_images
from probe_utils import load_pairs, load_split
from ranking_head import build_head
from utils import FEATURES, MANIFESTS, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp035"
N_BOOT = 2000
torch.set_num_threads(4)


# ── features ────────────────────────────────────────────────────────────────────────────────
def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def load_feature_sets(phase: str) -> tuple[list[str], dict]:
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].tolist()
    cidx = json.loads((FEATURES / "clip_vit_b32" / "index.json").read_text(encoding="utf-8"))["location_ids"]
    C = np.load(FEATURES / "clip_vit_b32" / "features.npy")
    pos = {lid: i for i, lid in enumerate(cidx)}
    C = C[[pos[i] for i in ids]]
    sets = {"dinov2": Z, "clip": C, "dinov2+clip": np.hstack([l2(Z), l2(C)]).astype(np.float32)}
    if phase in ("B", "final"):
        tdir = FEATURES / "dinov2_tokens"
        if (tdir / "features.npy").exists():
            tid = json.loads((tdir / "index.json").read_text(encoding="utf-8"))["location_ids"]
            T = np.load(tdir / "features.npy", mmap_mode="r")
            tp = {lid: i for i, lid in enumerate(tid)}
            T = np.asarray(T[[tp[i] for i in ids]])
            cls_avg = np.hstack([l2(T[:, 0]), l2(T[:, 1])]).astype(np.float32)
            multi = np.hstack([l2(T[:, k]) for k in range(5)]).astype(np.float32)
            sets.update({"dinov2_cls+avg": cls_avg, "dinov2_multi": multi,
                         "dinov2_cls+avg+clip": np.hstack([cls_avg, l2(C)]).astype(np.float32),
                         "dinov2_multi+clip": np.hstack([multi, l2(C)]).astype(np.float32)})
            del T
        else:
            log("AVISO: tokens DINOv2 aún no disponibles; fase B usa solo las features de la fase A")
    return ids, sets


# ── datos de entrenamiento ──────────────────────────────────────────────────────────────────
def train_frames(pairs: pd.DataFrame, ties: pd.DataFrame, split: dict, use_ties: bool) -> pd.DataFrame:
    tr = pairs.iloc[split["train"]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
    if not use_ties:
        return tr
    imgs = set(tr["left"]) | set(tr["right"])
    t = ties[ties["left"].isin(imgs) & ties["right"].isin(imgs)][["left", "right"]].assign(label=0.5)
    return pd.concat([tr, t], ignore_index=True)


def tensors(df, X, row):
    li = df["left"].map(row).to_numpy()
    ri = df["right"].map(row).to_numpy()
    ok = ~(pd.isna(li) | pd.isna(ri))
    li, ri = li[ok].astype(int), ri[ok].astype(int)
    return (torch.from_numpy(X[li]), torch.from_numpy(X[ri]),
            torch.from_numpy(df["label"].to_numpy()[ok].astype(np.float32)))


def fit(X, row, tr_df, va_df, head="linear", lr=1e-3, wd=1e-4, seed=SEED):
    set_seed(seed)
    zl, zr, y = tensors(tr_df, X, row)
    vl, vr, vy = tensors(va_df, X, row)
    h = build_head(head, embed_dim=X.shape[1])
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    best, state, bad = np.inf, None, 0
    for _ in range(50):
        h.train()
        perm = torch.randperm(len(y))
        for a in range(0, len(y), 4096):
            b = perm[a:a + 4096]
            loss = crit(h(zl[b], zr[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        h.eval()
        with torch.no_grad():
            v = float(crit(h(vl, vr), vy))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in h.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    h.load_state_dict(state)
    h.eval()
    return h


@torch.no_grad()
def logits(h, X, row, df):
    zl, zr, _ = tensors(df, X, row)
    return h(zl, zr).numpy()


def paired(c1, c0, rng):
    n = len(c1)
    bs = np.array([c1[ix].mean() - c0[ix].mean() for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(delta=c1.mean() - c0.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                mcnemar_p=binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


# ── fases ───────────────────────────────────────────────────────────────────────────────────
def grid(phase: str) -> None:
    ids, sets = load_feature_sets(phase)
    row = pd.Series(np.arange(len(ids)), index=ids)
    pairs, ties = load_pairs(), pd.read_csv(MANIFESTS / "pairs_safe_ties.csv", dtype=str)
    sp = load_split("image_disjoint")
    va = pairs.iloc[sp["val"]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
    names = list(sets) if phase == "A" else [k for k in sets if k.startswith("dinov2_")]
    configs = [(f, hd, wd, tie) for f in names for hd in ("linear", "mlp") for wd in (1e-4, 1e-3, 1e-2)
               for tie in (False, True)]
    if phase == "B":   # cabeza/regularización: rejilla reducida alrededor de lo elegido en A
        a = pd.read_csv(OUT / "grid_phaseA.csv").sort_values(["val_acc", "val_logloss"], ascending=[False, True]).iloc[0]
        configs = [(f, a["head"], wd, bool(a["ties"])) for f in names for wd in (1e-4, 1e-3, 1e-2)]
    rows = []
    for k, (f, hd, wd, tie) in enumerate(configs):
        t0 = time.perf_counter()
        h = fit(sets[f], row, train_frames(pairs, ties, sp, tie), va, hd, 1e-3, wd)
        lg = logits(h, sets[f], row, va)
        m = evaluate_all(lg, va["label"].to_numpy())
        rows.append(dict(features=f, dim=sets[f].shape[1], head=hd, wd=wd, ties=tie, val_acc=m["pairwise_accuracy"],
                         val_logloss=m["log_loss"]))
        log(f"  [{k + 1}/{len(configs)}] {f:<22} {hd:<6} wd={wd:<7g} empates={tie!s:<5} val_acc={m['pairwise_accuracy']:.4f} "
            f"ll={m['log_loss']:.4f} ({time.perf_counter() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT / f"grid_phase{phase}.csv", index=False, float_format="%.5f")


def final() -> None:
    rng = np.random.default_rng(SEED)
    g = pd.concat([pd.read_csv(OUT / f"grid_phase{p}.csv") for p in ("A", "B") if (OUT / f"grid_phase{p}.csv").exists()])
    best = g.sort_values(["val_acc", "val_logloss"], ascending=[False, True]).iloc[0]
    log(f"elegido en val: {best.to_dict()}")
    ids, sets = load_feature_sets("final")
    row = pd.Series(np.arange(len(ids)), index=ids)
    man_dir = {"imagenet": "imagenet_vit_b16", "simclr": "ours_simclr_vit_b16"}
    for k, enc in man_dir.items():
        sets[k] = load_images(enc)[1]
    pairs, ties = load_pairs(), pd.read_csv(MANIFESTS / "pairs_safe_ties.csv", dtype=str)
    variants = {
        "baseline_dinov2_fijo": ("dinov2", "linear", 1e-4, False),
        "v2_elegido": (best["features"], best["head"], float(best["wd"]), bool(best["ties"])),
        "v2_dinov2_solo": ("dinov2", best["head"], float(best["wd"]), bool(best["ties"])),
        "imagenet_fijo": ("imagenet", "linear", 1e-4, False),
        "imagenet_v2proto": ("imagenet", best["head"], float(best["wd"]), bool(best["ties"])),
        "simclr_fijo": ("simclr", "linear", 1e-4, False),
        "simclr_v2proto": ("simclr", best["head"], float(best["wd"]), bool(best["ties"])),
    }
    acc_rows, d_rows, preds_keep = [], [], {}
    for proto, test_key in (("image_disjoint", "test"), ("city_holdout", "test_strict"), ("paper_protocol", "test")):
        sp = load_split(proto)
        va = pairs.iloc[sp["val"]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
        te = pairs.iloc[sp[test_key]][["left", "right", "label"]].assign(label=lambda d: d["label"].astype(float))
        corr = {}
        for name, (f, hd, wd, tie) in variants.items():
            t0 = time.perf_counter()
            h = fit(sets[f], row, train_frames(pairs, ties, sp, tie), va, hd, 1e-3, wd)
            lg = logits(h, sets[f], row, te)
            m = evaluate_all(lg, te["label"].to_numpy())
            corr[name] = (lg > 0).astype(int) == te["label"].to_numpy()
            acc_rows.append(dict(protocol=proto, split=test_key, variant=name, features=f, head=hd, wd=wd, ties=tie,
                                 n=m["n"], accuracy=m["pairwise_accuracy"], ci_low=m["ci95_low"], ci_high=m["ci95_high"],
                                 log_loss=m["log_loss"]))
            log(f"  {proto:<14} {name:<22} acc={m['pairwise_accuracy']:.4f} [{m['ci95_low']:.4f},{m['ci95_high']:.4f}] "
                f"({time.perf_counter() - t0:.0f}s)")
        for a, b in (("v2_elegido", "baseline_dinov2_fijo"), ("v2_dinov2_solo", "baseline_dinov2_fijo"),
                     ("v2_elegido", "v2_dinov2_solo"), ("v2_dinov2_solo", "imagenet_v2proto"),
                     ("imagenet_v2proto", "simclr_v2proto")):
            d_rows.append(dict(protocol=proto, comparison=f"{a} - {b}", **paired(corr[a], corr[b], rng)))
            r = d_rows[-1]
            log(f"  {proto:<14} {a} − {b}: Δ={r['delta']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['mcnemar_p']:.2g}")
        pd.DataFrame(acc_rows).to_csv(OUT / "final_accuracy.csv", index=False, float_format="%.5f")
        pd.DataFrame(d_rows).to_csv(OUT / "final_paired.csv", index=False, float_format="%.5f")
    (OUT / "selected_config.json").write_text(json.dumps({k: (v.item() if hasattr(v, "item") else v)
                                                          for k, v in best.to_dict().items()}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], default=None)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase:
        grid(args.phase)
    if args.final:
        final()


if __name__ == "__main__":
    main()
