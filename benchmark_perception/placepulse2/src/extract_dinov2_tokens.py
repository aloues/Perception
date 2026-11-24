"""
Re-extracción de DINOv2 ViT-B/14 (224 px, mismo modelo y transformación que encoders.load_dinov2) guardando lo
que el protocolo de linear probing del paper de DINOv2 usa y nosotros no: además del token CLS final, el
PROMEDIO DE LOS TOKENS DE PARCHE y el CLS de las capas intermedias (Oquab et al. 2023, protocolo de linear
probing: "how many output layers we use, whether we concatenate the average-pooled patch token features with the
class token"). Insumo de EXP-035 (representación v2).

Por imagen se guardan 5 vectores de 768 (sin normalizar): [cls_L12, avg_L12, cls_L9, cls_L10, cls_L11], todos
después de la LayerNorm final del modelo. cls_L12 normalizado es idéntico al embedding actual (se verifica).

Mismo patrón que extract_clip.py / extract_dinov2_full.py (M03): subproceso por bloque, escritura atómica,
reanudable.

Uso:  python extract_dinov2_tokens.py         Salida: outputs/features/dinov2_tokens/{features.npy, index.json}
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

from utils import FEATURES, MANIFESTS, ZIP_PATH, log

OUT = FEATURES / "dinov2_tokens"
STAGE = OUT / "blocks"
BLOCK = 8192
PARTS = ["cls_L12", "avg_L12", "cls_L9", "cls_L10", "cls_L11"]


def ids_all() -> tuple[list[str], dict]:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    return man["location_id"].tolist(), dict(zip(man["location_id"], man["image_path"]))


def worker(block: int, threads: int, batch_size: int) -> None:
    import torch
    from PIL import Image

    from encoders import load_dinov2

    torch.set_num_threads(threads)
    ids, path_of = ids_all()
    ids = ids[block * BLOCK:(block + 1) * BLOCK]
    spec = load_dinov2("cpu")
    m = spec.model
    grabbed = {}
    hooks = [m.blocks[i].register_forward_hook(lambda mod, inp, out, i=i: grabbed.__setitem__(i, out))
             for i in (8, 9, 10)]
    vecs, t0 = [], time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH) as zf, torch.no_grad():
        for a in range(0, len(ids), batch_size):
            tens = []
            for lid in ids[a:a + batch_size]:
                with zf.open(path_of[lid]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    tens.append(spec.transform(im.convert("RGB")))
            f = m.forward_features(torch.stack(tens))
            out = [f[:, 0], f[:, m.num_prefix_tokens:].mean(1)] + [m.norm(grabbed[i])[:, 0] for i in (8, 9, 10)]
            vecs.append(torch.stack(out, 1).float().numpy())
            if (a // batch_size) % 50 == 0:
                gc.collect()
    for h in hooks:
        h.remove()
    X = np.concatenate(vecs).astype(np.float32)
    assert X.shape == (len(ids), 5, 768) and np.isfinite(X).all()
    tmp = STAGE / f"block_{block:03d}.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, STAGE / f"block_{block:03d}.npy")
    (STAGE / f"block_{block:03d}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"bloque {block}: {X.shape} en {(time.perf_counter() - t0) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.threads, args.batch_size)
        return
    STAGE.mkdir(parents=True, exist_ok=True)
    ids, _ = ids_all()
    nb = (len(ids) + BLOCK - 1) // BLOCK
    log(f"{len(ids):,} imágenes, {nb} bloques, {len(list(STAGE.glob('block_*[0-9].npy')))} hechos")
    for b in range(nb):
        if not (STAGE / f"block_{b:03d}.npy").exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(b),
                            "--threads", str(args.threads), "--batch-size", str(args.batch_size)], check=True)
    X = np.concatenate([np.load(STAGE / f"block_{b:03d}.npy") for b in range(nb)])
    got = sum((json.loads((STAGE / f"block_{b:03d}.ids.json").read_text()) for b in range(nb)), [])
    assert got == ids
    # verificación: cls_L12 normalizado == embedding DINOv2 actual
    ref_dir = FEATURES / "dinov2_vit_b14"
    ref_ids = json.loads((ref_dir / "index.json").read_text(encoding="utf-8"))["location_ids"]
    ref = np.concatenate([np.load(s) for s in sorted(ref_dir.glob("shard_*.npy"))])
    pos = {lid: i for i, lid in enumerate(ref_ids[: ref.shape[0]])}
    sample = [i for i in range(0, len(ids), 997) if ids[i] in pos]
    a = X[sample, 0] / np.linalg.norm(X[sample, 0], axis=1, keepdims=True)
    diff = float(np.abs(a - ref[[pos[ids[i]] for i in sample]]).max())
    log(f"verificación cls_L12 vs embedding actual: max |Δ| = {diff:.2e} en {len(sample)} imágenes")
    np.save(OUT / "features.npy", X)
    (OUT / "index.json").write_text(json.dumps({"encoder": "dinov2_vit_b14 timm lvd142m @224", "parts": PARTS,
                                                "shape": list(X.shape), "verification_max_abs_diff": diff,
                                                "location_ids": ids}), encoding="utf-8")
    log(f"consolidado {X.shape} -> {OUT}")


if __name__ == "__main__":
    main()
