"""
Extracción de embeddings CLIP ViT-B/32 (open_clip, laion2b_s34b_b79k) para las 110,987 imágenes de
Place Pulse 2.0. Insumo de EXP-031 (conceptos interpretables sin anotadores: concept bottleneck
"label-free" con prompts de texto del codebook).

Mismo patrón que extract_dinov2_full.py (M03): un SUBPROCESO por bloque (el heap se recicla), ZIP
abierto y cerrado dentro del bloque, escritura atómica, reanudable. Embeddings L2-normalizados.

Uso:
    python extract_clip.py              # lanza / reanuda y consolida
    python extract_clip.py --status
Salida: outputs/features/clip_vit_b32/{features.npy, index.json}  (no versionado)
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

HERE = Path(__file__).parent
OUT = FEATURES / "clip_vit_b32"
STAGE = OUT / "blocks"
BLOCK = 8192
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def all_ids() -> list[str]:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    return man["location_id"].tolist()


def worker(block: int, threads: int, batch_size: int) -> None:
    import open_clip
    import torch
    from PIL import Image

    torch.set_num_threads(threads)
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    ids = man["location_id"].tolist()[block * BLOCK:(block + 1) * BLOCK]
    path_of = dict(zip(man["location_id"], man["image_path"]))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    vecs, t0 = [], time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH) as zf, torch.no_grad():
        for i in range(0, len(ids), batch_size):
            tens = []
            for lid in ids[i:i + batch_size]:
                with zf.open(path_of[lid]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    tens.append(preprocess(im.convert("RGB")))
            z = model.encode_image(torch.stack(tens)).float()
            vecs.append(torch.nn.functional.normalize(z, dim=-1).numpy())
            if (i // batch_size) % 50 == 0:
                gc.collect()
    X = np.concatenate(vecs).astype(np.float32)
    assert X.shape[0] == len(ids) and np.isfinite(X).all()
    tmp = STAGE / f"block_{block:03d}.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, STAGE / f"block_{block:03d}.npy")
    (STAGE / f"block_{block:03d}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"bloque {block}: {X.shape} en {(time.perf_counter() - t0) / 60:.1f} min "
        f"({len(ids) / (time.perf_counter() - t0):.1f} img/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.threads, args.batch_size)
        return
    STAGE.mkdir(parents=True, exist_ok=True)
    ids = all_ids()
    n_blocks = (len(ids) + BLOCK - 1) // BLOCK
    done = sorted(STAGE.glob("block_*[0-9].npy"))
    log(f"{len(ids):,} imágenes, {n_blocks} bloques, {len(done)} hechos")
    if args.status:
        return
    for b in range(n_blocks):
        if (STAGE / f"block_{b:03d}.npy").exists():
            continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(b),
                        "--threads", str(args.threads), "--batch-size", str(args.batch_size)], check=True)
    X = np.concatenate([np.load(STAGE / f"block_{b:03d}.npy") for b in range(n_blocks)])
    got = sum((json.loads((STAGE / f"block_{b:03d}.ids.json").read_text()) for b in range(n_blocks)), [])
    assert got == ids and X.shape[0] == len(ids)
    np.save(OUT / "features.npy", X)
    (OUT / "index.json").write_text(json.dumps({
        "encoder": f"open_clip {MODEL} {PRETRAINED}", "embed_dim": int(X.shape[1]),
        "n_images": len(ids), "l2_normalized": True, "location_ids": ids}), encoding="utf-8")
    log(f"consolidado {X.shape} -> {OUT}")


if __name__ == "__main__":
    main()
