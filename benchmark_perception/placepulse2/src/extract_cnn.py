"""
Extracción de embeddings de CNN (timm) para las 110,987 imágenes de Place Pulse 2.0. Insumo de EXP-054
(asesoría 05/11: "puede ser dino, puede ser una CNN, ¿cuál le atina mejor?").

Encoders (congelados, ImageNet-1k, pooling global, sin cabeza):
    resnet50   -> timm "resnet50.tv2_in1k"       (2048-d; la familia de CNN de la literatura de Place Pulse)
    convnext_t -> timm "convnext_tiny.fb_in1k"   (768-d; CNN moderna, coste comparable)

Mismo protocolo que encoders.py: 224×224 por Resize (sin recorte), normalización ImageNet, salida L2.
Mismo patrón que extract_clip.py (M03): un SUBPROCESO por bloque, ZIP abierto y cerrado dentro del bloque,
escritura atómica, reanudable.

Uso:
    python extract_cnn.py --model resnet50          # lanza / reanuda y consolida
    python extract_cnn.py --model resnet50 --status
Salida: outputs/features/<resnet50_in1k|convnext_tiny_in1k>/{features.npy, index.json}  (no versionado)
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

BLOCK = 8192
MODELS = {
    "resnet50": ("resnet50.tv2_in1k", "resnet50_in1k"),
    "convnext_t": ("convnext_tiny.fb_in1k", "convnext_tiny_in1k"),
}


def all_ids() -> list[str]:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    return man["location_id"].tolist()


def worker(model_key: str, block: int, threads: int, batch_size: int) -> None:
    import timm
    import torch
    from PIL import Image

    from encoders import _transform

    timm_name, out_name = MODELS[model_key]
    stage = FEATURES / out_name / "blocks"
    torch.set_num_threads(threads)
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    ids = man["location_id"].tolist()[block * BLOCK:(block + 1) * BLOCK]
    path_of = dict(zip(man["location_id"], man["image_path"]))
    model = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()
    tf = _transform(224)
    vecs, t0 = [], time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH) as zf, torch.no_grad():
        for i in range(0, len(ids), batch_size):
            tens = []
            for lid in ids[i:i + batch_size]:
                with zf.open(path_of[lid]) as fh, Image.open(io.BytesIO(fh.read())) as im:
                    tens.append(tf(im.convert("RGB")))
            z = model(torch.stack(tens)).float()
            vecs.append(torch.nn.functional.normalize(z, dim=-1).numpy())
            if (i // batch_size) % 50 == 0:
                gc.collect()
    X = np.concatenate(vecs).astype(np.float32)
    assert X.shape[0] == len(ids) and np.isfinite(X).all()
    tmp = stage / f"block_{block:03d}.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, stage / f"block_{block:03d}.npy")
    (stage / f"block_{block:03d}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    log(f"{model_key} bloque {block}: {X.shape} en {(time.perf_counter() - t0) / 60:.1f} min "
        f"({len(ids) / (time.perf_counter() - t0):.1f} img/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.model, args.worker, args.threads, args.batch_size)
        return
    timm_name, out_name = MODELS[args.model]
    out = FEATURES / out_name
    stage = out / "blocks"
    stage.mkdir(parents=True, exist_ok=True)
    ids = all_ids()
    n_blocks = (len(ids) + BLOCK - 1) // BLOCK
    done = sorted(stage.glob("block_*[0-9].npy"))
    log(f"{args.model}: {len(ids):,} imágenes, {n_blocks} bloques, {len(done)} hechos")
    if args.status:
        return
    for b in range(n_blocks):
        if (stage / f"block_{b:03d}.npy").exists():
            continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                        "--worker", str(b), "--threads", str(args.threads),
                        "--batch-size", str(args.batch_size)], check=True)
    X = np.concatenate([np.load(stage / f"block_{b:03d}.npy") for b in range(n_blocks)])
    got = sum((json.loads((stage / f"block_{b:03d}.ids.json").read_text()) for b in range(n_blocks)), [])
    assert got == ids and X.shape[0] == len(ids)
    np.save(out / "features.npy", X)
    (out / "index.json").write_text(json.dumps({
        "encoder": f"timm {timm_name}", "embed_dim": int(X.shape[1]),
        "n_images": len(ids), "l2_normalized": True, "location_ids": ids}), encoding="utf-8")
    log(f"consolidado {X.shape} -> {out}")


if __name__ == "__main__":
    main()
