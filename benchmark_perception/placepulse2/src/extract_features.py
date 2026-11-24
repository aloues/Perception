"""
Extraccion de embeddings congelados para Place Pulse 2.0.

Las imagenes se leen directamente del ZIP (no se extraen a disco: son 2.6 GB).
El encoder permanece SIEMPRE congelado (torch.no_grad + eval); este script no
entrena nada.

Guarda de forma incremental en shards, para poder reanudar una corrida larga:
    outputs/features/<encoder>/shard_00000.npy
    outputs/features/<encoder>/index.json

Uso:
    python extract_features.py --encoder ours --batch-size 64
    python extract_features.py --encoder ours --limit 200      # prueba corta
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from encoders import embed, load_dinov2, load_imagenet_vit, load_ours
from utils import CKPT_DIR, FEATURES, MANIFESTS, ZIP_PATH, environment_info, log, set_seed

SHARD_SIZE = 8192


def build_encoder(name: str, ckpt: str | None, device: str):
    if name == "ours":
        if ckpt is None:
            ckpt = str(CKPT_DIR / "simclr_perception_224x_224_best_epoch_50.pt")
        return load_ours(ckpt, device=device)
    if name == "imagenet":
        return load_imagenet_vit(device=device)
    return load_dinov2(device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ours", choices=["ours", "imagenet", "dinov2"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="solo las primeras N imagenes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_encoder(args.encoder, args.ckpt, device)
    log(f"encoder={spec.name} device={device} image_size={spec.image_size}")

    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man["image_in_zip"] = man["image_in_zip"].map({"True": True, "False": False})
    man = man[man["image_in_zip"]].reset_index(drop=True)
    if args.limit:
        man = man.head(args.limit)

    loc_ids = man["location_id"].tolist()
    paths = man["image_path"].tolist()
    log(f"imagenes a procesar: {len(loc_ids):,}")

    out_dir = FEATURES / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    done_shards = sorted(out_dir.glob("shard_*.npy"))
    start = len(done_shards) * SHARD_SIZE
    if start:
        log(f"reanudando: {len(done_shards)} shards ya presentes ({start:,} imagenes)")

    t0 = time.perf_counter()
    buf: list[np.ndarray] = []
    shard_idx = len(done_shards)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i in range(start, len(loc_ids), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            tensors = []
            for p in chunk:
                with zf.open(p) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                tensors.append(spec.transform(img))

            z = embed(spec, torch.stack(tensors).to(device))
            buf.append(z.cpu().numpy().astype(np.float32))

            n_buf = sum(b.shape[0] for b in buf)
            if n_buf >= SHARD_SIZE:
                arr = np.concatenate(buf, axis=0)
                np.save(out_dir / f"shard_{shard_idx:05d}.npy", arr[:SHARD_SIZE])
                shard_idx += 1
                buf = [arr[SHARD_SIZE:]] if arr.shape[0] > SHARD_SIZE else []

            done = i + len(chunk) - start
            if done and (i // args.batch_size) % 20 == 0:
                rate = done / (time.perf_counter() - t0)
                eta = (len(loc_ids) - (i + len(chunk))) / max(rate, 1e-6) / 3600
                log(f"  {i + len(chunk):,}/{len(loc_ids):,}  {rate:.1f} img/s  ETA {eta:.2f} h")

    if buf:
        arr = np.concatenate(buf, axis=0)
        if arr.shape[0]:
            np.save(out_dir / f"shard_{shard_idx:05d}.npy", arr)

    index = {
        "encoder": spec.name,
        "embed_dim": spec.embed_dim,
        "image_size": spec.image_size,
        "n_images": len(loc_ids),
        "shard_size": SHARD_SIZE,
        "location_ids": loc_ids,
        "l2_normalized": True,
        "frozen_backbone": True,
        "environment": environment_info(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    log(f"\nGuardado en {out_dir}  ({index['elapsed_s']}s)")


if __name__ == "__main__":
    main()
