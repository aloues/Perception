"""
Extraccion de embeddings congelados SOLO para el subconjunto del smoke test.

Diferencias respecto a extract_features.py (que se mantiene intacto):

  1. procesa unicamente los location_id de data/smoke/subset_images.txt;
  2. REUTILIZA los embeddings ya calculados de la corrida completa: para cada
     imagen del subconjunto que ya tenga vector en outputs/features/<enc>/ se
     copia el vector en lugar de recomputarlo (la corrida de DINOv2 alcanzo a
     completar shard_00000 = 8,192 imagenes antes de caerse);
  3. guarda por CHUNKS pequenos con reanudacion, porque la corrida completa de
     DINOv2 murio con 0xC0000374 (corrupcion de heap) tras ~5 h de proceso;
     cada chunk es un fichero independiente, asi que una caida cuesta como
     maximo un chunk.

Salidas:
    outputs/features_smoke/<encoder>/chunk_XXXXX.npy
    outputs/features_smoke/<encoder>/ids_XXXXX.json
    outputs/features_smoke/<encoder>/index.json     (al terminar)

Uso:
    python extract_features_subset.py --encoder dinov2 --batch-size 16
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
from extract_features import build_encoder
from utils import FEATURES, MANIFESTS, OUTPUTS, PKG_ROOT, ZIP_PATH, environment_info, log, set_seed

FEATURES_SMOKE = OUTPUTS / "features_smoke"
SUBSET_FILE = PKG_ROOT / "data" / "smoke" / "subset_images.txt"
CHUNK = 1024


def reusable_from_full(enc_name: str, wanted: list[str]) -> dict[str, np.ndarray]:
    """Vectores ya calculados en la corrida completa (aunque este incompleta)."""
    d = FEATURES / enc_name
    shards = sorted(d.glob("shard_*.npy"))
    if not shards:
        return {}
    idx_path = d / "index.json"
    if idx_path.exists():
        ids_all = json.loads(idx_path.read_text(encoding="utf-8"))["location_ids"]
    else:
        # index.json no se escribio (corrida abortada): el orden es el del
        # manifiesto filtrado por image_in_zip, igual que en extract_features.py
        man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
        man = man[man["image_in_zip"] == "True"].reset_index(drop=True)
        ids_all = man["location_id"].tolist()
    X = np.concatenate([np.load(s) for s in shards], axis=0)
    ids_done = ids_all[: X.shape[0]]
    pos = {lid: i for i, lid in enumerate(ids_done)}
    out = {lid: X[pos[lid]] for lid in wanted if lid in pos}
    log(f"reutilizados de la corrida completa: {len(out):,} vectores")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2", choices=["ours", "imagenet", "dinov2"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="corta la extraccion al agotar el presupuesto y deja "
                         "los chunks para reanudar")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subset = SUBSET_FILE.read_text(encoding="utf-8").split()
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    path_of = dict(zip(man["location_id"], man["image_path"]))
    subset = [s for s in subset if s in path_of]
    log(f"subconjunto: {len(subset):,} imagenes")

    spec = build_encoder(args.encoder, args.ckpt, device)
    out_dir = FEATURES_SMOKE / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"encoder={spec.name} device={device} image_size={spec.image_size}")

    reuse = reusable_from_full(spec.name, subset)
    todo = [s for s in subset if s not in reuse]

    # chunks ya escritos (reanudacion)
    done_ids: set[str] = set()
    for f in sorted(out_dir.glob("ids_*.json")):
        done_ids.update(json.loads(f.read_text(encoding="utf-8")))
    if done_ids:
        log(f"reanudando: {len(done_ids):,} imagenes ya en chunks")
    todo = [s for s in todo if s not in done_ids]
    log(f"a calcular ahora: {len(todo):,}")

    t0 = time.perf_counter()
    next_chunk = len(list(out_dir.glob("chunk_*.npy")))
    aborted = False

    if todo:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            for c0 in range(0, len(todo), CHUNK):
                ids = todo[c0:c0 + CHUNK]
                vecs = []
                for i in range(0, len(ids), args.batch_size):
                    batch = ids[i:i + args.batch_size]
                    tensors = []
                    for lid in batch:
                        with zf.open(path_of[lid]) as fh:
                            img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                        tensors.append(spec.transform(img))
                    z = embed(spec, torch.stack(tensors).to(device))
                    vecs.append(z.cpu().numpy().astype(np.float32))
                arr = np.concatenate(vecs, axis=0)
                np.save(out_dir / f"chunk_{next_chunk:05d}.npy", arr)
                (out_dir / f"ids_{next_chunk:05d}.json").write_text(
                    json.dumps(ids), encoding="utf-8")
                next_chunk += 1

                done = c0 + len(ids)
                el = time.perf_counter() - t0
                rate = done / el
                log(f"  {done:,}/{len(todo):,}  {rate:.1f} img/s  "
                    f"ETA {(len(todo)-done)/max(rate,1e-6)/60:.1f} min")
                if args.max_minutes and el / 60 >= args.max_minutes:
                    log(f"  presupuesto de {args.max_minutes} min agotado: se corta aqui")
                    aborted = True
                    break

    # ── consolidar ────────────────────────────────────────────────────────
    vec_of: dict[str, np.ndarray] = dict(reuse)
    for f in sorted(out_dir.glob("ids_*.json")):
        ids = json.loads(f.read_text(encoding="utf-8"))
        arr = np.load(out_dir / ("chunk_" + f.stem.split("_")[1] + ".npy"))
        for j, lid in enumerate(ids):
            vec_of[lid] = arr[j]

    have = [s for s in subset if s in vec_of]
    X = np.stack([vec_of[s] for s in have]).astype(np.float32)
    np.save(out_dir / "features.npy", X)
    index = {
        "encoder": spec.name,
        "embed_dim": int(X.shape[1]),
        "image_size": spec.image_size,
        "n_images": len(have),
        "n_subset": len(subset),
        "complete": len(have) == len(subset),
        "location_ids": have,
        "l2_normalized": True,
        "frozen_backbone": True,
        "reused_from_full_run": len(reuse),
        "computed_now": len(have) - len(reuse),
        "aborted_on_time_budget": aborted,
        "environment": environment_info(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    log(f"\nGuardado {X.shape} en {out_dir}  (completo={index['complete']})")


if __name__ == "__main__":
    main()
