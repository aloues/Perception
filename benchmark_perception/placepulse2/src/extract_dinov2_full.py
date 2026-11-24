"""
FASE A — Orquestador de la extraccion COMPLETA de DINOv2 sobre Place Pulse 2.0.

Cierra el unico encoder que faltaba para tener el benchmark de tres encoders
sobre las 110,987 imagenes, con el mismo protocolo que OURS e ImageNet.

Diseno (ver dinov2_worker.py para el diagnostico del fallo anterior):

  - REUTILIZA todo embedding DINOv2 valido ya calculado:
      * outputs/features_smoke/dinov2_vit_b14/  (20,000, corrida de esta sesion)
      * outputs/features/dinov2_vit_b14/shard_00000.npy (8,192, corrida abortada)
    Se verifico que los 1,502 ids solapados entre ambas fuentes dan vectores
    BIT-IDENTICOS, lo que confirma que el encoder es determinista entre
    procesos y que ambas fuentes son intercambiables.
  - RESUMABLE: cada bloque es un fichero independiente; relanzar el script
    salta los bloques ya escritos.
  - MEMORY-SAFE: un subproceso por bloque; el heap se recicla entre bloques.
  - Escritura atomica: nunca queda un .npy a medias.
  - Registra inicio, fin, duracion, tasa y errores por bloque en
    outputs/dinov2_full_progress.json

Al terminar consolida en el formato canonico que ya usan los otros dos
encoders (shards de 8,192 + index.json), de modo que train_probe.py funciona
sin cambios.

Uso:
    python extract_dinov2_full.py                 # lanza / reanuda
    python extract_dinov2_full.py --consolidate   # solo consolidar
    python extract_dinov2_full.py --status        # solo informar
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils import FEATURES, MANIFESTS, OUTPUTS, environment_info, log

HERE = Path(__file__).parent
STAGING = OUTPUTS / "features_dinov2_staging"
SMOKE_DIR = OUTPUTS / "features_smoke" / "dinov2_vit_b14"
TARGET = FEATURES / "dinov2_vit_b14"
PROGRESS = OUTPUTS / "dinov2_full_progress.json"
PROVENANCE = TARGET / "provenance"

BLOCK = 4096
SHARD_SIZE = 8192          # identico al de extract_features.py
BATCH_SIZE = 32            # medido como optimo con 6 hilos (bench_dinov2_config.py)
THREADS = 6                # deja 6 hilos logicos libres para el resto del trabajo
EMBED_DIM = 768


def manifest_ids() -> list[str]:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"].reset_index(drop=True)
    return man["location_id"].tolist()


def existing_vectors(all_ids: list[str]) -> dict[str, np.ndarray]:
    """Embeddings DINOv2 ya calculados y validos, de cualquier fuente previa."""
    have: dict[str, np.ndarray] = {}

    shards = sorted(TARGET.glob("shard_*.npy"))
    if shards and not (TARGET / "index.json").exists():
        # corrida abortada: orden = manifiesto filtrado, igual que extract_features.py
        X = np.concatenate([np.load(s) for s in shards], axis=0)
        for lid, v in zip(all_ids[: X.shape[0]], X):
            have[lid] = v
        log(f"  corrida abortada: {X.shape[0]:,} vectores reutilizados")

    idx = SMOKE_DIR / "index.json"
    if idx.exists():
        meta = json.loads(idx.read_text(encoding="utf-8"))
        Xs = np.load(SMOKE_DIR / "features.npy")
        n_new = 0
        for lid, v in zip(meta["location_ids"], Xs):
            if lid not in have:
                n_new += 1
            have[lid] = v
        log(f"  subconjunto smoke: {Xs.shape[0]:,} vectores ({n_new:,} nuevos)")

    for f in sorted(STAGING.glob("block_*.ids.json")):
        npy = f.with_suffix("").with_suffix(".npy")
        if not npy.exists():
            continue
        ids = json.loads(f.read_text(encoding="utf-8"))
        X = np.load(npy)
        if X.shape[0] != len(ids):
            log(f"  AVISO: {npy.name} incoherente ({X.shape[0]} vs {len(ids)}), se ignora")
            continue
        for lid, v in zip(ids, X):
            have[lid] = v
    n_blocks = len(list(STAGING.glob("block_*.npy")))
    if n_blocks:
        log(f"  bloques de esta fase ya escritos: {n_blocks}")
    return have


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"started_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"block": BLOCK, "batch_size": BATCH_SIZE, "threads": THREADS},
            "environment": environment_info(), "blocks": [], "errors": []}


def save_progress(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")


def consolidate(all_ids: list[str]) -> bool:
    """Escribe el formato canonico (shards + index.json) si estan los 110,987."""
    have = existing_vectors(all_ids)
    missing = [i for i in all_ids if i not in have]
    if missing:
        log(f"NO se consolida: faltan {len(missing):,} vectores")
        return False

    # preservar provenance de la corrida abortada antes de reescribir shards
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    for s in sorted(TARGET.glob("shard_*.npy")):
        dst = PROVENANCE / f"aborted_run_{s.name}"
        if not dst.exists():
            shutil.copy2(s, dst)
            log(f"  provenance: copia de {s.name} -> {dst.relative_to(OUTPUTS)}")

    X = np.stack([have[i] for i in all_ids]).astype(np.float32)
    assert X.shape == (len(all_ids), EMBED_DIM)
    assert np.isfinite(X).all()
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)

    for k in range(0, X.shape[0], SHARD_SIZE):
        arr = X[k:k + SHARD_SIZE]
        p = TARGET / f"shard_{k // SHARD_SIZE:05d}.npy"
        tmp = p.with_name(p.stem + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(p)

    prog = load_progress()
    index = {
        "encoder": "dinov2_vit_b14",
        "embed_dim": EMBED_DIM,
        "image_size": 224,
        "n_images": len(all_ids),
        "shard_size": SHARD_SIZE,
        "location_ids": all_ids,
        "l2_normalized": True,
        "frozen_backbone": True,
        "environment": environment_info(),
        "provenance": {
            "note": "consolidado por extract_dinov2_full.py reutilizando embeddings "
                    "validos previos; los 1,502 ids solapados entre la corrida "
                    "abortada y el subconjunto smoke son bit-identicos",
            "sources": ["aborted_run_shard_00000 (8,192)",
                        "features_smoke (20,000)",
                        "bloques de esta fase"],
            "batch_size": BATCH_SIZE, "threads": THREADS,
            "blocks_computed": len(prog.get("blocks", [])),
            "wall_hours_this_phase": round(
                sum(b["seconds"] for b in prog.get("blocks", [])) / 3600, 2),
        },
    }
    tmp = TARGET / "index.json.tmp"
    tmp.write_text(json.dumps(index), encoding="utf-8")
    tmp.replace(TARGET / "index.json")
    log(f"\nCONSOLIDADO: {X.shape} -> {TARGET}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="corta tras este presupuesto; los bloques hechos se conservan")
    args = ap.parse_args()

    STAGING.mkdir(parents=True, exist_ok=True)
    TARGET.mkdir(parents=True, exist_ok=True)
    all_ids = manifest_ids()

    log(f"objetivo: {len(all_ids):,} imagenes")
    have = existing_vectors(all_ids)
    todo = [i for i in all_ids if i not in have]
    log(f"ya calculados: {len(have):,} | faltan: {len(todo):,}")

    if args.status:
        return
    if args.consolidate or not todo:
        consolidate(all_ids)
        return

    eta_h = len(todo) / 6.31 / 3600
    log(f"ETA estimada: {eta_h:.2f} h a 6.31 img/s ({len(todo) // BLOCK + 1} bloques de {BLOCK})")

    prog = load_progress()
    t_start = time.perf_counter()
    next_id = len(list(STAGING.glob("block_*.npy")))

    for b0 in range(0, len(todo), BLOCK):
        ids = todo[b0:b0 + BLOCK]
        prefix = STAGING / f"block_{next_id:05d}"
        ids_file = STAGING / f"block_{next_id:05d}.work.json"
        ids_file.write_text(json.dumps(ids), encoding="utf-8")

        ok = False
        for attempt in (1, 2, 3):
            log(f"\n=== bloque {next_id} ({len(ids):,} img) intento {attempt}/3 ===")
            t0 = time.perf_counter()
            r = subprocess.run(
                [sys.executable, str(HERE / "dinov2_worker.py"),
                 "--ids-file", str(ids_file), "--out-prefix", str(prefix),
                 "--batch-size", str(BATCH_SIZE), "--threads", str(THREADS)],
                cwd=HERE)
            dt = time.perf_counter() - t0
            if r.returncode == 0 and prefix.with_suffix(".npy").exists():
                prog["blocks"].append({
                    "block": next_id, "n": len(ids), "seconds": round(dt, 1),
                    "img_per_s": round(len(ids) / dt, 2),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "attempts": attempt})
                ok = True
                break
            prog["errors"].append({
                "block": next_id, "attempt": attempt, "returncode": r.returncode,
                "seconds": round(dt, 1),
                "utc": datetime.now(timezone.utc).isoformat()})
            log(f"bloque {next_id}: fallo codigo {r.returncode}, reintentando")
            save_progress(prog)
            time.sleep(20)

        save_progress(prog)
        if not ok:
            log(f"bloque {next_id}: FALLO tras 3 intentos; se conserva el progreso "
                f"y se continua con el siguiente bloque")
        else:
            ids_file.unlink(missing_ok=True)
        next_id += 1

        done_h = (time.perf_counter() - t_start) / 3600
        if args.max_hours and done_h >= args.max_hours:
            log(f"presupuesto de {args.max_hours} h agotado; los bloques hechos se conservan")
            break

    log(f"\nfase de computo terminada en {(time.perf_counter() - t_start)/3600:.2f} h")
    consolidate(all_ids)


if __name__ == "__main__":
    main()
