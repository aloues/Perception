"""
Worker de extraccion DINOv2: procesa UN bloque de ids y termina.

Se ejecuta como subproceso independiente (ver extract_dinov2_full.py). Que el
proceso muera al acabar cada bloque es deliberado: es la unica forma robusta de
garantizar que el heap se devuelve entero al sistema operativo entre bloques.

Diagnostico del fallo anterior (corrida del 2026-09-23):
  - la corrida completa duro 2d 6h de reloj para 14,752 imagenes y murio con
    codigo 1073807364 = 0xC0000374 = STATUS_HEAP_CORRUPTION;
  - el "desplome" de 7.0 a 0.1 img/s del log es en parte un artefacto: la tasa
    se calculaba como acumulada (done / tiempo_total), asi que cualquier
    suspension del portatil la hunde permanentemente;
  - pero la corrupcion de heap es real y ocurrio tras horas en un unico proceso
    de larga duracion con PIL + zipfile + MKL/OpenMP.

Mitigaciones aplicadas aqui:
  1. un proceso por bloque (~4,096 imagenes, ~10 min): el heap se recicla;
  2. `with Image.open(...)` para cerrar explicitamente cada imagen;
  3. el ZIP se abre y cierra dentro del bloque;
  4. gc.collect() periodico;
  5. escritura atomica (.tmp + rename) para que un corte nunca deje un .npy
     a medias que luego se lea como valido;
  6. tasa INSTANTANEA en el log, no acumulada, para que una suspension no
     falsee el diagnostico.

Uso (lo invoca el orquestador, no a mano):
    python dinov2_worker.py --ids-file <json> --out-prefix <path> --batch-size 32
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from encoders import embed, load_dinov2
from utils import MANIFESTS, ZIP_PATH, log, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    ids = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    path_of = dict(zip(man["location_id"], man["image_path"]))

    spec = load_dinov2(device="cpu")
    log(f"worker: {len(ids):,} imagenes | batch={args.batch_size} "
        f"threads={args.threads} pid={os.getpid()}")

    vecs: list[np.ndarray] = []
    t_block = time.perf_counter()
    t_win, n_win = time.perf_counter(), 0

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i in range(0, len(ids), args.batch_size):
            batch = ids[i:i + args.batch_size]
            tens = []
            for lid in batch:
                with zf.open(path_of[lid]) as fh:
                    data = fh.read()
                with Image.open(io.BytesIO(data)) as im:
                    tens.append(spec.transform(im.convert("RGB")))
                del data
            z = embed(spec, torch.stack(tens))
            vecs.append(z.cpu().numpy().astype(np.float32))
            del tens, z

            n_win += len(batch)
            if (i // args.batch_size) % 20 == 0 and n_win:
                # tasa INSTANTANEA de la ventana, no acumulada
                dt = time.perf_counter() - t_win
                log(f"  {i + len(batch):,}/{len(ids):,}  {n_win / dt:.2f} img/s (ventana)")
                t_win, n_win = time.perf_counter(), 0
                gc.collect()

    X = np.concatenate(vecs, axis=0).astype(np.float32)
    assert X.shape == (len(ids), spec.embed_dim), f"forma inesperada {X.shape}"
    assert np.isfinite(X).all(), "hay NaN/Inf en los embeddings del bloque"
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"embeddings no L2-normalizados: {norms.min()}..{norms.max()}"

    # Escritura atomica. OJO: np.save anade ".npy" si el nombre no acaba en
    # ".npy", asi que el temporal DEBE acabar en ".npy" o el rename falla.
    out = Path(args.out_prefix)
    final_npy = out.parent / f"{out.name}.npy"
    tmp_npy = out.parent / f"{out.name}.tmp.npy"
    np.save(tmp_npy, X)
    os.replace(tmp_npy, final_npy)
    final_ids = out.parent / f"{out.name}.ids.json"
    tmp_ids = out.parent / f"{out.name}.ids.json.tmp"
    tmp_ids.write_text(json.dumps(ids), encoding="utf-8")
    os.replace(tmp_ids, final_ids)

    el = time.perf_counter() - t_block
    log(f"worker OK: {X.shape} en {el/60:.1f} min ({len(ids)/el:.2f} img/s) -> {final_npy.name}")


if __name__ == "__main__":
    main()
