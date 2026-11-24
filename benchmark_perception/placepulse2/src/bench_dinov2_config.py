"""
Microbenchmark de configuracion para la extraccion completa de DINOv2.

Pregunta: con que batch size y cuantos hilos se extrae mas rapido en esta CPU
(i5-11260H, 6 nucleos / 12 hilos), sin cambiar el resultado numerico.

El batch size y el numero de hilos NO alteran la representacion (mismo modelo,
mismo preprocessing, mismo fp32, torch.no_grad): solo cambian como se agenda el
computo. Por eso se pueden ajustar sin romper la comparacion justa entre
encoders. Se verifica explicitamente que los vectores coinciden entre configs.

Uso:
    python bench_dinov2_config.py --n 192
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from encoders import embed, load_dinov2
from utils import MANIFESTS, OUTPUTS, ZIP_PATH, log, set_seed


def run_config(paths, batch_size: int, threads: int) -> tuple[float, np.ndarray]:
    torch.set_num_threads(threads)
    set_seed(42)
    spec = load_dinov2(device="cpu")
    out = []
    # calentamiento: la primera pasada paga la construccion de los kernels MKL
    with zipfile.ZipFile(ZIP_PATH) as zf:
        warm = paths[:batch_size]
        tens = []
        for p in warm:
            with zf.open(p) as fh:
                with Image.open(io.BytesIO(fh.read())) as im:
                    tens.append(spec.transform(im.convert("RGB")))
        embed(spec, torch.stack(tens))

        t0 = time.perf_counter()
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            tens = []
            for p in chunk:
                with zf.open(p) as fh:
                    with Image.open(io.BytesIO(fh.read())) as im:
                        tens.append(spec.transform(im.convert("RGB")))
            z = embed(spec, torch.stack(tens))
            out.append(z.cpu().numpy().astype(np.float32))
        el = time.perf_counter() - t0
    return len(paths) / el, np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    args = ap.parse_args()

    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"].reset_index(drop=True)
    # se toman del final del manifiesto: son imagenes que NO estan en el
    # subconjunto smoke ni en el shard abortado, asi el benchmark no se
    # beneficia de ninguna cache de disco previa
    paths = man["image_path"].tolist()[-args.n:]

    results = []
    ref = None
    for threads in (6, 12):
        for bs in (16, 32, 64):
            rate, X = run_config(paths, bs, threads)
            if ref is None:
                ref = X
                dif = 0.0
            else:
                dif = float(np.abs(X - ref).max())
            results.append({"threads": threads, "batch_size": bs,
                            "img_per_s": round(rate, 2),
                            "eta_84297_h": round(84297 / rate / 3600, 2),
                            "max_abs_dif_vs_ref": dif})
            log(f"threads={threads:>2} batch={bs:>2}  {rate:5.2f} img/s  "
                f"ETA {84297/rate/3600:4.2f} h  max|dif|={dif:.2e}")

    best = max(results, key=lambda r: r["img_per_s"])
    log(f"\nMEJOR: threads={best['threads']} batch={best['batch_size']} "
        f"-> {best['img_per_s']} img/s, ETA {best['eta_84297_h']} h")

    payload = {
        "n_images_benchmark": args.n,
        "note": "batch size y threads no alteran el resultado numerico; "
                "max_abs_dif_vs_ref debe ser 0 para todas las configuraciones",
        "results": results,
        "best": best,
    }
    p = OUTPUTS / "dinov2_config_benchmark.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"-> {p}")


if __name__ == "__main__":
    main()
