"""
Estimacion de tiempo de extraccion de features ANTES de lanzar una corrida larga.

Mide el pipeline completo (leer del ZIP -> decodificar -> transform -> forward)
con 100 y 1,000 imagenes, y extrapola al dataset completo.

Las imagenes se leen directamente del ZIP, sin extraerlas a disco.

Uso:
    python benchmark_throughput.py --encoder ours --batch-size 32
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from encoders import load_ours, load_imagenet_vit, load_dinov2, embed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ZIP_PATH = REPO_ROOT / "external_assets" / "placepulse2" / "raw" / "place-pulse-2.0.zip"
CKPT = REPO_ROOT / "external_assets" / "checkpoints" / "simclr_perception_224x_224_best_epoch_50.pt"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

N_TOTAL_IMAGES = 110_988


def image_names(zf: zipfile.ZipFile, limit: int) -> list[str]:
    names = []
    for n in zf.namelist():
        if n.startswith("images/") and n.lower().endswith(".jpg"):
            names.append(n)
            if len(names) >= limit:
                break
    return names


def run_trial(spec, zf: zipfile.ZipFile, names: list[str], batch_size: int) -> dict:
    t0 = time.perf_counter()
    n_done = 0
    t_io = 0.0
    t_fwd = 0.0

    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]

        t_a = time.perf_counter()
        tensors = []
        for name in chunk:
            with zf.open(name) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            tensors.append(spec.transform(img))
        batch = torch.stack(tensors)
        t_io += time.perf_counter() - t_a

        t_b = time.perf_counter()
        z = embed(spec, batch)
        t_fwd += time.perf_counter() - t_b

        n_done += len(chunk)
        assert z.shape[0] == len(chunk)

    elapsed = time.perf_counter() - t0
    return {
        "n_images": n_done,
        "elapsed_s": round(elapsed, 3),
        "img_per_s": round(n_done / elapsed, 2),
        "io_decode_s": round(t_io, 3),
        "forward_s": round(t_fwd, 3),
        "io_fraction": round(t_io / elapsed, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ours", choices=["ours", "imagenet", "dinov2"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}  threads={torch.get_num_threads()}")

    if args.encoder == "ours":
        spec = load_ours(str(CKPT), device=device)
    elif args.encoder == "imagenet":
        spec = load_imagenet_vit(device=device)
    else:
        spec = load_dinov2(device=device)
    print(f"encoder={spec.name}  image_size={spec.image_size}")

    results = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names_all = image_names(zf, max(args.sizes))

        # Calentamiento, para no contaminar la primera medicion.
        run_trial(spec, zf, names_all[:8], args.batch_size)

        for n in args.sizes:
            print(f"\nMidiendo con {n} imagenes...")
            r = run_trial(spec, zf, names_all[:n], args.batch_size)
            results[str(n)] = r
            print(f"  {r['elapsed_s']}s  ->  {r['img_per_s']} img/s "
                  f"(IO+decode {r['io_fraction']:.0%})")

    best = results[str(max(args.sizes))]
    rate = best["img_per_s"]
    est_s = N_TOTAL_IMAGES / rate

    report = {
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "encoder": spec.name,
        "batch_size": args.batch_size,
        "trials": results,
        "extrapolation": {
            "n_total_images": N_TOTAL_IMAGES,
            "img_per_s": rate,
            "estimated_seconds_one_encoder": round(est_s, 1),
            "estimated_hours_one_encoder": round(est_s / 3600, 2),
            "estimated_hours_three_encoders": round(3 * est_s / 3600, 2),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"throughput_{spec.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    e = report["extrapolation"]
    print(f"\n{'='*58}")
    print(f"  Throughput medido      : {rate} img/s ({device})")
    print(f"  Dataset completo       : {N_TOTAL_IMAGES:,} imagenes")
    print(f"  ESTIMADO 1 encoder     : {e['estimated_hours_one_encoder']} h")
    print(f"  ESTIMADO 3 encoders    : {e['estimated_hours_three_encoders']} h")
    print(f"{'='*58}")
    print(f"\nGuardado: {OUT}")


if __name__ == "__main__":
    main()
