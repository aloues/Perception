"""
Extensión P-LIMA (D4b): DINOv2 ViT-B/14 a 224 px sobre las vistas DESCARGADAS de San Isidro / Lince, con el mismo
modelo y transformación que Place Pulse (encoders.load_dinov2) y el vocabulario de parches de EXP-044 (ajustado con
imágenes de otras ciudades de PP; ninguna vista de Lima lo toca).

Por vista se guarda: CLS (L2), media de parches y el histograma de conteos (256 parches) en cada vocabulario de
EXP-044. Solo para San Isidro y Lince (B/LV no tienen imágenes en local): se reporta aparte, no entra en las tablas
de 4 columnas.

Uso:  python extract_plima_dinov2.py --folder Inseguros-San_Isidro-GGZ-2016
Salida: external_assets/embeddings_export_plima/<carpeta>/dinov2.npz (+ dinov2_index.json)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from exp044_patch_bovw import OUT as EXP044, l2n, vocab_names
from lima_views import COLLECTOR_DATA, PLIMA_EXPORTS
from utils import log


def main() -> None:
    import torch
    from PIL import Image

    from encoders import load_dinov2
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = COLLECTOR_DATA / args.folder
    paths = sorted(p for p in root.glob("*/heading_*.jpg"))
    out = PLIMA_EXPORTS / args.folder
    out.mkdir(parents=True, exist_ok=True)
    spec = load_dinov2("cpu")
    npre = spec.model.num_prefix_tokens
    C = {n: np.load(EXP044 / f"vocab_{n}.npy") for n in vocab_names()}
    C2 = {n: (c ** 2).sum(1) for n, c in C.items()}
    cls, avg, H, t0 = [], [], {n: [] for n in C}, time.perf_counter()
    with torch.no_grad():
        for a in range(0, len(paths), args.batch_size):
            tens = []
            for p in paths[a:a + args.batch_size]:
                with Image.open(p) as im:
                    tens.append(spec.transform(im.convert("RGB")))
            f = spec.model.forward_features(torch.stack(tens)).float().numpy()
            cls.append(l2n(f[:, 0]))
            avg.append(f[:, npre:].mean(1))
            tok = l2n(f[:, npre:])
            B, T, D = tok.shape
            flat = tok.reshape(-1, D)
            for n, c in C.items():
                w = (1.0 - 2.0 * flat @ c.T + C2[n][None]).argmin(1)
                H[n].append(np.stack([np.bincount(w[b * T:(b + 1) * T], minlength=len(c)) for b in range(B)]))
            if (a // args.batch_size) % 20 == 0:
                log(f"  {a + len(tens)}/{len(paths)} vistas, {(time.perf_counter() - t0) / 60:.1f} min")
    arrays = {"cls": np.concatenate(cls).astype(np.float32), "avg": np.concatenate(avg).astype(np.float32),
              **{f"hist_{n}": np.concatenate(H[n]).astype(np.uint16) for n in C}}
    tmp = out / "dinov2.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out / "dinov2.npz")
    (out / "dinov2_index.json").write_text(json.dumps({
        "encoder": "dinov2_vit_b14 timm lvd142m @224 (= encoders.load_dinov2)", "vocab_source": "outputs/exp044",
        "paths": [p.relative_to(COLLECTOR_DATA).as_posix() for p in paths]}), encoding="utf-8")
    log(f"{args.folder}: {len(paths)} vistas en {(time.perf_counter() - t0) / 60:.1f} min → {out / 'dinov2.npz'}")


if __name__ == "__main__":
    main()
