"""
FASE D (preparacion) — Muestra reproducible para el proof-of-concept semantico.

Objetivo: un conjunto pequeno pero VISUALMENTE DIVERSO sobre el que comparar
CLIP / Grounding DINO / YOLO-World, sin lanzar todavia extraccion masiva.

Criterio de muestreo
--------------------
Estratificado por CIUDAD (diversidad geografica y morfologica) y, dentro de
cada ciudad, por el TERCIL de tasa de victoria humana de la imagen en los pares
`safe` (baja / media / alta). Asi la muestra cubre el rango perceptual y no solo
la parte facil del espectro.

IMPORTANTE sobre la tasa de victoria: se usa UNICAMENTE como variable de
ESTRATIFICACION para diversificar la muestra. NO es ground truth de ningun
concepto del codebook ni se usa como etiqueta en el PoC. El ground truth del
benchmark sigue siendo la comparacion pareada humana.

Se exigen al menos `MIN_COMPARISONS` comparaciones por imagen para que el
tercil sea informativo y no ruido de una sola votacion.

Salidas:
    outputs/poc/poc_sample.csv          (location_id, city, stratum, ruta en zip)
    outputs/poc/images/<location_id>.jpg
    outputs/poc/poc_sample_summary.json

Uso:
    python build_poc_sample.py --per-city 5 --cities 12
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from utils import MANIFESTS, OUTPUTS, ZIP_PATH, log, set_seed

OUT = OUTPUTS / "poc"
SEED = 42
MIN_COMPARISONS = 4

# Ciudades elegidas a mano para cubrir continentes y morfologias urbanas muy
# distintas. Es una decision de diseno explicita, no un muestreo aleatorio de
# ciudades, porque con pocas ciudades el azar puede dar un conjunto homogeneo.
TARGET_CITIES = [
    "Lima" if False else "Mexico City",   # America Latina (proxy morfologico de Lima)
    "Sao Paulo", "Santiago",              # America Latina
    "New York", "Houston",                # Norteamerica
    "London", "Berlin", "Rome",           # Europa
    "Bangkok", "Tokyo",                   # Asia
    "Cape Town", "Johannesburg",          # Africa
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=5)
    ap.add_argument("--cities", type=int, default=12)
    args = ap.parse_args()

    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)

    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    pairs["label"] = pairs["label"].astype(int)

    # tasa de victoria por imagen (solo para estratificar)
    wins = Counter()
    shows = Counter()
    for left, right, lab in zip(pairs["left"], pairs["right"], pairs["label"]):
        shows[left] += 1
        shows[right] += 1
        wins[left if lab == 1 else right] += 1
    stat = pd.DataFrame({"location_id": list(shows)})
    stat["n_comparisons"] = stat["location_id"].map(shows)
    stat["win_rate"] = stat["location_id"].map(lambda i: wins[i] / shows[i])
    stat = stat[stat["n_comparisons"] >= MIN_COMPARISONS]

    df = man.merge(stat, on="location_id", how="inner")
    log(f"imagenes con >= {MIN_COMPARISONS} comparaciones: {len(df):,}")

    cities = [c for c in TARGET_CITIES if c in set(df["city"])][: args.cities]
    log(f"ciudades: {cities}")

    picked = []
    for city in cities:
        sub = df[df["city"] == city].copy()
        if len(sub) < args.per_city:
            continue
        # terciles de tasa de victoria DENTRO de la ciudad
        sub["stratum"] = pd.qcut(sub["win_rate"], 3,
                                 labels=["baja", "media", "alta"], duplicates="drop")
        per_stratum = max(1, args.per_city // 3)
        for s, g in sub.groupby("stratum", observed=True):
            take = min(per_stratum, len(g))
            idx = rng.choice(len(g), size=take, replace=False)
            picked.append(g.iloc[idx])
    sample = pd.concat(picked).drop_duplicates("location_id").reset_index(drop=True)
    sample = sample[["location_id", "city", "stratum", "win_rate",
                     "n_comparisons", "image_path"]]
    log(f"muestra: {len(sample)} imagenes, {sample['city'].nunique()} ciudades")
    log(f"  por estrato: {sample['stratum'].value_counts().to_dict()}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for _, r in sample.iterrows():
            dst = OUT / "images" / f"{r['location_id']}.jpg"
            if dst.exists():
                continue
            with zf.open(r["image_path"]) as fh:
                data = fh.read()
            with Image.open(io.BytesIO(data)) as im:
                im.convert("RGB").save(dst, quality=92)

    sample.to_csv(OUT / "poc_sample.csv", index=False)
    (OUT / "poc_sample_summary.json").write_text(json.dumps({
        "seed": SEED, "n_images": len(sample),
        "cities": sorted(sample["city"].unique().tolist()),
        "per_stratum": {k: int(v) for k, v in sample["stratum"].value_counts().items()},
        "min_comparisons": MIN_COMPARISONS,
        "stratification_note": "win_rate se usa SOLO para estratificar la muestra y "
                               "cubrir el rango perceptual; no es etiqueta ni ground "
                               "truth de ningun concepto del codebook",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
