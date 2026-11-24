"""
EXP-021 (preparación) — contexto espacial ego-0 / ego-1 sobre Place Pulse.

POR QUE
-------
EXP-013 cerro la rama de Lima con resultado negativo: el dataset de crimen es circular (las
imagenes se recolectaron en los puntos de delito, la etiqueta viene del conteo de delitos), asi
que no se puede usar para responder RQ3 (cuanto aporta el contexto espacial). Pero Place Pulse
tiene coordenadas reales para el 100% de sus 110,987 imagenes (vienen en el nombre del archivo),
con densidad espacial suficiente (mediana ~149 m al vecino mas cercano) para construir vecindarios,
y su ground truth (voto humano pareado) es INDEPENDIENTE de como se muestrearon las imagenes: no
puede sufrir el mismo tipo de circularidad que Lima.

DISEÑO
------
Para cada imagen i, dentro de su misma ciudad:
  - ego-0 = [z_i, 0]                                           (1536-d, sin contexto)
  - ego-1 = [mean(z_i, vecinas a <= RADIUS_M), std(...)]        (1536-d, con contexto)
Mismo numero de dimensiones en los dos casos (1536-d) para que la comparacion aisle el efecto del
CONTEXTO y no el de la capacidad del modelo lineal. Vecinas = otras imagenes de la MISMA ciudad
dentro de un radio en metros (no un grafo de calles: Place Pulse no trae un grafo vial, se aproxima
por distancia euclidea sobre lat/lon). Maximo NEIGH_MAX vecinas mas cercanas para acotar el coste y
mantener paridad de escala con el vecindario de Lima (que usaba 20 slots).

z_i es el embedding DINOv2 ya extraido (outputs/features/dinov2_vit_b14/), L2-normalizado. No se
vuelve a extraer nada.

Salidas (compatibles con train_probe.py --features-root):
    outputs/features_placepulse_context/dinov2_ego0/{index.json, features.npy}
    outputs/features_placepulse_context/dinov2_ego1/{index.json, features.npy}
    outputs/features_placepulse_context/context_build_summary.json

Uso:
    python build_placepulse_context.py --radius-m 250 --neigh-max 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from utils import FEATURES, MANIFESTS, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "features_placepulse_context"
COORD_RE = re.compile(r"images/(-?\d+\.\d+)_(-?\d+\.\d+)_")


def parse_coords(image_path: str) -> tuple[float, float] | tuple[None, None]:
    m = COORD_RE.match(image_path)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-m", type=float, default=250.0,
                    help="radio de vecindad en metros (p90 de dist. al vecino mas cercano ~283 m)")
    ap.add_argument("--neigh-max", type=int, default=20,
                    help="tope de vecinas mas cercanas dentro del radio (paridad con Lima: 20 slots)")
    ap.add_argument("--encoder", default="dinov2_vit_b14")
    args = ap.parse_args()

    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man.dropna(subset=["image_path"])
    ll = man["image_path"].apply(parse_coords)
    man["lat"] = ll.apply(lambda t: t[0])
    man["lon"] = ll.apply(lambda t: t[1])
    n_no_coord = int(man["lat"].isna().sum())
    man = man.dropna(subset=["lat", "lon"])
    log(f"manifiesto: {len(man):,} imagenes con coordenadas ({n_no_coord} sin coords descartadas)")

    d = FEATURES / args.encoder
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    shards = sorted(d.glob("shard_*.npy"))
    Z = np.concatenate([np.load(s) for s in shards], axis=0) if shards else np.load(d / "features.npy")
    ids_order = index["location_ids"][: Z.shape[0]]
    row_of = {lid: i for i, lid in enumerate(ids_order)}
    log(f"embeddings: {Z.shape} ({args.encoder})")

    man = man[man["location_id"].isin(row_of)].reset_index(drop=True)
    man["row"] = man["location_id"].map(row_of)
    dim = Z.shape[1]

    ego0 = np.zeros((len(man), 2 * dim), dtype=np.float32)
    ego1 = np.zeros((len(man), 2 * dim), dtype=np.float32)
    cov_rows = []

    for city, g in man.groupby("city"):
        idx_local = g.index.to_numpy()
        rows = g["row"].to_numpy()
        lat0 = g["lat"].astype(float).mean()
        # proyeccion equirectangular local (suficiente a escala de una ciudad)
        xy = np.c_[g["lon"].astype(float) * 111320 * np.cos(np.radians(lat0)),
                   g["lat"].astype(float) * 110540]
        tree = cKDTree(xy)
        n_neighbors = []
        for j, (li, r) in enumerate(zip(idx_local, rows)):
            z_self = Z[r]
            ego0[li, :dim] = z_self
            near = tree.query_ball_point(xy[j], args.radius_m)
            near = [k for k in near if idx_local[k] != li]
            if len(near) > args.neigh_max:
                dists = np.linalg.norm(xy[near] - xy[j], axis=1)
                near = [near[k] for k in np.argsort(dists)[: args.neigh_max]]
            n_neighbors.append(len(near))
            if near:
                neigh_z = Z[rows[near]]
                pool = np.vstack([z_self[None, :], neigh_z])
            else:
                pool = z_self[None, :]
            ego1[li, :dim] = pool.mean(axis=0)
            ego1[li, dim:] = pool.std(axis=0)
        nn = np.array(n_neighbors)
        cov_rows.append(dict(city=city, n_images=len(g), mean_neighbors=round(float(nn.mean()), 2),
                             pct_zero_neighbors=round(float((nn == 0).mean()), 3),
                             median_neighbors=int(np.median(nn))))
        log(f"  {city:<18} n={len(g):5d}  vecinas/imagen media={nn.mean():.1f}  "
            f"sin vecinas={((nn == 0).mean()):.1%}")

    ids_final = man["location_id"].tolist()
    for name, X in (("dinov2_ego0", ego0), ("dinov2_ego1", ego1)):
        sub = OUT / name
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "features.npy", X)
        (sub / "index.json").write_text(json.dumps({
            "encoder_base": args.encoder, "variant": name, "embed_dim": int(X.shape[1]),
            "n_images": len(ids_final), "location_ids": ids_final,
            "radius_m": args.radius_m, "neigh_max": args.neigh_max,
            "l2_normalized_base": True, "contains_zero_padding": name == "dinov2_ego0",
        }, indent=2), encoding="utf-8")
        log(f"-> {sub}  {X.shape}")

    cov = pd.DataFrame(cov_rows)
    (OUT / "context_build_summary.json").write_text(json.dumps({
        "seed": SEED, "radius_m": args.radius_m, "neigh_max": args.neigh_max,
        "encoder_base": args.encoder, "n_images_total": len(man),
        "n_images_sin_coords_descartadas": n_no_coord,
        "coverage_by_city": cov.to_dict("records"),
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    cov.to_csv(OUT / "context_coverage_by_city.csv", index=False)
    log(f"\ncobertura global: media {cov.mean_neighbors.mean():.1f} vecinas/imagen, "
        f"{cov.pct_zero_neighbors.mean():.1%} de imagenes sin ninguna vecina en el radio")
    log(f"listo en {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
