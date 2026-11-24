"""
Descarga desde Drive los assets de Lima que faltan en local y que hacen falta
para las fases E (visual / structural / fused) y F (ego-0..3).

Que falta en local y por que importa:
  - neighbourhood_tensors/*_neighbourhood_tensors.npy : el tensor por nodo con
    los embeddings de su vecindario. Es la entrada directa del experimento de
    profundidad espacial; sin el, las fases E y F estan BLOQUEADAS.
  - eigen_features/*_labels.npy                       : etiquetas high/low crime.
  - eigen_features/*_features_*.npy                   : features derivadas ya
    calculadas (se descargan para poder reproducir el baseline heredado).

NO se re-descarga nada que ya exista localmente con tamano > 0.
Se registra sha256 y tamano de cada fichero en un MANIFEST propio, para no
perder provenance.

Uso:
    python fetch_lima_assets.py            # descarga lo que falta
    python fetch_lima_assets.py --dry-run  # solo informa
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from utils import REPO_ROOT, log

INVENTORY = REPO_ROOT / "external_assets" / "drive_inventory.json"
DEST_ROOT = REPO_ROOT / "external_assets" / "lima_graph"
MANIFEST = DEST_ROOT / "MANIFEST_lima_graph.csv"

# Solo lo necesario para las fases E/F. No se baja todo el Drive.
WANTED = [
    "neighbourhood_tensors/Barranco_2016_neighbourhood_tensors.npy",
    "neighbourhood_tensors/La_Victoria_2016_neighbourhood_tensors.npy",
    "neighbourhood_tensors/Barranco_2016_node_ids_order.json",
    "neighbourhood_tensors/La_Victoria_2016_node_ids_order.json",
    "eigen_features/Barranco_2016_labels.npy",
    "eigen_features/La_Victoria_2016_labels.npy",
]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inv = {it["path"]: it["url"] for it in json.loads(INVENTORY.read_text(encoding="utf-8"))}
    missing_in_inventory = [w for w in WANTED if w not in inv]
    if missing_in_inventory:
        log(f"NO estan en el inventario de Drive: {missing_in_inventory}")

    rows = []
    for rel in WANTED:
        if rel not in inv:
            rows.append({"path": rel, "status": "NO_EN_INVENTARIO"})
            continue
        dest = DEST_ROOT / rel
        if dest.exists() and dest.stat().st_size > 0:
            log(f"ya existe: {rel} ({dest.stat().st_size/1e6:.1f} MB)")
            rows.append({"path": rel, "status": "ya_local",
                         "size_bytes": dest.stat().st_size, "sha256": sha256(dest)})
            continue
        if args.dry_run:
            log(f"[dry-run] descargaria {rel}")
            rows.append({"path": rel, "status": "pendiente"})
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"descargando {rel} ...")
        try:
            import gdown
            gdown.download(inv[rel], str(dest), quiet=True)
        except Exception as e:  # noqa: BLE001
            log(f"  FALLO: {e}")
            rows.append({"path": rel, "status": f"fallo: {type(e).__name__}"})
            continue
        if not dest.exists() or dest.stat().st_size == 0:
            log("  FALLO: fichero vacio (¿permiso de Drive?)")
            rows.append({"path": rel, "status": "fallo_vacio"})
            dest.unlink(missing_ok=True)
            continue
        log(f"  OK {dest.stat().st_size/1e6:.1f} MB")
        rows.append({"path": rel, "status": "descargado",
                     "size_bytes": dest.stat().st_size, "sha256": sha256(dest),
                     "source_url": inv[rel], "download_date": str(date.today())})

    if not args.dry_run and rows:
        import pandas as pd
        DEST_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(MANIFEST, index=False)
        log(f"\nmanifiesto -> {MANIFEST}")

    ok = sum(1 for r in rows if r["status"] in ("descargado", "ya_local"))
    log(f"\ndisponibles: {ok}/{len(WANTED)}")


if __name__ == "__main__":
    main()
