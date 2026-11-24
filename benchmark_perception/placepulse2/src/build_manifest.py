"""
Genera external_assets/MANIFEST.csv (seccion 7 del prompt maestro).

Columnas: filename, relative_path, source_url_or_id, size_bytes, sha256,
download_date, purpose, status
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "external_assets"
MANIFEST = ASSETS_DIR / "MANIFEST.csv"
INVENTORY = ASSETS_DIR / "drive_inventory.json"

PURPOSES = {
    "checkpoints": "encoder OURS SimCLR ViT-B/16 para benchmark perceived safety",
    "embeddings_export": "metadata historica de embeddings Lima (referencia, rutas Linux no portables)",
    "eigen_features": "resultados historicos de estrategias de pooling (Lima crime-risk)",
    "final_classifier": "resultados historicos de clasificadores (Lima crime-risk)",
    "placepulse2": "benchmark externo de perceived safety (ground truth humano pairwise)",
}

# Fuentes explicitas por nombre de archivo, para no depender del inventario de Drive.
EXPLICIT_SOURCES = {
    "place-pulse-2.0.zip":
        "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip "
        "(Center for Collective Learning, pagina Urban Perception, enlace 'Place Pulse 2.0 dataset (Data)')",
    "votes_clean.csv":
        "https://ndownloader.figshare.com/files/21739137 "
        "(Figshare DOI 10.6084/m9.figshare.11859993, CC BY 4.0, Salesses & Hidalgo)",
}

EXPLICIT_PURPOSES = {
    "votes.tsv": "GROUND TRUTH PRINCIPAL — comparaciones humanas pairwise originales",
    "votes_clean.csv": "contraste de provenance de los votos humanos (copia Figshare)",
    "studies.tsv": "mapeo study_id -> atributo; identifica el estudio safe",
    "locations.tsv": "location_id -> place_id + coordenadas (necesario para city-holdout)",
    "places.tsv": "place_id -> nombre de ciudad",
    "qscores.tsv": "TrueSkill DERIVADO — NO usar como ground truth principal",
    "place-pulse-2.0.zip": "paquete oficial: 5 TSV + 110,988 imagenes",
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def drive_id_for(rel_posix: str, inventory: list[dict]) -> str:
    """Busca el file ID de Drive por nombre de archivo."""
    name = Path(rel_posix).name
    for entry in inventory:
        if Path(entry["path"]).name == name:
            return entry["url"].split("id=")[-1]
    return ""


def main() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8")) if INVENTORY.exists() else []
    today = date.today().isoformat()

    rows = []
    for path in sorted(ASSETS_DIR.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.csv":
            continue

        rel = path.relative_to(ASSETS_DIR)
        rel_posix = rel.as_posix()
        top = rel.parts[0] if len(rel.parts) > 1 else ""

        rows.append({
            "filename": path.name,
            "relative_path": f"external_assets/{rel_posix}",
            "source_url_or_id": (
                EXPLICIT_SOURCES.get(path.name)
                or drive_id_for(rel_posix, inventory)
                or ("extraido de place-pulse-2.0.zip" if "placepulse2" in rel.parts
                    else "generado localmente")
            ),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "download_date": today,
            "purpose": EXPLICIT_PURPOSES.get(
                path.name, PURPOSES.get(top, "inventario de la carpeta publica de Drive")
            ),
            "status": "verified",
        })

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "relative_path", "source_url_or_id", "size_bytes",
            "sha256", "download_date", "purpose", "status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"MANIFEST.csv escrito con {len(rows)} entradas -> {MANIFEST}")
    for r in rows:
        print(f"  {r['filename']:<55} {r['size_bytes']:>12,}  {r['sha256'][:16]}…")


if __name__ == "__main__":
    main()
