"""
Fase D — Inspeccion del dataset Place Pulse 2.0 (secciones 13 y 21).

Fuente primaria : external_assets/placepulse2/extracted/votes.tsv
                  (comparaciones humanas originales, fuente oficial CCL)
Contraste       : external_assets/placepulse2/raw/votes_clean.csv
                  (deposito Figshare, DOI 10.6084/m9.figshare.11859993)

NO se usa qscores.tsv como target: contiene TrueSkill, que es un score DERIVADO.

Produce:
    data/manifests/dataset_summary.json
    data/manifests/images_manifest.csv
    data/manifests/safe_location_ids.txt
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PP = REPO_ROOT / "external_assets" / "placepulse2"
EXTRACTED = PP / "extracted"
ZIP_PATH = PP / "raw" / "place-pulse-2.0.zip"
VOTES_CLEAN = PP / "raw" / "votes_clean.csv"

MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"

TIE_LABEL = "equal"
VALID_CHOICES = {"left", "right", TIE_LABEL}


def load_studies() -> pd.DataFrame:
    return pd.read_csv(EXTRACTED / "studies.tsv", sep="\t", dtype=str)


def resolve_safe_study_id(studies: pd.DataFrame) -> str:
    """Identifica el study_id del atributo safe por su study_question, no hardcodeado."""
    match = studies[studies["study_question"].str.strip().str.lower().isin({"safer", "safe"})]
    if len(match) != 1:
        raise RuntimeError(
            f"Se esperaba exactamente 1 estudio de tipo safe, se encontraron {len(match)}:\n{match}"
        )
    return match.iloc[0]["_id"]


def image_index_from_zip() -> dict[str, str]:
    """
    Mapea location_id -> ruta dentro del ZIP, sin extraerlo.

    Convencion real de nombres (verificada, no asumida):
        images/{lat}_{lon}_{location_id}_{Ciudad}.JPG
    El location_id es el tercer campo separado por '_'; la ciudad del nombre
    puede venir sin espacios (CapeTown, NewYork), por lo que NO se usa como
    fuente de ciudad: para eso estan locations.tsv y places.tsv.
    """
    index: dict[str, str] = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for n in z.namelist():
            if not n.startswith("images/") or not n.lower().endswith(".jpg"):
                continue
            parts = Path(n).stem.split("_")
            if len(parts) < 4:
                continue
            index[parts[2]] = n
    return index


def analyse_pairs(df: pd.DataFrame) -> dict:
    left = df["left"].astype(str)
    right = df["right"].astype(str)

    lo = left.where(left <= right, right)
    hi = right.where(left <= right, left)
    unordered = lo + "|" + hi

    orient = pd.DataFrame({"u": unordered, "flipped": left > right}).drop_duplicates()
    n_inverse = int((orient.groupby("u")["flipped"].nunique() > 1).sum())

    return {
        "n_unique_ordered_pairs": int((left + "|" + right).nunique()),
        "n_unique_unordered_pairs": int(unordered.nunique()),
        "n_repeated_pairs": int(len(df) - unordered.nunique()),
        "n_pairs_with_both_orientations": n_inverse,
        "n_self_pairs": int((left == right).sum()),
    }


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    studies = load_studies()
    safe_study_id = resolve_safe_study_id(studies)
    print(f"study_id de safe (resuelto desde studies.tsv): {safe_study_id}")

    # ── Fuente primaria: votes.tsv ────────────────────────────────────────
    votes = pd.read_csv(EXTRACTED / "votes.tsv", sep="\t", dtype=str)
    total_votes = len(votes)
    print(f"votes.tsv: {total_votes:,} filas")

    safe = votes[votes["study_id"] == safe_study_id].copy()
    choice = safe["choice"].astype(str).str.strip().str.lower()
    safe_choice_counts = Counter(choice)

    invalid_mask = ~choice.isin(VALID_CHOICES)
    n_invalid = int(invalid_mask.sum())

    valid = safe.loc[~invalid_mask].copy()
    valid_choice = choice.loc[~invalid_mask]

    # ── Geografia: location -> place -> ciudad ────────────────────────────
    locations = pd.read_csv(EXTRACTED / "locations.tsv", sep="\t", dtype=str)
    places = pd.read_csv(EXTRACTED / "places.tsv", sep="\t", dtype=str)
    loc_to_place = dict(zip(locations["_id"], locations["place_id"]))
    place_to_city = dict(zip(places["_id"], places["place_name"]))

    loc_ids = pd.unique(pd.concat([
        valid["left"].astype(str), valid["right"].astype(str)
    ], ignore_index=True))

    cities = Counter()
    n_loc_without_location_row = 0
    for lid in loc_ids:
        pid = loc_to_place.get(lid)
        if pid is None:
            n_loc_without_location_row += 1
            continue
        cities[place_to_city.get(pid, f"<place_id {pid} sin nombre>")] += 1

    # ── Correspondencia con imagenes reales del ZIP ───────────────────────
    print("Leyendo indice de imagenes del ZIP…")
    zip_index = image_index_from_zip()
    zip_images = set(zip_index)
    referenced = set(map(str, loc_ids))
    missing = referenced - zip_images
    unused = zip_images - referenced

    # ── Contraste con la copia de Figshare ────────────────────────────────
    clean_safe_rows = 0
    if VOTES_CLEAN.exists():
        for chunk in pd.read_csv(VOTES_CLEAN, chunksize=500_000, low_memory=False):
            clean_safe_rows += int((chunk["study_id"].astype(str) == safe_study_id).sum())

    declared = studies.loc[studies["_id"] == safe_study_id, "num_votes"].iloc[0]

    report = {
        "primary_source": {
            "file": "external_assets/placepulse2/extracted/votes.tsv",
            "origin": "Center for Collective Learning — place-pulse-2.0.zip",
            "role": "comparaciones humanas originales (ground truth principal)",
        },
        "cross_check_source": {
            "file": "external_assets/placepulse2/raw/votes_clean.csv",
            "origin": "Figshare DOI 10.6084/m9.figshare.11859993",
            "license": "CC BY 4.0",
            "role": "contraste de provenance",
        },
        "excluded_from_ground_truth": {
            "qscores.tsv": "TrueSkill — score DERIVADO, no comparacion humana",
        },
        "studies": studies.to_dict(orient="records"),
        "safe_study_id": safe_study_id,
        "total_votes_all_attributes": total_votes,
        "safe": {
            "n_votes_declared_in_studies_tsv": int(declared),
            "n_votes_in_votes_tsv": int(len(safe)),
            "n_votes_in_votes_clean_csv": clean_safe_rows,
            "choice_values_raw": dict(safe_choice_counts.most_common()),
            "n_invalid_choice_rows": n_invalid,
            "n_votes_valid": int(len(valid)),
            "n_ties": int((valid_choice == TIE_LABEL).sum()),
            "tie_fraction": round(float((valid_choice == TIE_LABEL).mean()), 6),
            "n_left": int((valid_choice == "left").sum()),
            "n_right": int((valid_choice == "right").sum()),
            "n_voters": int(valid["voter_uniqueid"].nunique()),
            "n_unique_locations_referenced": int(len(referenced)),
            "n_cities": len(cities),
            "cities": dict(sorted(cities.items(), key=lambda kv: -kv[1])),
            "n_locations_without_row_in_locations_tsv": n_loc_without_location_row,
            **analyse_pairs(valid),
        },
        "images": {
            "n_images_in_zip": len(zip_images),
            "n_referenced_by_safe_votes": len(referenced),
            "n_missing_images": len(missing),
            "n_images_unused_by_safe": len(unused),
        },
    }

    (MANIFESTS / "dataset_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for lid in sorted(referenced):
        pid = loc_to_place.get(lid, "")
        rows.append({
            "location_id": lid,
            "place_id": pid,
            "city": place_to_city.get(pid, ""),
            "image_in_zip": lid in zip_index,
            "image_path": zip_index.get(lid, ""),
        })
    pd.DataFrame(rows).to_csv(MANIFESTS / "images_manifest.csv", index=False)
    (MANIFESTS / "safe_location_ids.txt").write_text(
        "\n".join(sorted(referenced)), encoding="utf-8"
    )

    s = report["safe"]
    print(f"\n  total votes (todos)     : {total_votes:,}")
    print(f"  safe declarado          : {s['n_votes_declared_in_studies_tsv']:,}")
    print(f"  safe en votes.tsv       : {s['n_votes_in_votes_tsv']:,}")
    print(f"  safe en votes_clean.csv : {s['n_votes_in_votes_clean_csv']:,}")
    print(f"  choice invalidos        : {s['n_invalid_choice_rows']:,}")
    print(f"  votos validos           : {s['n_votes_valid']:,}")
    print(f"  ties                    : {s['n_ties']:,} ({s['tie_fraction']:.2%})")
    print(f"  left / right            : {s['n_left']:,} / {s['n_right']:,}")
    print(f"  votantes                : {s['n_voters']:,}")
    print(f"  ciudades                : {s['n_cities']}")
    print(f"  locations referenciadas : {s['n_unique_locations_referenced']:,}")
    print(f"  pares repetidos         : {s['n_repeated_pairs']:,}")
    print(f"  pares ambas orientac.   : {s['n_pairs_with_both_orientations']:,}")
    print(f"  self pairs              : {s['n_self_pairs']:,}")
    i = report["images"]
    print(f"\n  imagenes en el ZIP      : {i['n_images_in_zip']:,}")
    print(f"  IMAGENES FALTANTES      : {i['n_missing_images']:,}")
    print(f"  imagenes no usadas      : {i['n_images_unused_by_safe']:,}")


if __name__ == "__main__":
    main()
