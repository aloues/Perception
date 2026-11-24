"""
Prepara los pares del atributo `safe` a partir de las comparaciones humanas
originales (votes.tsv). Escrito DESPUES de auditar la estructura real del
dataset (ver DATASET_AUDIT.md), no antes.

Decisiones explicitas:
  - Ground truth  : votes.tsv, study safe. NUNCA qscores.tsv (TrueSkill derivado).
  - choice        : se valida contra {left, right, equal}; nada se asume.
  - ties (equal)  : se SEPARAN, no se descartan en silencio. Quedan en su propio
                    archivo y se contabilizan. El probe se entrena solo con pares
                    decisivos; los ties se reportan aparte.
  - imagen ausente: se descarta el par completo si cualquiera de los dos extremos
                    no tiene archivo de imagen, y se reporta cuantos se pierden.
  - label         : y = 1 si gano la imagen IZQUIERDA, 0 si gano la DERECHA,
                    consistente con logit = s_left - s_right.

Salidas (data/manifests/):
    pairs_safe.csv          pares decisivos (entrenamiento/evaluacion)
    pairs_safe_ties.csv     pares empatados (solo reporte)
    pairs_summary.json      conteos de cada filtro aplicado
"""

from __future__ import annotations

import json

import pandas as pd

from utils import EXTRACTED, MANIFESTS, TIE_LABEL, VALID_CHOICES, log, resolve_safe_study_id


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    safe_study_id = resolve_safe_study_id()
    log(f"study_id safe: {safe_study_id}")

    votes = pd.read_csv(EXTRACTED / "votes.tsv", sep="\t", dtype=str)
    n_all = len(votes)

    safe = votes[votes["study_id"] == safe_study_id].copy()
    n_safe = len(safe)
    log(f"votos totales: {n_all:,} | votos safe: {n_safe:,}")

    safe["choice"] = safe["choice"].astype(str).str.strip().str.lower()

    # 1) Validacion explicita de choice (no se asume el dominio).
    valid_mask = safe["choice"].isin(VALID_CHOICES)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        log(f"  descartados por choice invalido: {n_invalid:,}")
        log(f"    valores: {safe.loc[~valid_mask, 'choice'].value_counts().head(10).to_dict()}")
    safe = safe[valid_mask].copy()

    # 2) Geografia por imagen (para city_holdout).
    locations = pd.read_csv(EXTRACTED / "locations.tsv", sep="\t", dtype=str)
    places = pd.read_csv(EXTRACTED / "places.tsv", sep="\t", dtype=str)
    place_name = dict(zip(places["_id"], places["place_name"]))
    loc_city = {
        lid: place_name.get(pid, "")
        for lid, pid in zip(locations["_id"], locations["place_id"])
    }

    # 3) Imagenes realmente disponibles.
    manifest = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    manifest["image_in_zip"] = manifest["image_in_zip"].map({"True": True, "False": False})
    available = dict(
        zip(manifest.loc[manifest["image_in_zip"], "location_id"],
            manifest.loc[manifest["image_in_zip"], "image_path"])
    )

    has_both = safe["left"].isin(available) & safe["right"].isin(available)
    n_dropped_missing = int((~has_both).sum())
    log(f"  descartados por imagen ausente: {n_dropped_missing:,} "
        f"({n_dropped_missing / max(len(safe), 1):.3%})")
    safe = safe[has_both].copy()

    # 4) Separacion de ties.
    is_tie = safe["choice"] == TIE_LABEL
    ties = safe[is_tie].copy()
    decisive = safe[~is_tie].copy()

    log(f"  ties separados: {len(ties):,} ({len(ties) / max(len(safe), 1):.2%})")
    log(f"  pares decisivos: {len(decisive):,}")

    # 5) Etiqueta: 1 = gano la izquierda.
    decisive["label"] = (decisive["choice"] == "left").astype(int)

    for df in (decisive, ties):
        df["city_left"] = df["left"].map(loc_city)
        df["city_right"] = df["right"].map(loc_city)
        df["image_left"] = df["left"].map(available)
        df["image_right"] = df["right"].map(available)

    cols = ["_id", "left", "right", "choice", "voter_uniqueid", "timestamp",
            "city_left", "city_right", "image_left", "image_right"]

    decisive[cols + ["label"]].to_csv(MANIFESTS / "pairs_safe.csv", index=False)
    ties[cols].to_csv(MANIFESTS / "pairs_safe_ties.csv", index=False)

    same_city = int((decisive["city_left"] == decisive["city_right"]).sum())

    summary = {
        "ground_truth_source": "external_assets/placepulse2/extracted/votes.tsv",
        "safe_study_id": safe_study_id,
        "excluded_source": {"qscores.tsv": "TrueSkill derivado — no es ground truth"},
        "funnel": {
            "votes_all_attributes": n_all,
            "votes_safe": n_safe,
            "dropped_invalid_choice": n_invalid,
            "dropped_missing_image": n_dropped_missing,
            "ties_separated": int(len(ties)),
            "decisive_pairs": int(len(decisive)),
        },
        "tie_policy": (
            "Los ties se excluyen del entrenamiento del probe y se reportan por "
            "separado en pairs_safe_ties.csv. No se eliminan en silencio."
        ),
        "label_convention": "label=1 si gano la imagen izquierda; logit = s_left - s_right",
        "decisive": {
            "n_pairs": int(len(decisive)),
            "n_label_1_left_wins": int(decisive["label"].sum()),
            "n_label_0_right_wins": int((decisive["label"] == 0).sum()),
            "balance_left": round(float(decisive["label"].mean()), 6),
            "n_unique_images": int(pd.unique(
                pd.concat([decisive["left"], decisive["right"]], ignore_index=True)
            ).size),
            "n_voters": int(decisive["voter_uniqueid"].nunique()),
            "n_cities": int(pd.unique(pd.concat(
                [decisive["city_left"], decisive["city_right"]], ignore_index=True
            )).size),
            "n_same_city_pairs": same_city,
            "same_city_fraction": round(same_city / max(len(decisive), 1), 6),
        },
        "ties": {
            "n_pairs": int(len(ties)),
            "n_unique_images": int(pd.unique(
                pd.concat([ties["left"], ties["right"]], ignore_index=True)
            ).size),
        },
    }

    (MANIFESTS / "pairs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    d = summary["decisive"]
    log("\n" + "=" * 58)
    log(f"  pares decisivos      : {d['n_pairs']:,}")
    log(f"  balance left/right   : {d['balance_left']:.4f}")
    log(f"  imagenes unicas      : {d['n_unique_images']:,}")
    log(f"  ciudades             : {d['n_cities']}")
    log(f"  pares misma ciudad   : {d['n_same_city_pairs']:,} ({d['same_city_fraction']:.2%})")
    log(f"  ties (reporte aparte): {summary['ties']['n_pairs']:,}")
    log("=" * 58)


if __name__ == "__main__":
    main()
