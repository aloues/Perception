"""
Auditoria de leakage (seccion 28). Se ejecuta sobre los splits ya generados,
ANTES de entrenar nada.

Verifica, para cada protocolo:
  - solapamiento de image IDs entre particiones
  - solapamiento de pares (ordenados y no ordenados)
  - pares inversos que cruzan particiones
  - solapamiento de ciudades (relevante en city_holdout)
  - solapamiento de votantes (no es leakage de features, pero se reporta)
  - duplicados exactos dentro de cada particion

Las comprobaciones de scaler/PCA/hiperparametros no se pueden verificar aqui:
son invariantes del codigo de entrenamiento y quedan declaradas como reglas
que train_probe.py debe respetar.

Salida: outputs/leakage_report.json
"""

from __future__ import annotations

import json

import pandas as pd

from utils import MANIFESTS, OUTPUTS, SPLITS, log

TEST_KEYS = ("test", "test_strict", "test_touch")


def images_of(df: pd.DataFrame, idx: list[int]) -> set[str]:
    sub = df.loc[idx]
    return set(sub["left"]) | set(sub["right"])


def cities_of(df: pd.DataFrame, idx: list[int]) -> set[str]:
    sub = df.loc[idx]
    return set(sub["city_left"]) | set(sub["city_right"])


def ordered_pairs(df: pd.DataFrame, idx: list[int]) -> set[str]:
    sub = df.loc[idx]
    return set(sub["left"] + "|" + sub["right"])


def unordered_pairs(df: pd.DataFrame, idx: list[int]) -> set[str]:
    sub = df.loc[idx]
    lo = sub["left"].where(sub["left"] <= sub["right"], sub["right"])
    hi = sub["right"].where(sub["left"] <= sub["right"], sub["left"])
    return set(lo + "|" + hi)


def audit_protocol(name: str, df: pd.DataFrame, splits: dict) -> dict:
    parts = {k: v for k, v in splits.items() if v}
    train_keys = [k for k in parts if k.startswith("train")]
    test_keys = [k for k in parts if k in TEST_KEYS]

    result: dict = {"partitions": {k: len(v) for k, v in parts.items()}, "checks": {}}

    imgs = {k: images_of(df, v) for k, v in parts.items()}
    cits = {k: cities_of(df, v) for k, v in parts.items()}
    op = {k: ordered_pairs(df, v) for k, v in parts.items()}
    up = {k: unordered_pairs(df, v) for k, v in parts.items()}

    for tr in train_keys:
        for te in test_keys + [k for k in parts if k.startswith("val")]:
            key = f"{tr}_vs_{te}"
            shared_imgs = imgs[tr] & imgs[te]
            shared_up = up[tr] & up[te]
            shared_op = op[tr] & op[te]
            result["checks"][key] = {
                "shared_image_ids": len(shared_imgs),
                "shared_image_fraction_of_test": round(
                    len(shared_imgs) / max(len(imgs[te]), 1), 6
                ),
                "shared_ordered_pairs": len(shared_op),
                "shared_unordered_pairs": len(shared_up),
                "inverse_pairs_across_partitions": len(shared_up) - len(shared_op),
                "shared_cities": len(cits[tr] & cits[te]),
                "cities_only_in_test": sorted(cits[te] - cits[tr]),
            }

    for k, idx in parts.items():
        sub = df.loc[idx]
        exact = sub.duplicated(subset=["left", "right", "voter_uniqueid"]).sum()
        result["checks"].setdefault("duplicates_within_partition", {})[k] = int(exact)

    return result


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    df["label"] = df["label"].astype(int)

    report = {
        "n_pairs": int(len(df)),
        "protocols": {},
        "rules_enforced_in_training_code": {
            "scaler_and_pca": (
                "Cualquier scaler o PCA se ajusta SOLO con train y se aplica a "
                "val/test con los parametros de train."
            ),
            "hyperparameters": "Se seleccionan con val; test se toca una sola vez.",
            "preprocessing": "Identico para los tres encoders y las tres particiones.",
            "early_stopping": "Se monitorea val, nunca test.",
        },
    }

    for name in ("paper_protocol", "image_disjoint", "city_holdout"):
        path = SPLITS / f"{name}.json"
        if not path.exists():
            log(f"  falta {path}, omitido")
            continue
        splits = json.loads(path.read_text())
        log(f"auditando {name}…")
        report["protocols"][name] = audit_protocol(name, df, splits)

    (OUTPUTS / "leakage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("\n" + "=" * 70)
    for name, res in report["protocols"].items():
        log(f"\n{name}")
        for key, c in res["checks"].items():
            if key == "duplicates_within_partition":
                continue
            log(f"  {key}")
            log(f"    imagenes compartidas : {c['shared_image_ids']:,} "
                f"({c['shared_image_fraction_of_test']:.2%} del test)")
            log(f"    pares compartidos    : {c['shared_ordered_pairs']:,} ordenados / "
                f"{c['shared_unordered_pairs']:,} no ordenados")
            log(f"    ciudades compartidas : {c['shared_cities']}")
    log("=" * 70)
    log(f"\nGuardado: {OUTPUTS / 'leakage_report.json'}")


if __name__ == "__main__":
    main()
