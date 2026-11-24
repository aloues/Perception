"""
Genera los tres protocolos de particion (secciones 22, 23 y 24).

  paper_protocol : split aleatorio de PARES 65/5/30. Comparable con la
                   literatura, pero una misma imagen puede aparecer en train y
                   en test. NO es evidencia suficiente de generalizacion.

  image_disjoint : se particionan las IMAGENES 65/5/30 y solo se conserva un par
                   si sus DOS extremos caen en la misma particion. Ninguna imagen
                   de test/val aparece en train.

  city_holdout   : ciudades completas fuera del entrenamiento. Se generan DOS
                   definiciones de test, porque solo el 2.21% de los pares son
                   intra-ciudad:
                     strict : ambos extremos en la ciudad held-out
                              (limpio, pero mediana de ~122 pares por ciudad)
                     touch  : al menos un extremo en la ciudad held-out
                              (potencia estadistica; el otro extremo viene del
                              dominio de entrenamiento)
                   Se reportan ambas; ninguna se presenta como la unica valida.

Todas las particiones se derivan con una semilla fija y se guardan como listas
de indices sobre pairs_safe.csv, de modo que sean reproducibles y auditables.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from utils import MANIFESTS, SPLITS, log, set_seed

TRAIN_FRAC, VAL_FRAC = 0.65, 0.05

# image_disjoint necesita fracciones propias: al exigir que AMBOS extremos caigan
# en la misma particion, la retencion de pares es cuadratica en la fraccion de
# imagenes. Con 5% de imagenes en val solo sobrevive 0.05^2 = 0.25% de los pares
# (~1.1k), insuficiente para seleccionar modelo. Con 12% se obtiene ~1.4% (~6k).
IMG_TRAIN_FRAC, IMG_VAL_FRAC = 0.60, 0.12

# Ciudades held-out por defecto: diversidad geografica y tamano suficiente.
DEFAULT_HOLDOUT_CITIES = [
    "Tokyo",          # Asia oriental
    "Bangkok",        # Sudeste asiatico
    "Cape Town",      # Africa austral
    "Gaborone",       # Africa austral, muestra pequena
    "Rio De Janeiro", # Sudamerica
    "Santiago",       # Sudamerica
    "Moscow",         # Europa oriental
    "Barcelona",      # Europa occidental
    "Melbourne",      # Oceania
    "Houston",        # Norteamerica
]


def split_paper_protocol(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    idx = rng.permutation(len(df))
    n_train = int(len(df) * TRAIN_FRAC)
    n_val = int(len(df) * VAL_FRAC)
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train:n_train + n_val].tolist(),
        "test": idx[n_train + n_val:].tolist(),
    }


def split_image_disjoint(df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, dict]:
    images = pd.unique(pd.concat([df["left"], df["right"]], ignore_index=True))
    perm = rng.permutation(len(images))
    n_train = int(len(images) * IMG_TRAIN_FRAC)
    n_val = int(len(images) * IMG_VAL_FRAC)

    assign = {}
    for rank, pos in enumerate(perm):
        part = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
        assign[images[pos]] = part

    part_left = df["left"].map(assign)
    part_right = df["right"].map(assign)
    same = part_left == part_right

    splits = {
        p: df.index[same & (part_left == p)].tolist()
        for p in ("train", "val", "test")
    }
    stats = {
        "n_images": int(len(images)),
        "n_images_per_partition": {
            p: int(sum(1 for v in assign.values() if v == p))
            for p in ("train", "val", "test")
        },
        "n_pairs_total": int(len(df)),
        "n_pairs_kept": int(same.sum()),
        "n_pairs_discarded_cross_partition": int((~same).sum()),
        "fraction_kept": round(float(same.mean()), 6),
    }
    return splits, stats


def split_city_holdout(df: pd.DataFrame, holdout: list[str], rng: np.random.Generator):
    held = set(holdout)
    left_in = df["city_left"].isin(held)
    right_in = df["city_right"].isin(held)

    touches = left_in | right_in
    both = left_in & right_in

    # Entrenamiento: ningun extremo en una ciudad held-out.
    train_pool = df.index[~touches].to_numpy()
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * (VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)))
    val_idx = train_pool[perm[:n_val]]
    train_idx = train_pool[perm[n_val:]]

    splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test_strict": df.index[both].tolist(),
        "test_touch": df.index[touches].tolist(),
    }

    per_city = {}
    for city in holdout:
        in_city = (df["city_left"] == city) | (df["city_right"] == city)
        per_city[city] = {
            # Dentro de test_strict (ambos extremos en el dominio held-out),
            # los pares que involucran esta ciudad. Es la cifra a reportar por
            # ciudad: mantiene el aislamiento del dominio y conserva tamano.
            "n_in_test_strict": int((both & in_city).sum()),
            # Solo intra-ciudad: muy pocos pares, se reporta como referencia.
            "n_intra_city_only": int(
                ((df["city_left"] == city) & (df["city_right"] == city)).sum()
            ),
            "n_in_test_touch": int((touches & in_city).sum()),
        }

    stats = {
        "holdout_cities": holdout,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test_strict": int(both.sum()),
        "n_test_touch": int(touches.sum()),
        "per_city": per_city,
        "warning_strict": (
            "test_strict tiene muy pocos pares por ciudad (solo el 2.21% de los "
            "pares del dataset son intra-ciudad). Reportar SIEMPRE con intervalo "
            "de confianza y tamano de muestra."
        ),
        "warning_touch": (
            "En test_touch una de las dos imagenes puede pertenecer a una ciudad "
            "vista en entrenamiento. Mide transferencia relativa, no aislamiento "
            "total del dominio."
        ),
    }
    return splits, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-cities", nargs="*", default=DEFAULT_HOLDOUT_CITIES)
    args = ap.parse_args()

    set_seed(args.seed)
    SPLITS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    log(f"pares decisivos: {len(df):,}")

    report = {"seed": args.seed, "n_pairs": int(len(df)), "protocols": {}}

    rng = np.random.default_rng(args.seed)
    paper = split_paper_protocol(df, rng)
    json.dump(paper, open(SPLITS / "paper_protocol.json", "w"))
    report["protocols"]["paper_protocol"] = {
        "sizes": {k: len(v) for k, v in paper.items()},
        "warning": (
            "Split aleatorio de PARES: una misma imagen puede aparecer en train y "
            "test. Se usa para comparabilidad historica, no como unica evidencia "
            "de generalizacion."
        ),
    }
    log(f"paper_protocol  : { {k: len(v) for k, v in paper.items()} }")

    rng = np.random.default_rng(args.seed)
    disj, disj_stats = split_image_disjoint(df, rng)
    json.dump(disj, open(SPLITS / "image_disjoint.json", "w"))
    report["protocols"]["image_disjoint"] = {
        "sizes": {k: len(v) for k, v in disj.items()},
        **disj_stats,
    }
    log(f"image_disjoint  : { {k: len(v) for k, v in disj.items()} } "
        f"(conservados {disj_stats['fraction_kept']:.1%})")

    rng = np.random.default_rng(args.seed)
    city, city_stats = split_city_holdout(df, args.holdout_cities, rng)
    json.dump(city, open(SPLITS / "city_holdout.json", "w"))
    report["protocols"]["city_holdout"] = {
        "sizes": {k: len(v) for k, v in city.items()},
        **city_stats,
    }
    log(f"city_holdout    : { {k: len(v) for k, v in city.items()} }")

    (SPLITS / "splits_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"\nGuardado en {SPLITS}")


if __name__ == "__main__":
    main()
