"""
Construye el subconjunto reproducible del SMOKE TEST de 3 encoders.

Motivacion (sesion 3, secciones 7-8): la extraccion completa de DINOv2 en CPU
tiene una ETA de ~4.3 h y ya fallo una vez por corrupcion de heap. En lugar de
consumir el presupuesto de tiempo, se define un subconjunto de imagenes que:

  - es identico para los tres encoders (OURS / ImageNet / DINOv2);
  - es reproducible (seed fija + criterio deterministico);
  - conserva las 56 ciudades, estratificado proporcionalmente;
  - conserva la estructura de los tres protocolos de validacion, porque los
    splits se RESTRINGEN a los pares del subconjunto en lugar de re-sortearse
    (un subconjunto de conjuntos de imagenes disjuntos sigue siendo disjunto,
    y las ciudades holdout siguen siendo holdout).

Criterio de seleccion: dentro de cada ciudad se toman las imagenes con mayor
grado en el grafo de comparaciones (numero de pares en que aparecen). No es un
sesgo de contenido: el grado mide cuantas veces el crowdsourcing expuso esa
imagen, y maximiza los pares utilizables por imagen extraida. Se documenta
explicitamente como parte del diseno del smoke test.

Salidas:
    data/smoke/subset_images.txt
    data/smoke/subset_summary.json
    data/smoke/splits/<protocol>.json     (indices de pares restringidos)
"""

from __future__ import annotations

import collections
import json

import pandas as pd

from utils import MANIFESTS, PKG_ROOT, SPLITS, log, set_seed

TARGET_N_IMAGES = 20_000
SEED = 42
OUT = PKG_ROOT / "data" / "smoke"


def main() -> None:
    set_seed(SEED)
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"].reset_index(drop=True)
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)

    deg = collections.Counter(pairs["left"])
    deg.update(pairs["right"])
    man["deg"] = man["location_id"].map(deg).fillna(0).astype(int)

    frac = TARGET_N_IMAGES / len(man)
    chosen = []
    for city, g in man.groupby("city", sort=True):
        k = max(1, int(round(len(g) * frac)))
        # nlargest es deterministico: desempata por orden del manifiesto
        chosen.append(g.sort_values(["deg", "location_id"], ascending=[False, True]).head(k))
    sub = pd.concat(chosen).sort_values("location_id").reset_index(drop=True)
    S = set(sub["location_id"])
    log(f"subconjunto: {len(S):,} imagenes en {sub['city'].nunique()} ciudades")

    keep = pairs["left"].isin(S) & pairs["right"].isin(S)
    kept_idx = set(pairs.index[keep].tolist())
    log(f"pares retenidos: {len(kept_idx):,} de {len(pairs):,} ({len(kept_idx)/len(pairs):.1%})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits").mkdir(exist_ok=True)
    (OUT / "subset_images.txt").write_text("\n".join(sub["location_id"]), encoding="utf-8")

    summary = {
        "seed": SEED,
        "target_n_images": TARGET_N_IMAGES,
        "n_images": len(S),
        "n_cities": int(sub["city"].nunique()),
        "selection_rule": "por ciudad, top-k por grado en el grafo de pares "
                          "(k proporcional al tamano de la ciudad); desempate "
                          "deterministico por location_id",
        "n_pairs_kept": len(kept_idx),
        "n_pairs_total": len(pairs),
        "fraction_pairs_kept": round(len(kept_idx) / len(pairs), 6),
        "label_balance_kept": float(pairs.loc[keep, "label"].astype(int).mean()),
        "per_city_images": sub["city"].value_counts().to_dict(),
        "protocols": {},
        "status": "PRELIMINARY / SMOKE TEST",
    }

    for proto in ("paper_protocol", "image_disjoint", "city_holdout"):
        full = json.loads((SPLITS / f"{proto}.json").read_text())
        restricted = {k: [i for i in v if i in kept_idx] for k, v in full.items()
                      if isinstance(v, list)}
        (OUT / "splits" / f"{proto}.json").write_text(json.dumps(restricted),
                                                      encoding="utf-8")
        sizes = {k: len(v) for k, v in restricted.items()}
        summary["protocols"][proto] = {
            "sizes": sizes,
            "sizes_full": {k: len(v) for k, v in full.items() if isinstance(v, list)},
        }
        log(f"  {proto}: {sizes}")

    (OUT / "subset_summary.json").write_text(json.dumps(summary, indent=2,
                                                       ensure_ascii=False),
                                             encoding="utf-8")
    log(f"\nGuardado en {OUT}")


if __name__ == "__main__":
    main()
