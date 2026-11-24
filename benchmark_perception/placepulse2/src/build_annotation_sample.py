"""
Muestra de VALIDACION DE MEDICION de conceptos (nivel 2 del codebook).

- MAIN (300): 30 ciudades elegidas al azar (seed 42) entre las de >=40 imagenes, 10 imagenes por
  ciudad, repartidas por tercil de win-rate (se usa SOLO para diversificar; no es etiqueta).
- STRESS (150): de un pool aleatorio de 4,000 imagenes (disjunto de MAIN y del PoC) se toman los
  extremos de imagen: 50 mas oscuras, 50 menos nitidas (varianza del laplaciano), 50 de mayor
  contraste/saturacion de brillo. Son casos DIFICILES a proposito.
- Disjunto de las 72 imagenes del PoC (para no evaluar sobre lo usado para ajustar prompts).
- LIMA: BLOQUEADO — las imagenes de Lima no estan disponibles localmente (solo embeddings).
  Cuando existan, anadir con split='lima'.

Salidas en outputs/concept_validation/: annotation_sample.csv, images/, sample_summary.json
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from utils import MANIFESTS, OUTPUTS, ZIP_PATH, log, set_seed

OUT = OUTPUTS / "concept_validation"
SEED = 42
MIN_CMP = 4


def stats(im: Image.Image):
    g = np.asarray(im.convert("L").resize((256, 192)), dtype=np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(g.mean()), float(lap.var()), float(g.std())


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for sub in ("concept_definitions", "preannotations", "human_annotations", "metrics"):
        (OUT / sub).mkdir(exist_ok=True)

    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    pairs["label"] = pairs["label"].astype(int)
    win, show = {}, {}
    for l, r, y in zip(pairs["left"], pairs["right"], pairs["label"]):
        show[l] = show.get(l, 0) + 1; show[r] = show.get(r, 0) + 1
        w = l if y == 1 else r
        win[w] = win.get(w, 0) + 1
    man["n_cmp"] = man["location_id"].map(show).fillna(0).astype(int)
    man["win_rate"] = man["location_id"].map(lambda i: win.get(i, 0) / max(show.get(i, 0), 1))
    man = man[man.n_cmp >= MIN_CMP]
    poc = set(pd.read_csv(OUTPUTS / "poc" / "poc_sample.csv", dtype=str)["location_id"])
    man = man[~man["location_id"].isin(poc)]

    counts = man["city"].value_counts()
    eligible = sorted(counts[counts >= 40].index)
    cities = sorted(rng.choice(eligible, size=min(30, len(eligible)), replace=False).tolist())
    rows = []
    for c in cities:
        sub = man[man.city == c].copy()
        sub["stratum"] = pd.qcut(sub.win_rate.rank(method="first"), 3, labels=["baja", "media", "alta"])
        take = {"baja": 3, "media": 4, "alta": 3}
        for s, g in sub.groupby("stratum", observed=True):
            idx = rng.choice(len(g), size=min(take[s], len(g)), replace=False)
            for _, r in g.iloc[idx].iterrows():
                rows.append(dict(image_id=r.location_id, split="main", city=c, stratum=str(s),
                                 stress_reason="", image_path=r.image_path))
    main = pd.DataFrame(rows)
    log(f"main: {len(main)} imagenes, {main.city.nunique()} ciudades")

    rest = man[~man.location_id.isin(set(main.image_id))]
    pool = rest.iloc[rng.choice(len(rest), size=4000, replace=False)]
    recs = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for _, r in pool.iterrows():
            with Image.open(io.BytesIO(zf.read(r.image_path))) as im:
                b, sharp, con = stats(im)
            recs.append((r.location_id, r.city, r.image_path, b, sharp, con))
    P = pd.DataFrame(recs, columns=["image_id", "city", "image_path", "bright", "sharp", "contrast"])
    dark = P.nsmallest(50, "bright"); used = set(dark.image_id)
    blur = P[~P.image_id.isin(used)].nsmallest(50, "sharp"); used |= set(blur.image_id)
    hic = P[~P.image_id.isin(used)].nlargest(50, "contrast")
    stress = pd.concat([dark.assign(stress_reason="oscura"), blur.assign(stress_reason="baja_nitidez"),
                        hic.assign(stress_reason="alto_contraste")])
    stress = stress.assign(split="stress", stratum="")[["image_id", "split", "city", "stratum", "stress_reason", "image_path"]]

    S = pd.concat([main, stress], ignore_index=True)
    # 100 imagenes MAIN con 3er anotador (conceptos subjetivos) — subconjunto aleatorio fijo
    tri = set(rng.choice(main.image_id.to_numpy(), size=100, replace=False))
    S["third_annotator_subset"] = S.image_id.isin(tri)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for _, r in S.iterrows():
            dst = OUT / "images" / f"{r.image_id}.jpg"
            if not dst.exists():
                with Image.open(io.BytesIO(zf.read(r.image_path))) as im:
                    im.convert("RGB").save(dst, quality=92)
    S.drop(columns=["image_path"]).to_csv(OUT / "annotation_sample.csv", index=False)
    (OUT / "sample_summary.json").write_text(json.dumps(dict(
        seed=SEED, n_main=int((S.split == "main").sum()), n_stress=int((S.split == "stress").sum()),
        cities=cities, stress_breakdown=stress.stress_reason.value_counts().to_dict(),
        third_annotator_subset=len(tri), lima="BLOQUEADO: imagenes de Lima no disponibles localmente",
        disjoint_from_poc=True), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
