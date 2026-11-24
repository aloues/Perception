"""
EXP-036 — Conceptos v2: prompt ensembling de CLIP (Radford et al. 2021, §3.1.4).

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, programa v2, 2026-10-09).

EXP-031 medía cada concepto con UN par de frases. CLIP es sensible a la redacción del prompt; promediar en el
espacio de embeddings de texto varias paráfrasis y plantillas de contexto reduce esa varianza. Aquí cada polo se
mide con 4 paráfrasis × 3 plantillas (12 textos) promediados. El control aleatorio usa la misma construcción.

Uso:  python exp036_concepts_v2.py        Salidas: outputs/exp036/ (concept_scores_std.csv.gz, comparaciones)
Luego: python exp032_concepts_neighborhood.py --concepts ../outputs/exp036/concept_scores_std.csv.gz --out exp036_barrio
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from exp026_additive_context import load_images
from exp031_concept_bottleneck import EXPECTED, NOUNS, concept_scores, standardize
from probe_utils import load_pairs, load_split, train_scores, votes
from utils import FEATURES, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp036"
N_BOOT = 2000
N_RANDOM = 10
TEMPLATES = ["a street view photo of {}.", "a Google Street View image of {}.", "a photo of {}."]

# (positivo, negativo): 4 paráfrasis por polo, fieles a la definición operacional del codebook
PHRASES = {
    "vegetation": (["a street with a lot of green vegetation and plants", "a leafy street with bushes and grass",
                    "a green street with gardens and shrubs", "a street surrounded by lush greenery"],
                   ["a street with no vegetation, only concrete and asphalt", "a barren street without plants or grass",
                    "a grey street with no greenery at all", "a paved street with no trees or bushes"]),
    "tree_canopy": (["a street lined with tall trees", "a street shaded by a canopy of large trees",
                     "an avenue with rows of mature trees", "a road under big leafy trees"],
                    ["a street without any trees", "a road with no trees along it",
                     "a treeless street", "a street with no tree canopy overhead"]),
    "visual_openness": (["a wide open street with a lot of visible sky", "a broad open space with a clear view of the sky",
                         "a spacious street with low buildings and open sky", "an open wide road with distant views"],
                        ["a narrow enclosed street with little visible sky", "a cramped alley between tall walls",
                         "a tight narrow passage with no view of the sky", "a closed-in street hemmed by high buildings"]),
    "graffiti": (["a wall covered with graffiti tags", "buildings with spray-painted graffiti",
                  "a street full of vandalized walls with graffiti", "a shutter painted with graffiti scribbles"],
                 ["a clean wall without graffiti", "buildings with clean unpainted walls",
                  "a street with no graffiti or tags", "a well kept wall with no vandalism"]),
    "litter": (["a street with garbage and litter on the ground", "a sidewalk with trash and rubbish scattered around",
                "a dirty street with piles of garbage bags", "a road with litter and debris on the ground"],
               ["a clean street without any garbage", "a tidy sidewalk with no litter",
                "a spotless street with no trash", "a well swept road free of rubbish"]),
    "facade_condition": (["buildings with damaged, deteriorated facades", "run-down houses with peeling paint and cracks",
                          "dilapidated buildings in poor condition", "decaying facades with broken walls"],
                         ["buildings with well maintained facades", "neat houses with freshly painted walls",
                          "buildings in good condition", "well kept facades in good repair"]),
    "security_bars": (["windows protected with metal security bars", "houses with iron grilles over the windows",
                       "doors and windows covered by metal bars", "a building with barred windows"],
                      ["windows without any bars or grilles", "houses with open unbarred windows",
                       "a building with plain glass windows and no grilles", "a facade with no security bars"]),
    "blank_wall": (["a long blank wall with no windows or doors", "a solid windowless wall along the street",
                    "a high featureless wall facing the sidewalk", "a street bordered by a long blank fence wall"],
                   ["a building front with many windows and doors", "a facade with many windows facing the street",
                    "a street lined with doors and shop windows", "buildings with entrances and windows at street level"]),
    "sidewalk": (["a street with a proper paved sidewalk", "a road with wide pedestrian sidewalks",
                  "a street with a clear footpath for pedestrians", "a well built sidewalk next to the road"],
                 ["a street without any sidewalk", "a road with no footpath for pedestrians",
                  "a street where people must walk on the road", "a road shoulder with no pavement for walking"]),
    "street_lighting": (["a street with street lamps and light posts", "a road lined with street lights",
                         "a street with lamp posts along the sidewalk", "public lighting poles along a street"],
                        ["a street without any street lamps", "a road with no light posts",
                         "a street with no public lighting", "a dark road without lamp posts"]),
    "pedestrians": (["a street with people walking", "a sidewalk with many pedestrians",
                     "a busy street with people on foot", "people walking along a street"],
                    ["an empty street with nobody", "a deserted street with no people",
                     "a street with no pedestrians at all", "an empty sidewalk without anyone"]),
    "vehicles": (["a street with cars and vehicles", "a road with traffic and parked cars",
                  "a street full of cars", "vehicles driving and parked along a street"],
                 ["a street without any vehicles", "an empty road with no cars",
                  "a street with no traffic", "a road with no parked cars"]),
    "commercial_activity": (["a street with shops and storefronts", "a commercial street with stores and signs",
                             "a street with restaurants and small businesses", "a shopping street with open shops"],
                            ["a purely residential street with no shops", "a street with only houses and no stores",
                             "a quiet residential road without businesses", "a neighborhood street with homes only"]),
    "vacant_lot": (["an empty vacant lot with no building", "an abandoned empty plot of land",
                    "a vacant lot with weeds and rubble", "an unbuilt empty parcel next to the street"],
                   ["a fully built up street", "a street with continuous buildings",
                    "a densely built street with no empty lots", "a block with buildings on every plot"]),
    "street_signage": (["a street with traffic signs", "a road with many road signs",
                        "traffic signs and signals along a street", "a street with direction and warning signs"],
                       ["a street without traffic signs", "a road with no signs",
                        "a street with no road signage", "a road without any traffic signals or signs"]),
    "street_furniture": (["a street with benches and street furniture", "a sidewalk with benches and bins",
                          "a street with public benches and planters", "urban furniture like benches along a street"],
                         ["a street with no benches or street furniture", "a bare sidewalk with no benches",
                          "a street without public seating", "a sidewalk with no urban furniture"]),
    "sightline_obstruction": (["a street where the view ahead is blocked by walls, hedges or large vehicles",
                               "a street with obstacles blocking the line of sight", "a view obstructed by high hedges and walls",
                               "a corner where trucks and walls hide what is ahead"],
                              ["a street with a clear and unobstructed view ahead", "a street with a long clear line of sight",
                               "an open view down the street with nothing blocking it", "a straight road with good visibility"]),
}


def ensemble_text(poles: list[tuple[list[str], list[str]]]):
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    T = []
    with torch.no_grad():
        for pos, neg in poles:
            pair = []
            for phrases in (pos, neg):
                e = model.encode_text(tok([tp.format(p) for p in phrases for tp in TEMPLATES])).float()
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                pair.append(torch.nn.functional.normalize(e, dim=-1).numpy())
            T.append(pair)
        scale = float(model.logit_scale.exp())
    return np.array(T), scale


def correct(s: np.ndarray, ids: list[str], df: pd.DataFrame) -> np.ndarray:
    sc = pd.Series(s, index=ids)
    return ((sc.reindex(df["left"]).to_numpy() - sc.reindex(df["right"]).to_numpy()) > 0).astype(int) == df["label"].to_numpy()


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, _ = load_images("dinov2_vit_b14")
    ids = man["location_id"].tolist()
    cidx = json.loads((FEATURES / "clip_vit_b32" / "index.json").read_text(encoding="utf-8"))["location_ids"]
    pos = {lid: i for i, lid in enumerate(cidx)}
    C = np.load(FEATURES / "clip_vit_b32" / "features.npy")[[pos[i] for i in ids]]
    pairs = load_pairs()
    sp, ch = load_split("image_disjoint"), load_split("city_holdout")
    te = pairs.iloc[sp["test"]]
    tr_ids = set(votes(pairs, sp["train"])["id"])
    train_rows = np.array([i in tr_ids for i in ids])
    names = list(PHRASES)

    T, scale = ensemble_text([PHRASES[c] for c in names])
    F2 = standardize(concept_scores(C, T, scale), train_rows)
    pd.DataFrame(F2, columns=names).assign(location_id=ids).to_csv(OUT / "concept_scores_std.csv.gz", index=False,
                                                                   float_format="%.4f")
    F1 = pd.read_csv(OUTPUTS / "exp031" / "concept_scores_std.csv.gz").set_index("location_id").reindex(ids)[names].to_numpy(np.float32)

    rows = []
    s2, i2 = train_scores(F2, ids, pairs, sp, "test")
    s1, i1 = train_scores(F1, ids, pairs, sp, "test")
    c2, c1 = correct(s2, ids, te), correct(s1, ids, te)
    rows.append(dict(variant="conceptos_v2_ensemble", protocol="image_disjoint", accuracy=c2.mean()))
    rows.append(dict(variant="conceptos_v1", protocol="image_disjoint", accuracy=c1.mean()))
    _, ich = train_scores(F2, ids, pairs, ch, "test_strict")
    rows.append(dict(variant="conceptos_v2_ensemble", protocol="city_holdout", accuracy=ich["metrics"]["pairwise_accuracy"]))
    rand = []
    for r in range(N_RANDOM):
        nn_ = rng.choice(NOUNS, size=(len(names), 8), replace=True)
        Tr, _ = ensemble_text([([f"a {a}", f"a {b}", f"a big {a}", f"a small {b}"], [f"a {c}", f"a {d}", f"a big {c}", f"a small {d}"])
                               for a, b, c, d, *_ in nn_])
        Fr = standardize(concept_scores(C, Tr, scale), train_rows)
        sr, _ = train_scores(Fr, ids, pairs, sp, "test")
        rand.append(correct(sr, ids, te))
        rows.append(dict(variant=f"aleatorio_v2_{r}", protocol="image_disjoint", accuracy=rand[-1].mean()))
        log(f"  aleatorio {r}: {rand[-1].mean():.4f}")
    best_rand = max(rand, key=lambda c: c.mean())

    def paired(a, b):
        n = len(a)
        bs = np.array([a[ix].mean() - b[ix].mean() for ix in (rng.integers(0, n, n) for _ in range(N_BOOT))])
        return dict(delta=a.mean() - b.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5))

    comps = pd.DataFrame([dict(comparison="v2 - v1", **paired(c2, c1)),
                          dict(comparison="v2 - mejor aleatorio v2", **paired(c2, best_rand))])
    acc = pd.DataFrame(rows)
    log(acc.round(4).to_string(index=False))
    log(comps.round(4).to_string(index=False))
    acc.to_csv(OUT / "accuracy.csv", index=False, float_format="%.5f")
    comps.to_csv(OUT / "paired.csv", index=False, float_format="%.5f")

    w = pd.DataFrame({"concept": names, "weight_v2": i2["weights"], "weight_v1": i1["weights"],
                      "expected_sign": [EXPECTED.get(n, 0) for n in names]})
    d = w[w["expected_sign"] != 0]
    ok2 = int((np.sign(d["weight_v2"]) == d["expected_sign"]).sum())
    w.to_csv(OUT / "weights.csv", index=False, float_format="%.4f")
    log(w.round(4).to_string(index=False))
    margin_v1 = json.loads((OUTPUTS / "exp031" / "summary.json").read_text())
    summ = dict(acc_v2=float(c2.mean()), acc_v1=float(c1.mean()), acc_v2_city_holdout=ich["metrics"]["pairwise_accuracy"],
                best_random_v2=float(best_rand.mean()), margin_v2=float(c2.mean() - best_rand.mean()),
                margin_v1=float(margin_v1["acc_concepts"] - margin_v1["acc_random_max"]),
                signs_ok_v2=ok2, signs_total=int(len(d)),
                sign_test_p=float(binomtest(ok2, len(d), 0.5, alternative="greater").pvalue),
                seconds=round(time.perf_counter() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
