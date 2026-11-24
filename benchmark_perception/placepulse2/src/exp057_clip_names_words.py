"""
EXP-057 — ¿Puede CLIP nombrar las palabras visuales sin inspección humana? (asesoría 05/11, A8: "que el clip
describa esto"). Diseño registrado en docs/EXPERIMENT_REGISTRY.md antes de ejecutar.

Para cada palabra k del vocabulario K100_s0 (EXP-044): las N imágenes de Place Pulse piloto (Atlanta, Berlín,
Santiago) donde la palabra es más frecuente en términos relativos → dirección CLIP = media de sus embeddings de imagen
− media de todas las imágenes piloto. Cada frase de un vocabulario fijo de 30 conceptos urbanos se puntúa por el
coseno entre su embedding de texto (CLIP ViT-B/32 laion2b, plantillas promediadas) y esa dirección.
Validación contra las etiquetas MANUALES de EXP-044/053 (17 palabras): acierto = alguna frase aceptable (fijada antes
de ejecutar en ACCEPT) está entre las 3 primeras de CLIP. Nulo: permutar las palabras (1,000 permutaciones).
Límite: CLIP ve la imagen entera, no el parche; la selección por frecuencia relativa concentra la palabra, no la aísla.

Uso:  python exp057_clip_names_words.py        Salidas: outputs/exp057/
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp044_patch_bovw import load_blocks
from utils import FEATURES, OUTPUTS, log, set_seed

SEED = 42
OUT = OUTPUTS / "exp057"
VOCAB = "K100_s0"
TOP_N = 40
TOP_K = 3
N_PERM = 1000
TEMPLATES = ("a street view photo of {}.", "a photo of {}.", "a Google Street View image of {}.")
PROMPTS = {
    "grass": "a grass verge next to the sidewalk", "garden": "a front garden with plants",
    "lowrise": "low-rise houses with tiled roofs", "trees": "trees and green foliage",
    "blankwall": "a blank wall without windows", "fence": "a tall fence or wall blocking the view",
    "cables": "overhead power lines and cables against the sky", "overcast": "a grey overcast sky",
    "clearsky": "a clear blue sky", "shopsigns": "shop fronts with commercial signs and awnings",
    "trucks": "trucks and freight vehicles", "worn": "cracked and worn asphalt road",
    "market": "street vendors and market stalls on the sidewalk", "shutters": "closed metal shutters and gates",
    "cars": "parked cars along the street", "highrise": "tall apartment buildings",
    "sidewalk": "a wide clean sidewalk", "graffiti": "graffiti on the walls",
    "trash": "trash and litter on the street", "pedestrians": "pedestrians walking on the street",
    "road": "an empty asphalt road", "construction": "a construction site", "parking": "a parking lot",
    "industrial": "warehouses and industrial buildings", "streetlight": "street lights and poles",
    "hedge": "a hedge along the sidewalk", "busy_road": "a wide road with heavy traffic",
    "glass": "modern glass office buildings", "old_facade": "old colonial building facades",
    "brick": "unfinished brick buildings",
}
# Etiqueta manual (EXP-044 inspección; EXP-053 inspección) → frases aceptables. FIJADO ANTES DE EJECUTAR.
MANUAL = {
    5: ("berma de césped y vereda", {"grass", "sidewalk", "hedge"}), 43: ("jardín frontal", {"garden", "hedge", "trees"}),
    42: ("techos de casas bajas", {"lowrise"}), 4: ("follaje", {"trees"}), 32: ("follaje", {"trees"}),
    82: ("muro o reja ciega", {"blankwall", "fence"}), 41: ("cables aéreos contra el cielo", {"cables"}),
    73: ("cables aéreos contra el cielo", {"cables"}), 17: ("cables aéreos contra el cielo", {"cables"}),
    76: ("cielo nublado con poca vegetación", {"overcast"}), 28: ("muro liso / fachada ciega", {"blankwall", "fence"}),
    2: ("fachada comercial con letreros", {"shopsigns"}), 44: ("camiones y vehículos de carga", {"trucks"}),
    55: ("calzada desgastada", {"worn", "road"}), 68: ("vereda con actividad / puestos (candidato)", {"market", "pedestrians"}),
    46: ("persianas metálicas / portones cerrados", {"shutters"}), 85: ("cielo nublado", {"overcast"}),
}


def text_embeddings() -> np.ndarray:
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    out = []
    with torch.no_grad():
        for p in PROMPTS.values():
            t = model.encode_text(tok([tpl.format(p) for tpl in TEMPLATES])).float()
            t = torch.nn.functional.normalize(t, dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(t, dim=0).numpy())
    return np.stack(out)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ids, B = load_blocks()
    H = B[f"hist_{VOCAB}"].astype(np.float64)
    H = H / H.sum(1, keepdims=True)
    idx = json.loads((FEATURES / "clip_vit_b32" / "index.json").read_text())["location_ids"]
    row = {lid: i for i, lid in enumerate(idx)}
    C = np.load(FEATURES / "clip_vit_b32" / "features.npy", mmap_mode="r")
    C = np.asarray(C[[row[i] for i in ids]], np.float64)
    rel = H / H.mean(0, keepdims=True)
    base = C.mean(0)
    T = text_embeddings()
    keys = list(PROMPTS)
    rows, Dirs = [], []
    for k in range(H.shape[1]):
        top = np.argsort(-rel[:, k])[:TOP_N]
        d = C[top].mean(0) - base
        d /= np.linalg.norm(d)
        Dirs.append(d)
        s = T @ d
        order = np.argsort(-s)[:TOP_K]
        rows.append({"word": k, "mean_freq": float(H[:, k].mean()),
                     **{f"clip_top{j + 1}": keys[o] for j, o in enumerate(order)},
                     **{f"clip_top{j + 1}_text": PROMPTS[keys[o]] for j, o in enumerate(order)},
                     **{f"clip_top{j + 1}_cos": float(s[o]) for j, o in enumerate(order)}})
    W = pd.DataFrame(rows)
    W.to_csv(OUT / "clip_names_all_words.csv", index=False, float_format="%.4f")
    D = np.stack(Dirs)
    topk = np.argsort(-(D @ T.T), axis=1)[:, :TOP_K]
    words = np.array(list(MANUAL))

    def hits(word_order):
        return np.array([bool({keys[o] for o in topk[w]} & MANUAL[m][1]) for w, m in zip(word_order, words)])

    h = hits(words)
    null = np.array([hits(rng.permutation(np.arange(H.shape[1]))[:len(words)]).mean() for _ in range(N_PERM)])
    V = pd.DataFrame({"word": words, "manual_label": [MANUAL[w][0] for w in words],
                      "clip_top3": ["; ".join(PROMPTS[keys[o]] for o in topk[w]) for w in words], "hit": h})
    V.to_csv(OUT / "validation_vs_manual.csv", index=False)
    summ = {"vocab": VOCAB, "top_n_images": TOP_N, "top_k": TOP_K, "n_words_validated": int(len(words)),
            "hit_rate": float(h.mean()), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value_perm": float((np.sum(null >= h.mean()) + 1) / (N_PERM + 1)), "n_prompts": len(PROMPTS)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(f"acierto CLIP vs etiqueta manual: {h.mean():.2f} ({h.sum()}/{len(h)}); nulo {null.mean():.2f} "
        f"(p95 {np.percentile(null, 95):.2f}); p = {summ['p_value_perm']:.4f}")
    log(V.to_string(index=False))


if __name__ == "__main__":
    main()
