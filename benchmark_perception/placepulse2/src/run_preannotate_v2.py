"""
PRE-ANOTACION v2 (YOLO-World + CLIP) con prompts mejorados. NO es ground truth.

Cambios frente al PoC v1 (segun recomendacion metodologica):
- Noun phrases independientes y cortas para YOLO-World (evita "window with metal bars").
- Ensembles contrastivos en CLIP (varios prompts positivos y negativos por concepto).
- No se fijan umbrales: se guardan scores crudos; el umbral se elegira con anotacion humana.
- Grounding DINO NO se ejecuta aqui (17.6 s/img en CPU): se reserva para referring
  expressions sobre subconjuntos con desacuerdo.

Entradas: concept_validation/images (450) y poc/images (72)
Salidas : concept_validation/preannotations/preann_v2_yoloworld.csv, preann_v2_clip.csv
          poc/poc_v2_prompts_{yoloworld,clip}.csv
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from run_poc_concepts import CLIP_PROMPTS
from utils import OUTPUTS, log, set_seed

warnings.filterwarnings("ignore")
CV, POC = OUTPUTS / "concept_validation", OUTPUTS / "poc"

YOLO_PHRASES = {
    "graffiti": ["graffiti", "spray-painted graffiti", "painted tag", "graffiti on wall"],
    "litter": ["trash bag", "garbage pile", "scattered litter", "discarded cardboard", "waste on sidewalk"],
    "commercial_activity": ["storefront", "shop entrance", "display window", "shop sign", "market stall", "street vendor"],
    "security_bars": ["window bars", "barred window", "metal grille", "security grille", "metal gate"],
    "tree_canopy": ["tree"],
    "pedestrians": ["person"],
    "vehicles": ["car", "bus", "truck", "motorcycle"],
    "lighting_infrastructure": ["street lamp", "lamp post"],
    "street_signage": ["traffic sign", "street sign"],
    "street_furniture": ["bench", "bus stop"],
}

CLIP_ENS = {   # concepto -> (positivos, negativos)
    "graffiti": (["a street wall covered with graffiti", "spray-painted tags on an urban wall"],
                 ["a clean wall", "a decorative mural"]),
    "litter": (["a street with visible litter on the ground", "garbage accumulated on a sidewalk"],
               ["a clean sidewalk", "a normal garbage container"]),
}
for k, (p, n) in CLIP_PROMPTS.items():
    key = {"vegetation": "greenery", "street_lighting": "lighting_infrastructure",
           "blank_wall": "fortification_transparency"}.get(k, k)
    CLIP_ENS.setdefault(key, ([p], [n]))


def run_yolo(paths, tag):
    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")
    phrases, owner = [], []
    for c, ps in YOLO_PHRASES.items():
        for p in ps:
            phrases.append(p); owner.append(c)
    model.set_classes(phrases)
    rows = []
    for p in paths:
        r = model.predict(str(p), conf=0.05, verbose=False, device="cpu")[0]
        cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.array([], int)
        cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
        xy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        for c in YOLO_PHRASES:
            m = np.array([owner[i] == c for i in cls], bool) if len(cls) else np.zeros(0, bool)
            best, box, ph, n = 0.0, "", "", int(m.sum())
            if m.any():
                j = np.where(m)[0][cf[m].argmax()]
                best, box, ph = float(cf[j]), ",".join(str(int(v)) for v in xy[j]), phrases[cls[j]]
            rows.append(dict(image_id=p.stem, concept=c, tool="yoloworld_v2", score=round(best, 4),
                             n_boxes=n, best_phrase=ph, bbox=box, preannotation_only=True))
    return pd.DataFrame(rows)


def run_clip(paths):
    import open_clip, torch
    torch.set_num_threads(3)
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval(); tok = open_clip.get_tokenizer("ViT-B-32")
    enc = {}
    with torch.no_grad():
        for c, (pos, neg) in CLIP_ENS.items():
            t = model.encode_text(tok(pos + neg)); t = t / t.norm(dim=-1, keepdim=True)
            enc[c] = (t[:len(pos)], t[len(pos):])
    rows = []
    for p in paths:
        with Image.open(p) as im:
            x = pre(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            for c, (tp, tn) in enc.items():
                sp = (100 * f @ tp.T).mean(); sn = (100 * f @ tn.T).mean()
                prob = float(torch.softmax(torch.stack([sp, sn]), 0)[0])
                rows.append(dict(image_id=p.stem, concept=c, tool="clip_ens_v2", score=round(prob, 4),
                                 preannotation_only=True))
    return pd.DataFrame(rows)


def main():
    set_seed(42)
    import torch; torch.set_num_threads(3)
    for name, folder, outy, outc in [
        ("poc", POC / "images", POC / "poc_v2_prompts_yoloworld.csv", POC / "poc_v2_prompts_clip.csv"),
        ("annotation", CV / "images", CV / "preannotations" / "preann_v2_yoloworld.csv",
         CV / "preannotations" / "preann_v2_clip.csv")]:
        paths = sorted(Path(folder).glob("*.jpg"))
        log(f"{name}: {len(paths)} imagenes")
        t0 = time.perf_counter(); run_yolo(paths, name).to_csv(outy, index=False)
        log(f"  yolo-world {time.perf_counter()-t0:.0f}s")
        t0 = time.perf_counter(); run_clip(paths).to_csv(outc, index=False)
        log(f"  clip {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
