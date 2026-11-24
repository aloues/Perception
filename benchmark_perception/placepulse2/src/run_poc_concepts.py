"""
FASE D — Proof-of-concept semantico: ¿se pueden MEDIR los conceptos del codebook?

Pregunta que responde
---------------------
Antes de lanzar ninguna extraccion masiva de conceptos hay que saber si los
conceptos de `codebook_v0` son medibles de forma REPRODUCIBLE sobre street view.
Este script compara tres herramientas candidatas sobre la misma muestra
reproducible y reporta cuatro propiedades que si se pueden evaluar sin
anotacion humana:

  1. DETERMINISMO      misma imagen, misma salida al repetir la ejecucion.
  2. TASA DE DISPARO   ¿el concepto llega a activarse alguna vez? Un concepto
                       que nunca dispara (o que dispara siempre) no aporta
                       informacion al bottleneck.
  3. DISCRIMINACION    ¿la puntuacion varia entre imagenes, o es constante?
  4. VALIDEZ CONVERGENTE  para los conceptos que dos herramientas independientes
                       pueden medir, ¿coinciden? (correlacion de Spearman).

Lo que este PoC NO puede responder
----------------------------------
Sin anotacion humana de los conceptos sobre estas imagenes no hay precision ni
recall. La validez convergente y la inspeccion visual son evidencia debil de
validez aparente, no una medida de exactitud. Queda registrado como el
siguiente paso necesario: anotar manualmente una submuestra.

Herramientas
------------
  CLIP (open_clip ViT-B-32)  : presencia zero-shot por pares de prompts
                               contrastivos -> probabilidad [0,1]
  YOLO-World (ultralytics)   : deteccion open-vocabulary -> cajas + confianza
  Grounding DINO (HF, tiny)  : deteccion open-vocabulary guiada por texto

Salidas:
    outputs/poc/poc_detections.csv     un registro por (imagen, concepto, herramienta)
    outputs/poc/poc_metrics.json       determinismo, disparo, discriminacion, acuerdo
    outputs/poc/POC_REPORT.md

Uso:
    python run_poc_concepts.py --tools clip,yoloworld,gdino
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils import OUTPUTS, log, set_seed

warnings.filterwarnings("ignore")

POC = OUTPUTS / "poc"
CODEBOOK = OUTPUTS / "codebook" / "codebook_v0.json"
SEED = 42

# Prompts contrastivos para CLIP: (positivo, negativo). La decision de usar
# pares y no similitud cruda es deliberada: la similitud absoluta de CLIP no es
# comparable entre conceptos, la probabilidad de un par contrastivo si.
CLIP_PROMPTS = {
    "vegetation":        ("a street with a lot of green vegetation and plants",
                          "a street with no vegetation, only concrete and asphalt"),
    "tree_canopy":       ("a street lined with tall trees",
                          "a street without any trees"),
    "visual_openness":   ("a wide open street with a lot of visible sky",
                          "a narrow enclosed street with little visible sky"),
    "graffiti":          ("a wall covered with graffiti tags",
                          "a clean wall without graffiti"),
    "litter":            ("a street with garbage and litter on the ground",
                          "a clean street without any garbage"),
    "facade_condition":  ("buildings with damaged, deteriorated facades",
                          "buildings with well maintained facades"),
    "security_bars":     ("windows protected with metal security bars",
                          "windows without any bars or grilles"),
    "blank_wall":        ("a long blank wall with no windows or doors",
                          "a building front with many windows and doors"),
    "sidewalk":          ("a street with a proper paved sidewalk",
                          "a street without any sidewalk"),
    "street_lighting":   ("a street with street lamps and light posts",
                          "a street without any street lamps"),
    "pedestrians":       ("a street with people walking",
                          "an empty street with nobody"),
    "vehicles":          ("a street with cars and vehicles",
                          "a street without any vehicles"),
    "commercial_activity": ("a street with shops and storefronts",
                            "a purely residential street with no shops"),
    "vacant_lot":        ("an empty vacant lot with no building",
                          "a fully built up street"),
    "street_signage":    ("a street with traffic signs",
                          "a street without traffic signs"),
    "street_furniture":  ("a street with benches and street furniture",
                          "a street with no benches or street furniture"),
}

# Prompts de deteccion (objetos localizables). Los conceptos de escena
# (apertura visual, estado de fachada) no son detectables como objeto y se
# marcan como no aplicables para los detectores: eso tambien es un resultado.
DETECT_PROMPTS = {
    "tree_canopy": "tree",
    "graffiti": "graffiti",
    "litter": "garbage bag",
    "security_bars": "window with metal bars",
    "pedestrians": "person",
    "vehicles": "car",
    "commercial_activity": "storefront sign",
    "street_lighting": "street lamp",
    "street_signage": "traffic sign",
    "street_furniture": "bench",
}
NOT_DETECTABLE = ["vegetation", "visual_openness", "sightline_obstruction",
                  "facade_condition", "blank_wall", "sidewalk", "vacant_lot"]


# ── CLIP ──────────────────────────────────────────────────────────────────
def run_clip(paths: list[Path]) -> list[dict]:
    import open_clip
    import torch

    torch.set_num_threads(4)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    concepts = list(CLIP_PROMPTS)
    texts = [t for c in concepts for t in CLIP_PROMPTS[c]]
    with torch.no_grad():
        tf = model.encode_text(tok(texts))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    rows = []
    for p in paths:
        with Image.open(p) as im:
            x = preprocess(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = (100.0 * f @ tf.T).squeeze(0)
        for i, c in enumerate(concepts):
            pair = sims[[2 * i, 2 * i + 1]]
            prob = float(torch.softmax(pair, dim=0)[0])
            rows.append({"location_id": p.stem, "concept": c, "tool": "clip",
                         "score": round(prob, 6), "n_detections": np.nan,
                         "bbox": ""})
    return rows


# ── YOLO-World ────────────────────────────────────────────────────────────
def run_yoloworld(paths: list[Path], conf: float = 0.05) -> list[dict]:
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8s-worldv2.pt")
    concepts = list(DETECT_PROMPTS)
    model.set_classes([DETECT_PROMPTS[c] for c in concepts])

    rows = []
    for p in paths:
        res = model.predict(str(p), conf=conf, verbose=False)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.array([])
        cf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
        xy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        for i, c in enumerate(concepts):
            m = cls == i
            best = float(cf[m].max()) if m.any() else 0.0
            box = ""
            if m.any():
                b = xy[m][cf[m].argmax()]
                box = ",".join(str(int(v)) for v in b)
            rows.append({"location_id": p.stem, "concept": c, "tool": "yoloworld",
                         "score": round(best, 6), "n_detections": int(m.sum()),
                         "bbox": box})
    return rows


# ── Grounding DINO ────────────────────────────────────────────────────────
GDINO_RAW: list = []


def run_gdino(paths: list[Path], conf: float = 0.20) -> list[dict]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(4)
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()

    concepts = list(DETECT_PROMPTS)
    prompt = ". ".join(DETECT_PROMPTS[c] for c in concepts) + "."

    rows = []
    for p in paths:
        with Image.open(p) as im:
            img = im.convert("RGB")
            inputs = proc(images=img, text=prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=conf, text_threshold=conf,
                target_sizes=[img.size[::-1]])[0]
        labels = [str(t).lower() for t in res.get("text_labels", res.get("labels", []))]
        scores = res["scores"].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()
        GDINO_RAW.append({"location_id": p.stem, "labels": labels,
                          "scores": [round(float(x), 4) for x in scores],
                          "boxes": [[int(v) for v in b] for b in boxes]})
        for c in concepts:
            key = DETECT_PROMPTS[c].lower()
            # coincidencia EXACTA de frase (v1 usaba `lb in key`: "window" disparaba
            # security_bars y etiquetas parciales disparaban varios conceptos)
            m = np.array([lb == key for lb in labels], dtype=bool) \
                if labels else np.zeros(0, dtype=bool)
            best = float(scores[m].max()) if m.any() else 0.0
            box = ""
            if m.any():
                b = boxes[m][scores[m].argmax()]
                box = ",".join(str(int(v)) for v in b)
            rows.append({"location_id": p.stem, "concept": c, "tool": "gdino",
                         "score": round(best, 6), "n_detections": int(m.sum()),
                         "bbox": box})
    return rows


RUNNERS = {"clip": run_clip, "yoloworld": run_yoloworld, "gdino": run_gdino}


def compute_metrics(df, paths, timings, blocked):
    metrics = {"n_images": len(paths), "seed": SEED, "timings": timings,
               "blocked": blocked, "per_tool_concept": [], "agreement": []}

    for (tool, c), g in df.groupby(["tool", "concept"]):
        s = g["score"].to_numpy(float)
        thr = 0.5 if tool == "clip" else 0.05
        metrics["per_tool_concept"].append({
            "tool": tool, "concept": c,
            "fire_rate": round(float((s > thr).mean()), 4),
            "score_mean": round(float(s.mean()), 4),
            "score_std": round(float(s.std()), 4),
            "score_min": round(float(s.min()), 4),
            "score_max": round(float(s.max()), 4),
            "degenerate": bool(s.std() < 1e-3),
        })

    piv = df.pivot_table(index="location_id", columns=["tool", "concept"],
                         values="score")
    for c in DETECT_PROMPTS:
        cols = [t for t in ("clip", "yoloworld", "gdino") if (t, c) in piv.columns]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x = piv[(cols[i], c)]
                y = piv[(cols[j], c)]
                ok = x.notna() & y.notna()
                if ok.sum() < 5 or x[ok].std() < 1e-9 or y[ok].std() < 1e-9:
                    rho = float("nan")
                else:
                    rho = float(x[ok].rank().corr(y[ok].rank()))
                metrics["agreement"].append(
                    {"concept": c, "tool_a": cols[i], "tool_b": cols[j],
                     "spearman": None if np.isnan(rho) else round(rho, 4),
                     "n": int(ok.sum())})

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="clip,yoloworld,gdino")
    ap.add_argument("--determinism-n", type=int, default=8,
                    help="imagenes que se reprocesan para comprobar determinismo")
    args = ap.parse_args()

    set_seed(SEED)
    sample = pd.read_csv(POC / "poc_sample.csv", dtype=str)
    paths = [POC / "images" / f"{i}.jpg" for i in sample["location_id"]]
    paths = [p for p in paths if p.exists()]
    log(f"muestra: {len(paths)} imagenes")

    all_rows, timings, blocked = [], {}, {}
    for tool in args.tools.split(","):
        tool = tool.strip()
        if tool not in RUNNERS:
            continue
        log(f"\n=== {tool} ===")
        t0 = time.perf_counter()
        try:
            rows = RUNNERS[tool](paths)
        except Exception as e:  # noqa: BLE001
            log(f"  BLOQUEADA: {type(e).__name__}: {e}")
            blocked[tool] = f"{type(e).__name__}: {e}"
            continue
        dt = time.perf_counter() - t0
        timings[tool] = {"seconds": round(dt, 1),
                         "s_per_image": round(dt / max(len(paths), 1), 2)}
        log(f"  {len(rows)} registros en {dt/60:.1f} min "
            f"({dt/len(paths):.2f} s/imagen)")

        # determinismo: reprocesar un subconjunto y comparar
        sub = paths[: args.determinism_n]
        rows2 = RUNNERS[tool](sub)
        a = pd.DataFrame(rows)[lambda d: d.location_id.isin([p.stem for p in sub])]
        b = pd.DataFrame(rows2)
        m = a.merge(b, on=["location_id", "concept"], suffixes=("_1", "_2"))
        dmax = float((m["score_1"] - m["score_2"]).abs().max())
        timings[tool]["determinism_max_abs_diff"] = dmax
        timings[tool]["deterministic"] = bool(dmax < 1e-6)
        log(f"  determinismo sobre {len(sub)} imagenes: max|dif| = {dmax:.2e}")
        all_rows.extend(rows)

    if not all_rows:
        log("ninguna herramienta produjo resultados")
        return

    df = pd.DataFrame(all_rows)
    df = df.merge(sample[["location_id", "city", "stratum"]], on="location_id", how="left")
    # re-ejecucion parcial: conservar (y respaldar) las filas de las otras herramientas
    prev_csv, prev_json = POC / "poc_detections.csv", POC / "poc_metrics.json"
    if prev_csv.exists():
        import shutil
        shutil.copy2(prev_csv, POC / "poc_detections_prev.csv")
        shutil.copy2(prev_json, POC / "poc_metrics_prev.json")
        old = pd.read_csv(prev_csv)
        done = set(df["tool"])
        df = pd.concat([old[~old["tool"].isin(done)], df], ignore_index=True)
        old_t = json.loads(prev_json.read_text(encoding="utf-8"))["timings"]
        timings = {**{k: v for k, v in old_t.items() if k not in done}, **timings}
    if GDINO_RAW:
        (POC / "gdino_raw_detections.json").write_text(json.dumps(GDINO_RAW), encoding="utf-8")
    df.to_csv(POC / "poc_detections.csv", index=False)

    metrics = compute_metrics(df, paths, timings, blocked)
    (POC / "poc_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n-> {POC / 'poc_metrics.json'}")


if __name__ == "__main__":
    main()
