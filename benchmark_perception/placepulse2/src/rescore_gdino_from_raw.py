"""
Re-puntua Grounding DINO desde las detecciones CRUDAS guardadas (gdino_raw_detections.json)
con coincidencia EXACTA de frase, sin volver a ejecutar el modelo (la v1 tenia un bug de
matching por subcadena; ver POC_REPORT.md). Respalda los archivos previos.
"""
import json, shutil
import numpy as np, pandas as pd
from PIL import Image
from run_poc_concepts import DETECT_PROMPTS, POC, compute_metrics
from utils import log

raw = {}
for r in json.loads((POC / "gdino_raw_detections.json").read_text()):
    raw.setdefault(r["location_id"], r)            # ignora el reproceso de determinismo
rows = []
for lid, r in raw.items():
    labels, scores, boxes = r["labels"], np.array(r["scores"]), np.array(r["boxes"])
    # Mascara de marca de agua: el logo "Google" (esquina inferior izquierda) dispara
    # "graffiti" en ~50% de las cajas. Se descartan cajas con y0 > 85% alto y x0 < 25% ancho.
    W, H = Image.open(POC / "images" / f"{lid}.jpg").size
    keep = ~((boxes[:, 1] > 0.85 * H) & (boxes[:, 0] < 0.25 * W)) if len(boxes) else np.zeros(0, bool)
    labels = [l for l, k in zip(labels, keep) if k]; scores = scores[keep]; boxes = boxes[keep]
    for c, key in DETECT_PROMPTS.items():
        m = np.array([lb == key for lb in labels], dtype=bool)
        best, box = 0.0, ""
        if m.any():
            best = float(scores[m].max()); box = ",".join(map(str, boxes[m][scores[m].argmax()]))
        rows.append({"location_id": lid, "concept": c, "tool": "gdino", "score": round(best, 6),
                     "n_detections": int(m.sum()), "bbox": box})
new = pd.DataFrame(rows)
cur = pd.read_csv(POC / "poc_detections.csv")
meta = cur.drop_duplicates("location_id").set_index("location_id")[["city", "stratum"]]
new = new.join(meta, on="location_id")
import os
if not (POC / "poc_detections_prev_v1b.csv").exists():
    shutil.copy2(POC / "poc_detections.csv", POC / "poc_detections_prev_v1b.csv")
    shutil.copy2(POC / "poc_metrics.json", POC / "poc_metrics_prev_v1b.json")
df = pd.concat([cur[cur.tool != "gdino"], new], ignore_index=True)
df.to_csv(POC / "poc_detections.csv", index=False)
old = json.loads((POC / "poc_metrics.json").read_text(encoding="utf-8"))
old["rescored_note"] = "gdino re-puntuado offline: coincidencia exacta + mascara de marca de agua Google (desde gdino_raw_detections.json)"
m = compute_metrics(df, list(range(df.location_id.nunique())), old["timings"], old["blocked"])
m["rescored_note"] = old["rescored_note"]
(POC / "poc_metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
for r in m["per_tool_concept"]:
    if r["tool"] == "gdino": print(r["concept"], r["fire_rate"], r["score_mean"], r["score_std"])
for a in m["agreement"]:
    if a["tool_b"] == "gdino" or a["tool_a"] == "gdino": print(a["concept"], a["tool_a"], a["tool_b"], a["spearman"])
