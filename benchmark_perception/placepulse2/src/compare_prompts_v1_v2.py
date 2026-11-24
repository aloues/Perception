"""Compara prompts v1 vs v2 en las 72 imagenes del PoC (sin ground truth: solo fire-rate,
discriminacion y acuerdo entre herramientas; NO precision). Salida: outputs/poc/prompts_v1_v2_comparison.csv"""
import numpy as np, pandas as pd
from utils import OUTPUTS
P = OUTPUTS / "poc"
v1 = pd.read_csv(P / "poc_detections.csv"); v1 = v1.rename(columns={"location_id": "image_id"})
y2 = pd.read_csv(P / "poc_v2_prompts_yoloworld.csv"); c2 = pd.read_csv(P / "poc_v2_prompts_clip.csv")
name = {"street_lighting": "lighting_infrastructure"}
v1["concept"] = v1["concept"].replace(name)
piv = lambda d, t: d[d.tool == t].pivot_table(index="image_id", columns="concept", values="score")
Y1, G1, C1 = piv(v1, "yoloworld"), piv(v1, "gdino"), piv(v1, "clip")
Y2 = y2.pivot_table(index="image_id", columns="concept", values="score"); C2 = c2.pivot_table(index="image_id", columns="concept", values="score")
rows = []
for c in Y2.columns:
    r = dict(concept=c)
    r["yolo_v1_fire"] = round(float((Y1[c] > .05).mean()), 3) if c in Y1 else None
    r["yolo_v2_fire"] = round(float((Y2[c] > .05).mean()), 3)
    r["yolo_v2_score_std"] = round(float(Y2[c].std()), 3)
    if c in Y1:
        r["spearman_yolo_v1_v2"] = round(float(Y1[c].rank().corr(Y2.loc[Y1.index, c].rank())), 3) if Y1[c].std() > 0 and Y2[c].std() > 0 else None
    if c in G1 and G1[c].std() > 0 and Y2[c].std() > 0:
        r["spearman_yolo_v2_vs_gdino_v1"] = round(float(Y2.loc[G1.index, c].rank().corr(G1[c].rank())), 3)
    if c in C2:
        r["clip_v2_score_std"] = round(float(C2[c].std()), 3)
        if c in C1 and C1[c].std() > 0:
            r["spearman_clip_v1_v2"] = round(float(C1[c].rank().corr(C2.loc[C1.index, c].rank())), 3)
    rows.append(r)
for c in C2.columns:
    if c not in Y2.columns:
        rows.append(dict(concept=c, clip_v2_score_std=round(float(C2[c].std()), 3)))
d = pd.DataFrame(rows); d.to_csv(P / "prompts_v1_v2_comparison.csv", index=False); print(d.to_string(index=False))
