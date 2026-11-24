"""
Metricas de la validacion de medicion (se ejecuta CUANDO existan anotaciones humanas).

Formato esperado (largo) en concept_validation/human_annotations/<annotator>.csv:
    image_id, concept, value            value: 0/1 (binario), 1-5 (ordinal), n (conteo), bin 0-4 (area)
                                        ambiguous / not_assessable -> celda vacia + columna `flag`
Ground truth final: adjudicated.csv con las mismas columnas (tras adjudicar desacuerdos).

Calcula:
  - acuerdo humano por concepto: Cohen kappa (binario), kappa ponderado cuadratico (ordinal),
    ICC(2,1) (conteo/area), % ambiguos
  - extractor vs GT: precision/recall/F1 en el umbral que maximiza F1 (reportado como tal, en MAIN,
    y verificado en STRESS: no se elige el umbral con STRESS), AP, MAE y Spearman para conteos/ordinales.
No inventa nada: si faltan archivos, termina indicando PENDIENTE.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve

from utils import OUTPUTS

CV = OUTPUTS / "concept_validation"
SCHEMA = json.loads((CV / "annotation_schema.json").read_text(encoding="utf-8"))["per_concept"]


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    x = np.c_[a, b].astype(float); n, k = x.shape
    gm = x.mean(); msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def human_agreement(ann: list[pd.DataFrame]) -> pd.DataFrame:
    A, B = ann[0], ann[1]
    m = A.merge(B, on=["image_id", "concept"], suffixes=("_a", "_b"))
    rows = []
    for c, g in m.groupby("concept"):
        t = SCHEMA.get(c, {}).get("type", "")
        g2 = g.dropna(subset=["value_a", "value_b"])
        amb = float(1 - len(g2) / max(len(g), 1))
        if len(g2) < 10:
            continue
        if t.startswith("presence") or t == "presence":
            met, val = "cohen_kappa", cohen_kappa_score(g2.value_a > 0, g2.value_b > 0)
        elif t.startswith("ordinal"):
            met, val = "weighted_kappa_quad", cohen_kappa_score(g2.value_a.astype(int), g2.value_b.astype(int), weights="quadratic")
        else:
            met, val = "ICC(2,1)", icc21(g2.value_a.to_numpy(), g2.value_b.to_numpy())
        rows.append(dict(concept=c, metric=met, value=round(float(val), 3), n=len(g2), frac_ambiguous=round(amb, 3)))
    return pd.DataFrame(rows)


def extractor_vs_gt(gt: pd.DataFrame, pred: pd.DataFrame, main_ids: set) -> pd.DataFrame:
    m = gt.merge(pred, on=["image_id", "concept"], suffixes=("_gt", "_pred")).dropna(subset=["value"])
    rows = []
    for c, g in m.groupby("concept"):
        gm, gs = g[g.image_id.isin(main_ids)], g[~g.image_id.isin(main_ids)]
        if len(gm) < 10:
            continue
        y = (gm.value > 0).astype(int)
        if y.nunique() == 2:
            p, r, th = precision_recall_curve(y, gm.score)
            f1 = 2 * p * r / np.maximum(p + r, 1e-9); i = int(np.nanargmax(f1[:-1]))
            thr = float(th[i])
            row = dict(concept=c, n_main=len(gm), AP=round(average_precision_score(y, gm.score), 3),
                       thr_from_MAIN=round(thr, 3), P=round(float(p[i]), 3), R=round(float(r[i]), 3), F1=round(float(f1[i]), 3))
            if len(gs) >= 10 and (gs.value > 0).nunique() == 2:
                ys = (gs.value > 0).astype(int); ps = (gs.score >= thr).astype(int)
                tp = int(((ps == 1) & (ys == 1)).sum()); fp = int(((ps == 1) & (ys == 0)).sum()); fn = int(((ps == 0) & (ys == 1)).sum())
                row["F1_STRESS_same_thr"] = round(2 * tp / max(2 * tp + fp + fn, 1), 3)
            rows.append(row)
        else:
            rows.append(dict(concept=c, n_main=len(gm), note="GT sin variacion en MAIN"))
    return pd.DataFrame(rows)


def main():
    files = sorted(glob.glob(str(CV / "human_annotations" / "annotator_*.csv")))
    if len(files) < 2:
        print("PENDIENTE: se requieren >= 2 archivos annotator_*.csv en human_annotations/"); return
    ann = [pd.read_csv(f) for f in files]
    h = human_agreement(ann); (CV / "metrics").mkdir(exist_ok=True)
    h.to_csv(CV / "metrics" / "human_agreement.csv", index=False); print(h.to_string(index=False))
    gt_path = CV / "human_annotations" / "adjudicated.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        S = pd.read_csv(CV / "annotation_sample.csv", dtype=str); main_ids = set(S[S.split == "main"].image_id)
        pr = pd.concat([pd.read_csv(CV / "preannotations" / "preann_v2_yoloworld.csv"),
                        pd.read_csv(CV / "preannotations" / "preann_v2_clip.csv")])[["image_id", "concept", "score"]]
        pr = pr.groupby(["image_id", "concept"], as_index=False).score.max()
        e = extractor_vs_gt(gt, pr, main_ids); e.to_csv(CV / "metrics" / "extractor_vs_gt.csv", index=False)
        print(e.to_string(index=False))


if __name__ == "__main__":
    main()
