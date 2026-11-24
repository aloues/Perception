"""
EXP-063 — Ejemplo numérico AUDITABLE para la asesoría (R1): dos pares REALES del test de image_disjoint con el puntaje
de la cabeza lineal-BT (DINOv2 congelado, seed 42), σ(s_i − s_j), voto humano observado y acierto/error.

Se eligen de forma reproducible (no a mano): (a) el par con mayor |s_i − s_j| entre los pares de test con un voto
acertado de ciudades distintas; (b) un par ERRADO con |Δs| alto (para mostrar que el modelo se equivoca con confianza);
(c) un par con Δs ≈ 0 (incertidumbre). Salidas: outputs/exp063/worked_example.json y
deliverables/figures/worked_example_pairs.{png,pdf}.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURES, MANIFESTS, OUTPUTS, REPO_ROOT, ZIP_PATH  # noqa: E402

OD = OUTPUTS / "exp063" / "image_disjoint"
FIG = REPO_ROOT / "deliverables" / "figures"


def main() -> None:
    rows = np.load(OD / "rows_dinov2_vit_b14_z.npy")
    g = np.load(OD / "scores" / "dinov2_vit_b14__z__linear_bt__s42.npy").astype(np.float64)
    tp = np.load(OD / "test_pairs_dinov2_vit_b14_z.npz")
    li, ri, y = tp["li"], tp["ri"], tp["y"]
    ids = json.loads((FEATURES / "dinov2_vit_b14" / "index.json").read_text())["location_ids"]
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).set_index("location_id")
    d = g[li] - g[ri]
    p = 1 / (1 + np.exp(-d))
    pred = (d > 0).astype(int)
    ok = pred == y
    city = man["city"]
    gid_l, gid_r = [ids[rows[k]] for k in li], [ids[rows[k]] for k in ri]
    diffcity = np.array([city[a] != city[b] for a, b in zip(gid_l, gid_r)])
    pick = {
        "acierto_confiado": int(np.argmax(np.where(ok & diffcity, np.abs(d), -1))),
        "error_confiado": int(np.argmax(np.where(~ok & diffcity, np.abs(d), -1))),
        "incierto": int(np.argmin(np.where(diffcity, np.abs(d), np.inf))),
    }
    # percentiles del puntaje en test para dar escala
    test_rows = np.unique(np.r_[li, ri])
    pct = lambda v: float((g[test_rows] < v).mean() * 100)
    out = {"source": "outputs/exp063/image_disjoint/scores/dinov2_vit_b14__z__linear_bt__s42.npy",
           "model": "DINOv2 ViT-B/14 congelado + cabeza lineal-BT (s = w·z, sin bias), seed 42",
           "split": "image_disjoint / test (imágenes nunca vistas en train)",
           "test_accuracy": float(ok.mean()), "n_test_pairs": int(len(y)), "pairs": {}}
    for name, k in pick.items():
        a, b = gid_l[k], gid_r[k]
        out["pairs"][name] = {
            "left_id": a, "right_id": b, "left_city": city[a], "right_city": city[b],
            "left_image": man.loc[a, "image_path"], "right_image": man.loc[b, "image_path"],
            "s_left": float(g[li[k]]), "s_right": float(g[ri[k]]), "s_left_minus_s_right": float(d[k]),
            "P_left_looks_safer": float(p[k]), "model_choice": "left" if pred[k] else "right",
            "human_vote": "left" if y[k] == 1 else "right", "correct": bool(ok[k]),
            "s_left_test_percentile": pct(g[li[k]]), "s_right_test_percentile": pct(g[ri[k]])}
    (OUTPUTS / "exp063").mkdir(exist_ok=True)
    (OUTPUTS / "exp063" / "worked_example.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(10, 11.5))
    titles = {"acierto_confiado": "Confident and correct", "error_confiado": "Confident but wrong",
              "incierto": "Uncertain (Δs ≈ 0)"}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for r, (name, e) in enumerate(out["pairs"].items()):
            for c, side in enumerate(("left", "right")):
                im = Image.open(io.BytesIO(zf.read(e[f"{side}_image"]))).convert("RGB")
                ax = axes[r, c]
                ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
                won = e["human_vote"] == side
                chosen = e["model_choice"] == side
                ax.set_title(f"{side.upper()} · {e[f'{side}_city']}\n s = {e[f's_{side}']:+.3f}"
                             f"  (test pct {e[f's_{side}_test_percentile']:.0f})"
                             f"{'  · human vote ✓' if won else ''}{'  · model pick' if chosen else ''}",
                             fontsize=10, color="#1B2A3A")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2F7D5B" if won else "#B8C0C8"); sp.set_linewidth(4 if won else 1)
            axes[r, 0].set_ylabel(f"{titles[name]}\nP(left safer) = σ({e['s_left_minus_s_right']:+.2f}) = "
                                  f"{e['P_left_looks_safer']:.2f}\n{'CORRECT' if e['correct'] else 'WRONG'}",
                                  fontsize=11, rotation=0, ha="right", va="center", labelpad=10)
    fig.suptitle("Real test pairs (image_disjoint): frozen DINOv2 + linear Bradley–Terry head", fontsize=13)
    fig.text(0.5, 0.005, "Images: Place Pulse 2.0 (Dubey et al., 2016), Google Street View. "
             "s = w·z; P(left looks safer) = σ(s_left − s_right). Source: outputs/exp063/worked_example.json",
             ha="center", fontsize=8, color="#5E6B78")
    fig.tight_layout(rect=[0.12, 0.02, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"worked_example_pairs.{ext}", dpi=200)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
