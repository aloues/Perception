"""
EXP-031 — (1) figura de pesos de conceptos con IC y dirección esperada; (2) muestra de VALIDACIÓN HUMANA dirigida.

(2) es el siguiente paso de RQ2 tras EXP-031/032: ya no hace falta anotar 450 imágenes "a ciegas" para tener un
resultado; hace falta comprobar si CLIP MIDE lo que dice cada concepto (nivel 2 de M10). Para eso se eligen, por
concepto direccional, 10 imágenes con puntaje CLIP alto y 10 con puntaje bajo (entre imágenes de test), más 10
intermedias de control; una persona responde presencia sí/no sin ver el puntaje. Con 30 imágenes por concepto se
estima la discriminación (AUC) de CLIP frente al juicio humano. La muestra se arma con seed 42 y no usa etiquetas.

Uso:  python exp031_figure_and_validation_sample.py
Salidas: outputs/exp031/exp031_concept_weights.png, outputs/exp031/validation_sample_clip_concepts.csv
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exp031_concept_bottleneck import EXPECTED  # noqa: E402
from probe_utils import load_pairs, load_split, votes  # noqa: E402
from utils import MANIFESTS, OUTPUTS  # noqa: E402

OUT = OUTPUTS / "exp031"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {+1: "#2a78d6", -1: "#eb6834", 0: "#8a8984"}
LAB = {+1: "La literatura espera: más seguro", -1: "La literatura espera: menos seguro", 0: "Sin dirección esperada"}

w = pd.read_csv(OUT / "concept_weights.csv").sort_values("weight")
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for y, (_, r) in enumerate(w.iterrows()):
    c = COL[int(r["expected_sign"])]
    ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=c, lw=2, solid_capstyle="round")
    ax.plot(r["weight"], y, "o", ms=7, color=c, mec=SURF, mew=2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(w)), w["concept"].str.replace("_", " "), fontsize=8, color=INK)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
ax.set_xlabel("Peso en el modelo de 17 conceptos (estandarizado; IC 95 % bootstrap)", fontsize=8, color=INK2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(handles=[plt.Line2D([], [], color=COL[k], marker="o", lw=2, label=LAB[k]) for k in (+1, -1, 0)],
          loc="lower right", fontsize=7.5, frameon=False)
ax.set_title("EXP-031 · Conceptos del codebook medidos con CLIP: 8 de 8 signos coinciden con la literatura",
             fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(OUT / "exp031_concept_weights.png", facecolor=SURF)

# muestra de validación
rng = np.random.default_rng(42)
F = pd.read_csv(OUT / "concept_scores_std.csv.gz").set_index("location_id")
pairs = load_pairs()
test_ids = set(votes(pairs, load_split("image_disjoint")["test"])["id"])
man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).set_index("location_id")
F = F[F.index.isin(test_ids)]
rows = []
for c in [k for k, v in EXPECTED.items() if v != 0]:
    s = F[c].sort_values()
    groups = {"alto": s.index[-200:], "bajo": s.index[:200],
              "medio": s.index[len(s) // 2 - 100: len(s) // 2 + 100]}
    for g, pool in groups.items():
        for lid in rng.choice(pool, 10, replace=False):
            rows.append(dict(concept=c, clip_group=g, location_id=lid, city=man.loc[lid, "city"],
                             image_path=man.loc[lid, "image_path"]))
sample = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
sample.insert(0, "item", range(1, len(sample) + 1))
sample["human_present_yes_no_unsure"] = ""
sample.to_csv(OUT / "validation_sample_clip_concepts.csv", index=False)
print(len(sample), "ítems ->", OUT / "validation_sample_clip_concepts.csv")
