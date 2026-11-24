"""
E6 / EXP-050 — Estímulos de la encuesta pareada de Lima: 4 vistas de 100 spots por distrito en cuadrícula 2×2
(legible en celular; una tira 1280×200 quedaba de ~56 px de alto). Desde el 06/11 (M30, asesoría 05/11): las 4 vistas
son las DIAGONALES a la calle (REL_X de EXP-056); `--views abs4` reproduce la versión anterior (0/90/180/270).

Selección (semilla 42): spots 12/12 con snap a segmento (EXP-041 --plima, un spot por panorama), como mucho un spot por
segmento, estratificados por quintil de percepción PREDICHA dentro del distrito (EXP-051; solo para cubrir el rango,
nunca como etiqueta). REL_X: mejor regla de 4 vistas en los 4 distritos (EXP-056). Disposición 2×2 como si se mirara
desde la vereda: arriba las dos diagonales hacia adelante (izquierda, derecha), abajo las dos hacia atrás.

Salidas: outputs/survey/stimuli/<stim_id>.jpg (640×400; 4 × 320×200) y outputs/survey/stimuli_manifest.csv
(stim_id anónimo → distrito, spot, segmento; el distrito NO se muestra en la encuesta).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image

from lima_views import COLLECTOR_DATA
from utils import OUTPUTS

SEED = 42
PER_DISTRICT = 100
HEADINGS = (0, 90, 180, 270)
FOLDER = {"barranco": "Inseguros-Barranco-GGZ-2016", "la_victoria": "Inseguros-La_Victoria-GGZ-2016",
          "san_isidro": "Inseguros-San_Isidro-GGZ-2016", "lince": "StreetSample-Lince-s50"}
OUT = OUTPUTS / "survey"


def relx_order(views: str) -> list[int]:
    """REL_X de EXP-056 en orden [b+45, b+135, b+225, b+315] → [b+315, b+45, b+225, b+135] (adelante-izq, adelante-der,
    atrás-izq, atrás-der)."""
    h = [int(x) for x in views.split("|")]
    return [h[3], h[0], h[2], h[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", choices=["relx", "abs4"], default="relx")
    views_mode = ap.parse_args().views
    rng = np.random.default_rng(SEED)
    P = pd.read_csv(OUTPUTS / "exp051" / "spot_predicted_perception.csv", dtype={"spot_id": str})
    P = P[(P["seg_idx"] >= 0)]
    rows = []
    for d, g in P.groupby("district"):
        g = g[[all((COLLECTOR_DATA / FOLDER[d] / s / f"heading_{h}.jpg").exists() for h in HEADINGS)
               for s in g["spot_id"]]]
        g = g.sample(frac=1, random_state=SEED).drop_duplicates("seg_idx")        # ≤ 1 spot por segmento
        g = g.assign(q=pd.qcut(g["pct_pp"].rank(method="first"), 5, labels=False))
        take = pd.concat(x.sample(min(len(x), PER_DISTRICT // 5), random_state=SEED) for _, x in g.groupby("q"))
        rows.append(take.assign(folder=FOLDER[d]))
    S = pd.concat(rows, ignore_index=True)
    S = S.sample(frac=1, random_state=SEED).reset_index(drop=True)
    S["stim_id"] = [f"s{i:03d}" for i in range(1, len(S) + 1)]
    if views_mode == "relx":
        X = pd.read_csv(OUTPUTS / "exp056" / "per_spot_selection.csv", dtype={"spot_id": str}).set_index(["district", "spot_id"])
        S["headings"] = ["|".join(map(str, relx_order(X.loc[(d, sp), "REL_X_views"]))) if (d, sp) in X.index else ""
                         for d, sp in zip(S["district"], S["spot_id"])]
        missing = (S["headings"] == "").sum()
        assert missing == 0, f"{missing} spots sin rumbo de calle (REL_X)"
    else:
        S["headings"] = "|".join(map(str, HEADINGS))
    (OUT / "stimuli").mkdir(parents=True, exist_ok=True)
    for r in S.itertuples():
        strip = Image.new("RGB", (640, 400))
        for k, h in enumerate(int(x) for x in r.headings.split("|")):
            with Image.open(COLLECTOR_DATA / r.folder / r.spot_id / f"heading_{h}.jpg") as im:
                strip.paste(im.convert("RGB").resize((320, 200), Image.LANCZOS), ((k % 2) * 320, (k // 2) * 200))
        strip.save(OUT / "stimuli" / f"{r.stim_id}.jpg", quality=82, optimize=True)
    S[["stim_id", "district", "spot_id", "folder", "seg_idx", "pct_pp", "q", "headings"]].to_csv(
        OUT / "stimuli_manifest.csv", index=False, float_format="%.3f")
    print(S["district"].value_counts().to_dict(), len(S))


if __name__ == "__main__":
    main()
