"""
EXP-050 (modo en vivo, 06/11) — Panorama de Street View de cada uno de los 400 estímulos, para que la encuesta pida las
fotos a la API en el navegador en lugar de republicar imágenes descargadas.

- San Isidro y Lince: `pano_id` del manifiesto de la captura (las mismas fotos que usa el modelo).
- Barranco y La Victoria: la captura de 2016 se pidió por ubicación y no guardó `pano_id`; se consulta la metadata
  (GRATUITA, no descarga imágenes) en la coordenada del spot → panorama ACTUAL. Esas fotos difieren de las de 2016 del
  modelo: para el análisis de la encuesta se deberán extraer las mismas 4 vistas de ese panorama (≈ 800 imágenes).
Direcciones: REL_X (EXP-056), las mismas de los estímulos regenerados. Salida: outputs/survey/stimuli_live.csv
La clave se lee del .env y nunca se imprime ni se escribe.
"""

from __future__ import annotations

import sys

import pandas as pd

from exp041_lima_segment_graph import lima_spots
from utils import OUTPUTS, REPO_ROOT, log

sys.path.insert(0, str(REPO_ROOT / "image-extraction"))
import streetview_collector as sv  # noqa: E402

OUT = OUTPUTS / "survey"


def main() -> None:
    M = pd.read_csv(OUT / "stimuli_manifest.csv", dtype=str)
    rows, client = [], None
    S, *_ = lima_spots(plima=True)
    coord = S.set_index(["district", "spot_id"])[["lat", "lon"]]
    for r in M.itertuples():
        if r.district in ("san_isidro", "lince"):
            V = pd.read_csv(sv.REPO_ROOT / "image-extraction" / "data" / r.folder / "_manifest" / "views.csv", dtype=str) \
                if not hasattr(main, r.folder) else getattr(main, r.folder)
            setattr(main, r.folder, V)
            v = V[V["spot_id"] == r.spot_id]
            rows.append(dict(stim_id=r.stim_id, district=r.district, pano_id=v["pano_id"].iloc[0],
                             pano_date=v["capture_date_if_known"].iloc[0], headings=r.headings, source="manifest"))
            continue
        if client is None:
            cfg = sv.load_config()
            sv.load_env_file()
            import os
            client = sv.StreetViewClient(cfg["camera"], cfg["network"], os.environ[sv.KEY_ENV])
        lat, lon = coord.loc[(r.district, r.spot_id)]
        meta, _ = client.metadata(float(lat), float(lon))
        rows.append(dict(stim_id=r.stim_id, district=r.district, pano_id=meta.get("pano_id"), pano_date=meta.get("date"),
                         headings=r.headings, source="metadata_current", status=meta.get("status")))
    L = pd.DataFrame(rows).sort_values("stim_id")
    L.to_csv(OUT / "stimuli_live.csv", index=False)
    log(L.groupby(["district", "source"]).agg(n=("stim_id", "size"), with_pano=("pano_id", lambda x: x.notna().sum()),
                                              year_min=("pano_date", "min"), year_max=("pano_date", "max")).to_string())


if __name__ == "__main__":
    main()
