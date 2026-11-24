"""
Vistas Street View históricas de Lima (Barranco / La Victoria) como tabla vista → spot → coordenadas.

Fuente: `external_assets/embeddings_export/metadata.json` (rutas, distrito, crime_id, heading) +
`external_assets/lima_graph/embeddings_export/embeddings.npy` (30,866 × 768, SimCLR ViT-B/16
`simclr_perception_224x_224_best_epoch_50.pt`, L2-normalizado; mismo checkpoint y preprocesado que
las features `ours_simclr_vit_b16` de Place Pulse). Las IMÁGENES no están en local: solo embeddings.

Discrepancia histórica detectada (2026-10-23): extract_embeddings.py asigna el distrito con
`"Barranco" if "Barranco" in root else "La Victoria"`. Por eso:
  - carpeta `Inseguros/` (428 spots, 5,136 vistas) quedó rotulada "La Victoria", pero sus IDs están
    en image-extraction/INseguros.csv con lat ≈ −23.5 → son de SÃO PAULO. Se EXCLUYEN.
  - carpeta `Inseguros-Barranco/` (362 spots, IDs 95–460) no tiene IDs del CSV de crímenes ni
    coordenadas recuperables → origen DESCONOCIDO. Se EXCLUYE.
Solo se usan las carpetas `Inseguros-{Distrito}-GGZ-2016/`, cuyos IDs se cruzan con
`distritos/crimen_distritos_seleccionados.csv` (año 2016) para obtener lat/lon solicitadas.
La coordenada es la del REGISTRO DE CRIMEN pedido a la API; la posición real del panorama no se
guardó en el histórico (sin metadata).

P-LIMA (2026-10-29): San Isidro (`crime_points` 2016) y Lince (`street_sampling` 50 m) se capturaron con el MISMO
protocolo (image-extraction/streetview_collector.py, huella c06e19f3699baa3e) y sus embeddings se extrajeron con el
MISMO script y checkpoint (graph/extract_embeddings.py, epoch_50) en `external_assets/embeddings_export_plima/<carpeta>/`.
`load_views(plima=True)` los concatena a los históricos; el distrito sale de la carpeta (el campo `districts` del
script histórico dice "La Victoria" para todo lo que no es Barranco). Coordenadas: CSV de crímenes para `crime_points`,
manifiesto del collector (`lat_req`) para `street_sampling`; el manifiesto aporta además pano, fecha y posición real.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from utils import REPO_ROOT

META = REPO_ROOT / "external_assets" / "embeddings_export" / "metadata.json"
EMB = REPO_ROOT / "external_assets" / "lima_graph" / "embeddings_export" / "embeddings.npy"
CRIMES = REPO_ROOT / "distritos" / "crimen_distritos_seleccionados.csv"
FOLDER_TO_DISTRICT = {
    "Inseguros-Barranco-GGZ-2016": ("barranco", "Barranco", "included"),
    "Inseguros-La_Victoria-GGZ-2016": ("la_victoria", "La Victoria", "included"),
    "Inseguros": ("sao_paulo", None, "excluded:sao_paulo_ids_in_INseguros.csv"),
    "Inseguros-Barranco": ("unknown", None, "excluded:unknown_origin_no_coordinates"),
    "Inseguros-San_Isidro-GGZ-2016": ("san_isidro", "San Isidro", "included"),
    "StreetSample-Lince-s50": ("lince", None, "included"),
    "StreetSample-San_Isidro-s50": ("san_isidro", None, "included:bridge"),
}
SAMPLING = {"Inseguros-Barranco-GGZ-2016": "crime_points", "Inseguros-La_Victoria-GGZ-2016": "crime_points",
            "Inseguros-San_Isidro-GGZ-2016": "crime_points", "StreetSample-Lince-s50": "street_sampling",
            "StreetSample-San_Isidro-s50": "street_sampling"}
PLIMA_EXPORTS = REPO_ROOT / "external_assets" / "embeddings_export_plima"
COLLECTOR_DATA = REPO_ROOT / "image-extraction" / "data"


def _views_from_meta(m: dict, folder_re: str) -> pd.DataFrame:
    V = pd.DataFrame({k: m[k] for k in ("paths", "districts", "crime_ids", "headings")})
    V = V.rename(columns={"paths": "path", "districts": "district_hist", "crime_ids": "spot_id",
                          "headings": "heading"})
    V["folder"] = V["path"].str.replace("\\", "/", regex=False).str.extract(folder_re)[0]
    return V


def load_views(include_excluded: bool = False, plima: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla de vistas alineada con las filas de embeddings (columna `row`). `plima=True` añade San Isidro y Lince."""
    V = _views_from_meta(json.loads(META.read_text()), r"image-extraction/([^/]+)/")
    E = np.load(EMB, mmap_mode="r")
    assert E.shape[0] == len(V), "embeddings.npy y metadata.json no están alineados"
    if plima:
        parts, blocks = [V], [np.asarray(E)]
        for d in sorted(PLIMA_EXPORTS.glob("*/metadata.json")):
            Vk = _views_from_meta(json.loads(d.read_text()), r"([^/]+)/[^/]+/heading_\d+\.\w+$")
            Ek = np.load(d.parent / "embeddings.npy")
            assert Ek.shape[0] == len(Vk) and (Vk["folder"] == d.parent.name).all(), d
            parts.append(Vk)
            blocks.append(Ek)
        V, E = pd.concat(parts, ignore_index=True), np.concatenate(blocks)
    V["row"] = np.arange(len(V))
    mapped = V["folder"].map(FOLDER_TO_DISTRICT)
    V["district"] = mapped.map(lambda t: t[0])
    V["crime_district"] = mapped.map(lambda t: t[1])
    V["inclusion"] = mapped.map(lambda t: t[2])
    V["sampling"] = V["folder"].map(SAMPLING)
    if not include_excluded:
        V = V[V["inclusion"].fillna("").str.startswith("included")].copy() if plima else             V[V["inclusion"] == "included"].copy()
    return V.reset_index(drop=True), E


def attach_coordinates(V: pd.DataFrame, year: int = 2016) -> pd.DataFrame:
    C = pd.read_csv(CRIMES, sep=";")
    C = C[C["YEAR"] == year]
    C["spot_id"] = C["ID"].map(lambda v: f"{int(float(v))}.0")
    C = C.drop_duplicates("spot_id")[["spot_id", "DISTRITO", "latitude", "longitude", "MODALIDAD"]]
    out = V.merge(C, on="spot_id", how="left")
    out["coord_ok"] = out["latitude"].notna() & (out["DISTRITO"] == out["crime_district"])
    # P-LIMA: metadata del collector (pano, fecha, posición real) y coordenada de los spots por calle
    man = []
    for folder in out["folder"].dropna().unique():
        p = COLLECTOR_DATA / folder / "_manifest" / "spots.csv"
        if p.exists():
            M = pd.read_csv(p, dtype={"spot_id": str, "pano_id": str, "segment_key": str})
            man.append(M.assign(folder=folder)[["folder", "spot_id", "lat_req", "lon_req", "segment_key", "pano_id",
                                                "lat_pano", "lon_pano", "capture_date_if_known"]])
    if man:
        out = out.merge(pd.concat(man), on=["folder", "spot_id"], how="left")
        street = out["sampling"].eq("street_sampling") & out["lat_req"].notna() if "sampling" in out else False
        out.loc[street, "latitude"] = out.loc[street, "lat_req"]
        out.loc[street, "longitude"] = out.loc[street, "lon_req"]
        out.loc[street, "coord_ok"] = True
    return out


def spots_table(V: pd.DataFrame) -> pd.DataFrame:
    g = V.groupby(["district", "spot_id"])
    S = g.agg(n_views=("heading", "nunique"), lat=("latitude", "first"), lon=("longitude", "first"),
              coord_ok=("coord_ok", "all")).reset_index()
    S["complete_12"] = S["n_views"] == 12
    return S
