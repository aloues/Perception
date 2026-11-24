"""
EXP-025 — diagnóstico POST-HOC (no estaba en el diseño registrado; no cambia ningún criterio).

El criterio R1 (>= 95 % de imágenes a <= 100 m de su nodo) falló. Este script mide si la causa es
que las imágenes están lejos de la RED (mala cobertura OSM / imágenes fuera de vías "drive") o solo
lejos de las INTERSECCIONES (cuadras largas): distancia de cada imagen a la arista más cercana.

Uso:  python exp025_posthoc_edge_snap.py   (usa los .graphml cacheados por build_street_graph_pilot.py)
"""
import re

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from utils import MANIFESTS, OUTPUTS, log

OUT = OUTPUTS / "street_graph_pilot"
COORD_RE = re.compile(r"images/(-?\d+\.\d+)_(-?\d+\.\d+)_")

man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).dropna(subset=["image_path"])
ll = man["image_path"].str.extract(COORD_RE).astype(float)
man["lat"], man["lon"] = ll[0], ll[1]
rows = []
for city in ["Atlanta", "Berlin", "Santiago"]:
    g = man[man["city"] == city].dropna(subset=["lat"])
    Gp = ox.project_graph(ox.load_graphml(OUT / "graphs" / f"{city}_drive.graphml"))
    pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").to_crs(Gp.graph["crs"])
    _, d = ox.distance.nearest_edges(Gp, pts.x.to_numpy(), pts.y.to_numpy(), return_dist=True)
    d = np.asarray(d, dtype=float)
    rows.append(dict(city=city, n_images=len(g), edge_dist_median_m=round(float(np.median(d)), 1),
                     edge_dist_p90_m=round(float(np.percentile(d, 90)), 1),
                     pct_edge_le_25m=round(100 * float((d <= 25).mean()), 1),
                     pct_edge_le_100m=round(100 * float((d <= 100).mean()), 1)))
    log(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "posthoc_edge_snap.csv", index=False)
