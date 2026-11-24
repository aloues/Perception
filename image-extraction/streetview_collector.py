"""
Collector de Street View por distrito — réplica parametrizada de downloadImage.py (Barranco / La Victoria).

POR QUE
-------
downloadImage.py tenía el distrito y el año escritos a mano, no guardaba metadata (pano id, fecha,
posición real), no distinguía "sin cobertura" de una imagen válida (guardaba el placeholder gris de
~10 KB; 415 acabaron en graph/blacklist.json) y su única forma de reanudar era "la carpeta del punto
no está vacía". Este módulo conserva EXACTAMENTE la petición de imagen histórica

    https://maps.googleapis.com/maps/api/streetview?size=640x400&location={lat},{lon}
        &heading={h}&pitch=0&fov=90&key=...           h ∈ range(0, 360, 30)

y la convención de carpetas `<dataset>/<spot_id>/heading_{h}.jpg` (la que parsea
graph/extract_embeddings.py), y añade: metadata gratuita antes de pagar imágenes, manifiestos de spots
/ vistas / incidencias, reanudación idempotente con hash, reintentos con backoff, límite de ritmo,
dry-run con costo estimado y QA.

Fuentes de spots (ver configs/streetview_lima.yaml):
  crime_points    réplica exacta del histórico (un spot por registro de labels_{year}_{distrito}.csv);
                  existe para San Isidro, NO para Lince (no hay registros en distritos/).
  street_sampling puntos cada `spacing_m` a lo largo de cada segmento OSM (recomendado: no depende del
                  crimen — evita la circularidad de EXP-013 — y alimenta la unidad "segmento").
  grid            grilla regular de image-extraction/extract-district.py recortada al polígono.

Uso (desde la raíz del repo):
    python image-extraction/streetview_collector.py plan    --district lince --spot-source street_sampling
    python image-extraction/streetview_collector.py collect --district lince --spot-source street_sampling --dry-run
    python image-extraction/streetview_collector.py collect --district san_isidro --spot-source crime_points --limit 5
    python image-extraction/streetview_collector.py verify  --district lince --spot-source street_sampling
    python image-extraction/streetview_collector.py fingerprint

La clave se lee SOLO de GOOGLE_STREETVIEW_API_KEY (o de image-extraction/.env, ignorado por Git) y nunca
se escribe en logs ni manifiestos. `collect` real exige `--max-billable-images` y anota cada corrida en el
ledger (`cost.ledger`); se niega a superar `cost.cap_usd_total`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "streetview_lima.yaml"
STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
KEY_ENV = "GOOGLE_STREETVIEW_API_KEY"
ENV_FILE = Path(__file__).resolve().parent / ".env"      # ignorado por Git
# parámetros de cámara que definen el "criterio de extracción" (entran en la huella)
CAMERA_KEYS = ("provider", "size", "fov", "pitch", "heading_start", "heading_step", "n_headings",
               "request_by", "source", "radius_m")

log = logging.getLogger("streetview_collector")


# ─── configuración ────────────────────────────────────────────────────────────

def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_env_file(path: Path = ENV_FILE, environ: dict | None = None) -> list[str]:
    """Cargador mínimo de `KEY=VALUE` (python-dotenv no está instalado). No sobrescribe variables ya
    definidas. Devuelve solo los NOMBRES cargados: los valores nunca se imprimen."""
    environ = os.environ if environ is None else environ
    path = Path(path)
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and k not in environ:
            environ[k] = v
            loaded.append(k)
    return loaded


def headings(camera: dict) -> list[int]:
    """Headings normalizados a [0, 360), sin duplicar 0/360. Histórico: range(0, 360, 30)."""
    hs = [int(camera["heading_start"] + i * camera["heading_step"]) % 360
          for i in range(int(camera["n_headings"]))]
    if len(set(hs)) != len(hs):
        raise ValueError(f"headings duplicados: {hs}")
    return hs


def camera_fingerprint(camera: dict) -> str:
    """Hash del criterio de cámara canonicalizado: dos distritos con la misma huella usan la misma captura."""
    canon = json.dumps({k: camera[k] for k in CAMERA_KEYS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def redact(url: str) -> str:
    return re.sub(r"key=[^&]+", "key=REDACTED", url)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── geometría del distrito y red OSM ─────────────────────────────────────────

def _osmnx(cfg: dict):
    import osmnx as ox
    ox.settings.cache_folder = str(REPO_ROOT / cfg["osm"]["cache_folder"])
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 300
    return ox


def _snapshot(cfg: dict, district: str, suffix: str) -> Path:
    return REPO_ROOT / cfg["osm"]["cache_folder"] / "graphs" / f"{district}_{suffix}"


def district_polygon(cfg: dict, district: str):
    """Polígono administrativo OSM (Nominatim), el mismo que usa graph_from_place en el histórico.

    Si existe el snapshot congelado (external_assets/osm_cache/graphs/) se usa ese, para que el plan
    no cambie con ediciones posteriores de OSM.
    """
    snap = _snapshot(cfg, district, "boundary.geojson")
    if snap.exists():
        import geopandas as gpd
        return gpd.read_file(snap).geometry.iloc[0]
    ox = _osmnx(cfg)
    gdf = ox.geocode_to_gdf(cfg["districts"][district]["osm_query"])
    return gdf.geometry.iloc[0]


def district_graph(cfg: dict, district: str):
    ox = _osmnx(cfg)
    snap = _snapshot(cfg, district, f"{cfg['osm']['network_type']}.graphml")
    if snap.exists():
        return ox.load_graphml(snap)
    return ox.graph_from_place(cfg["districts"][district]["osm_query"],
                               network_type=cfg["osm"]["network_type"],
                               simplify=cfg["osm"]["simplify"], retain_all=False)


# ─── planificación de spots ───────────────────────────────────────────────────

SPOT_COLUMNS = ["spot_id", "district", "spot_source", "source_ref", "lat_req", "lon_req",
                "segment_key", "in_district"]


def spots_from_crime_points(cfg: dict, district: str, year: int | None = None) -> pd.DataFrame:
    """Réplica de downloadImage.py: filas de labels_{year}_{Distrito}.csv en su orden, spot_id = ID."""
    d = cfg["districts"][district]
    if not d.get("crime_name"):
        raise ValueError(f"{district}: no hay registros de crimen en distritos/ — la fuente crime_points "
                         f"no es replicable para este distrito (usar street_sampling o grid)")
    year = year or cfg["spot_sources"]["crime_points"]["year"]
    path = REPO_ROOT / cfg["spot_sources"]["crime_points"]["labels_dir"] / \
        f"labels_{year}_{d['crime_name'].replace(' ', '_')}.csv"
    L = pd.read_csv(path)
    # normalize_crime_id de graph/extract_embeddings.py: float → "123.0"
    sid = L["ID"].map(lambda v: f"{int(float(v))}.0")
    return pd.DataFrame({"spot_id": sid, "district": district, "spot_source": "crime_points",
                         "source_ref": f"{path.relative_to(REPO_ROOT).as_posix()}#ID",
                         "lat_req": L["latitude"].astype(float), "lon_req": L["longitude"].astype(float),
                         "segment_key": "", "in_district": np.nan})


def sample_along_segments(segments: Iterable[tuple[str, object]], spacing_m: float,
                          min_segment_m: float) -> list[tuple[str, int, float, float]]:
    """Puntos equiespaciados sobre geometrías en METROS: offsets spacing/2, 3·spacing/2, ...

    Un segmento más corto que `min_segment_m` no recibe spot; uno más corto que `spacing_m` recibe
    uno en su punto medio. Devuelve (segment_key, idx, x, y).
    """
    out = []
    for key, geom in segments:
        L = geom.length
        if L < min_segment_m:
            continue
        n = max(1, int(L // spacing_m))
        step = L / n
        for i in range(n):
            p = geom.interpolate(step * (i + 0.5))
            out.append((key, i, p.x, p.y))
    return out


def spots_from_street_sampling(cfg: dict, district: str) -> pd.DataFrame:
    import geopandas as gpd
    ox = _osmnx(cfg)
    sp = cfg["spot_sources"]["street_sampling"]
    G = ox.convert.to_undirected(district_graph(cfg, district))
    Gp = ox.project_graph(G)
    edges = ox.graph_to_gdfs(Gp, nodes=False).reset_index()
    segs = [(f"{min(u, v)}_{max(u, v)}_{k}", g) for u, v, k, g in
            zip(edges["u"], edges["v"], edges["key"], edges.geometry)]
    pts = sample_along_segments(segs, sp["spacing_m"], sp["min_segment_m"])
    gp = gpd.GeoDataFrame({"segment_key": [p[0] for p in pts], "i": [p[1] for p in pts]},
                          geometry=gpd.points_from_xy([p[2] for p in pts], [p[3] for p in pts]),
                          crs=Gp.graph["crs"]).to_crs(4326)
    code = district[:3].upper()
    return pd.DataFrame({"spot_id": [f"{code}-{k}-{i:03d}" for k, i in zip(gp["segment_key"], gp["i"])],
                         "district": district, "spot_source": "street_sampling",
                         "source_ref": f"osm:{cfg['osm']['network_type']}:spacing={sp['spacing_m']}",
                         "lat_req": gp.geometry.y.round(7), "lon_req": gp.geometry.x.round(7),
                         "segment_key": gp["segment_key"], "in_district": np.nan})


def grid_points(bounds: tuple[float, float, float, float], step_m: float) -> list[tuple[float, float]]:
    """Misma aritmética que extract-district.py: paso en grados desde la latitud media del bbox."""
    min_lon, min_lat, max_lon, max_lat = bounds
    step_lat = step_m / 111000
    step_lon = step_m / (111000 * math.cos(math.radians((min_lat + max_lat) / 2)))
    pts, lat = [], min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            pts.append((round(lat, 7), round(lon, 7)))
            lon += step_lon
        lat += step_lat
    return pts


def spots_from_grid(cfg: dict, district: str) -> pd.DataFrame:
    poly = district_polygon(cfg, district)
    step = cfg["spot_sources"]["grid"]["step_m"]
    pts = grid_points(poly.bounds, step)
    code = district[:3].upper()
    return pd.DataFrame({"spot_id": [f"{code}-g{i:05d}" for i in range(len(pts))], "district": district,
                         "spot_source": "grid", "source_ref": f"extract-district.py:step={step}",
                         "lat_req": [p[0] for p in pts], "lon_req": [p[1] for p in pts],
                         "segment_key": "", "in_district": np.nan})


def plan_spots(cfg: dict, district: str, source: str, year: int | None = None,
               polygon=None) -> pd.DataFrame:
    from shapely.geometry import Point
    fn = {"crime_points": lambda: spots_from_crime_points(cfg, district, year),
          "street_sampling": lambda: spots_from_street_sampling(cfg, district),
          "grid": lambda: spots_from_grid(cfg, district)}[source]
    S = fn()
    poly = polygon if polygon is not None else district_polygon(cfg, district)
    S["in_district"] = [poly.covers(Point(lon, lat)) for lat, lon in zip(S["lat_req"], S["lon_req"])]
    if S["spot_id"].duplicated().any():
        raise ValueError("spot_id duplicados en el plan")
    if source == "grid":                       # la grilla es un bbox: fuera del polígono no se pide
        S = S[S["in_district"]].reset_index(drop=True)
    return S[SPOT_COLUMNS]


def dataset_folder(cfg: dict, district: str, source: str, year: int | None = None) -> str:
    d = cfg["districts"][district]
    if source == "crime_points":
        return d["folder_crime"].format(year=year or cfg["spot_sources"]["crime_points"]["year"])
    if source == "street_sampling":
        return cfg["spot_sources"]["street_sampling"]["folder"].format(
            name=d["name"].replace(" ", "_"), spacing=cfg["spot_sources"]["street_sampling"]["spacing_m"])
    return cfg["spot_sources"]["grid"]["folder"].format(
        name=d["name"].replace(" ", "_"), step=cfg["spot_sources"]["grid"]["step_m"])


def estimate_cost(n_spots: int, cfg: dict, coverage_rate: float = 1.0) -> dict:
    n_views = int(round(n_spots * coverage_rate)) * cfg["camera"]["n_headings"]
    c = cfg["cost"]
    billable = max(0, n_views - c["free_calls_per_month"])
    return {"n_spots": n_spots, "assumed_coverage_rate": coverage_rate,
            "metadata_requests": n_spots, "image_requests": n_views,
            "billable_image_requests_if_free_quota_unused": billable,
            "usd_estimate_if_free_quota_unused": round(billable / 1000 * c["usd_per_1000_static"], 2),
            "usd_estimate_without_free_quota": round(n_views / 1000 * c["usd_per_1000_static"], 2),
            "price_source": "configs/streetview_lima.yaml [VERIFICAR en consola]"}


LEDGER_COLUMNS = ["date_utc", "tramo", "district", "spot_source", "images_requested", "images_cumulative",
                  "usd_est_no_quota", "notes", "run_id"]


class Budget:
    """Tope duro de imágenes facturables + ledger CSV (una fila por corrida, reescrita de forma atómica).

    Cuenta cada INTENTO al endpoint de imagen (reintentos incluidos; cota superior conservadora: los 404
    de `return_error_code` y los 5xx probablemente no se facturan). La metadata no cuenta (es gratis).
    Se niega a pedir una imagen si el peor caso de esa petición (1 + max_retries intentos) supera el
    tope total del ledger o el tope de la corrida (`--max-billable-images`).
    """

    def __init__(self, ledger: Path, cap_total_images: int, max_run_images: int, usd_per_1000: float,
                 tramo: str = "", district: str = "", spot_source: str = "", run_id: str | None = None):
        self.ledger, self.cap_total, self.max_run = Path(ledger), int(cap_total_images), int(max_run_images)
        self.usd_per_1000 = usd_per_1000
        self.row = {"tramo": tramo, "district": district, "spot_source": spot_source,
                    "run_id": run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}
        self.prior_rows = self._read()
        self.prior = int(pd.to_numeric(self.prior_rows["images_requested"], errors="coerce").fillna(0).sum())
        self.used = 0

    def _read(self) -> pd.DataFrame:
        if not self.ledger.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        return pd.read_csv(self.ledger, dtype=str).reindex(columns=LEDGER_COLUMNS)

    def remaining(self) -> int:
        return min(self.cap_total - self.prior - self.used, self.max_run - self.used)

    def can_spend(self, n: int) -> bool:
        return self.remaining() >= n

    def charge(self, n: int) -> None:
        self.used += int(n)

    def commit(self, notes: str = "") -> None:
        """Relee el ledger y reemplaza SOLO la fila de esta corrida: dos corridas en paralelo (p. ej. T1 y T2)
        no se pisan. `images_cumulative` = suma de todas las filas en ese momento."""
        if self.used == 0:
            return
        others = self._read()
        others = others[others["run_id"].astype(str) != self.row["run_id"]]
        self.prior = int(pd.to_numeric(others["images_requested"], errors="coerce").fillna(0).sum())
        cum = self.prior + self.used                 # el tope total ve también lo gastado por la otra corrida
        rec = {**self.row, "date_utc": utcnow()[:10], "images_requested": self.used, "images_cumulative": cum,
               "usd_est_no_quota": round(cum / 1000 * self.usd_per_1000, 2), "notes": notes}
        df = pd.concat([others, pd.DataFrame([rec])], ignore_index=True)[LEDGER_COLUMNS]
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_suffix(f".{os.getpid()}.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.ledger)


def cap_total_images(cfg: dict) -> int:
    c = cfg["cost"]
    return int(c["cap_usd_total"] / c["usd_per_1000_static"] * 1000)


# ─── cliente HTTP ─────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, per_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.dt, self.clock, self.sleep, self.last = 1.0 / per_s, clock, sleep, -1e9

    def wait(self) -> None:
        gap = self.clock() - self.last
        if gap < self.dt:
            self.sleep(self.dt - gap)
        self.last = self.clock()


@dataclass
class StreetViewClient:
    """Envuelve las dos llamadas de la API. `session` es inyectable (tests usan un falso)."""
    camera: dict
    network: dict
    api_key: str
    session: object = None
    sleep: Callable[[float], None] = time.sleep
    limiter: RateLimiter = field(init=False)

    def __post_init__(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        self.limiter = RateLimiter(self.network["max_requests_per_s"], sleep=self.sleep)

    def _get(self, url: str, params: dict) -> tuple[int, bytes, int]:
        """GET con reintentos (backoff exponencial) en timeouts, 429 y 5xx. Devuelve (status, body, intentos)."""
        attempts, status, body = 0, -1, b""
        for attempts in range(1, self.network["max_retries"] + 2):
            self.limiter.wait()
            try:
                r = self.session.get(url, params={**params, "key": self.api_key},
                                     timeout=self.network["timeout_s"])
                status, body = r.status_code, r.content
            except Exception as e:                         # timeout / conexión
                status, body = -1, repr(e).encode()
            if status == 200 or status in (400, 403, 404):
                break
            self.sleep(self.network["backoff_base_s"] ** attempts)
        return status, body, attempts

    def metadata(self, lat: float, lon: float) -> tuple[dict, int]:
        status, body, att = self._get(METADATA_URL, {"location": f"{lat},{lon}",
                                                     "radius": self.camera["radius_m"],
                                                     "source": self.camera["source"]})
        try:
            meta = json.loads(body.decode()) if status == 200 else {"status": f"HTTP_{status}"}
        except ValueError:
            meta = {"status": "BAD_JSON"}
        meta.pop("error_message", None)
        return meta, att

    def image(self, lat: float, lon: float, heading: int, pano: str | None = None) -> tuple[int, bytes, int]:
        params = {"size": self.camera["size"], "heading": heading, "pitch": self.camera["pitch"],
                  "fov": self.camera["fov"]}
        if self.camera["request_by"] == "pano" and pano:
            params["pano"] = pano
        else:
            params["location"] = f"{lat},{lon}"
            params["radius"] = self.camera["radius_m"]
        if self.camera["source"] != "default":
            params["source"] = self.camera["source"]
        if self.camera.get("return_error_code"):
            params["return_error_code"] = "true"
        return self._get(STATIC_URL, params)


# ─── manifiestos y validación de una vista ────────────────────────────────────

VIEW_COLUMNS = ["spot_id", "view_id", "heading_deg", "pitch_deg", "fov_deg", "path", "image_width",
                "image_height", "bytes", "sha256", "provider", "pano_id", "capture_date_if_known",
                "download_status", "retrieval_attempts", "http_status", "requested_at_utc"]
SPOT_MANIFEST_COLUMNS = SPOT_COLUMNS + ["pano_id", "lat_pano", "lon_pano", "capture_date_if_known",
                                        "metadata_status", "metadata_attempts", "n_views_ok", "status",
                                        "provider", "camera_fingerprint", "queried_at_utc", "dup_of", "dup_reason"]


def inspect_image(blob: bytes) -> dict:
    """Decodifica con PIL; detecta corrupción y placeholders casi uniformes (desv. estándar baja)."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(blob))
        im.load()
    except Exception as e:
        return {"ok": False, "reason": f"corrupt:{type(e).__name__}"}
    arr = np.asarray(im.convert("L"), dtype=np.float32)
    return {"ok": True, "width": im.width, "height": im.height, "mode": im.mode,
            "std": float(arr.std()), "mean": float(arr.mean())}


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


class Manifest:
    """CSV de spots / vistas / incidencias en `<dataset>/_manifest/`. Escritura atómica."""

    def __init__(self, folder: Path):
        self.dir = folder / "_manifest"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.views = self._load("views.csv", VIEW_COLUMNS)
        self.spots = self._load("spots.csv", SPOT_MANIFEST_COLUMNS)
        self.incidents: list[dict] = self._load("incidents.csv", ["at_utc", "spot_id", "kind", "detail"]).to_dict("records")

    def _load(self, name, cols) -> pd.DataFrame:
        p = self.dir / name
        if not p.exists():
            return pd.DataFrame(columns=cols)
        return pd.read_csv(p, dtype={"spot_id": str, "pano_id": str, "dup_of": str}).reindex(columns=cols)

    def valid_view(self, spot_id: str, heading: int, path: Path, min_bytes: int) -> bool:
        """Una vista es válida si el archivo existe, supera el tamaño mínimo y su hash coincide con el manifiesto."""
        if not path.exists() or path.stat().st_size < min_bytes:
            return False
        row = self.views[(self.views["spot_id"].astype(str) == str(spot_id)) &
                         (self.views["heading_deg"].astype(int) == int(heading)) &
                         (self.views["download_status"] == "ok")]
        return len(row) > 0 and row["sha256"].iloc[-1] == sha256(path.read_bytes())

    @staticmethod
    def _replace(df: pd.DataFrame, mask: pd.Series, rec: dict) -> pd.DataFrame:
        keep = df[~mask]
        new = pd.DataFrame([rec])
        return new if keep.empty else pd.concat([keep, new], ignore_index=True)

    def upsert_view(self, rec: dict) -> None:
        m = (self.views["spot_id"].astype(str) == str(rec["spot_id"])) & \
            (self.views["heading_deg"].astype(int) == int(rec["heading_deg"]))
        self.views = self._replace(self.views, m, rec)

    def upsert_spot(self, rec: dict) -> None:
        self.spots = self._replace(self.spots, self.spots["spot_id"].astype(str) == str(rec["spot_id"]), rec)

    def incident(self, spot_id: str, kind: str, detail: str) -> None:
        self.incidents.append({"at_utc": utcnow(), "spot_id": spot_id, "kind": kind, "detail": detail})

    def save(self) -> None:
        for name, df in (("views.csv", self.views.reindex(columns=VIEW_COLUMNS)),
                         ("spots.csv", self.spots.reindex(columns=SPOT_MANIFEST_COLUMNS)),
                         ("incidents.csv", pd.DataFrame(self.incidents, columns=["at_utc", "spot_id", "kind", "detail"]))):
            tmp = self.dir / (name + ".tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, self.dir / name)


# ─── recolección ──────────────────────────────────────────────────────────────

def collect(cfg: dict, spots: pd.DataFrame, folder: Path, client: StreetViewClient,
            limit: int | None = None, skip_no_coverage: bool = True,
            include_outside_district: bool = False, budget: Budget | None = None,
            dedup_coords: bool = True, dedup_pano: bool = True) -> dict:
    """Recolecta en orden, idempotente: una vista válida (archivo + hash del manifiesto) no se vuelve a pedir.

    Por defecto solo se piden spots dentro del polígono del distrito. El histórico NO filtraba
    (`include_outside_district=True` lo replica literalmente). Con `dedup_coords`, un spot con la misma
    lat/lon que otro anterior (varios crímenes en el mismo punto) no se pide: queda en el manifiesto con
    `status=dup` y `dup_of` = spot canónico (las imágenes serían idénticas). Con `dedup_pano`, ídem si la
    metadata (gratis) devuelve un `pano_id` ya descargado completo por otro spot: verificado en San Isidro
    (2026-10-29), 378/420 vistas de spots con el mismo pano eran idénticas byte a byte y las 42 restantes
    diferían < 0.01 niveles de gris (solo la codificación JPEG). `dup_reason` ∈ {coord, pano}. Con `budget`, se detiene
    antes de superar el tope de imágenes facturables (`stats["budget_stopped"]`).
    """
    cam, st = cfg["camera"], cfg["storage"]
    hs = headings(cam)
    fp = camera_fingerprint(cam)
    man = Manifest(folder)
    prev_fp = set(man.spots["camera_fingerprint"].dropna().astype(str)) - {""}
    if prev_fp and prev_fp != {fp}:
        raise RuntimeError(f"{folder.name}: el manifiesto tiene huella {sorted(prev_fp)} ≠ {fp}; "
                           f"no se mezclan criterios de cámara en un dataset")
    todo = spots if include_outside_district else spots[spots["in_district"].astype(bool)]
    if limit:
        todo = todo.head(limit)
    stats = {"spots_attempted": 0, "spots_no_coverage": 0, "spots_dup": 0, "views_downloaded": 0,
             "views_skipped_valid": 0, "views_failed": 0, "aborted": False, "budget_stopped": False}
    worst_case = cfg["network"]["max_retries"] + 1
    canonical: dict[tuple[float, float], str] = {}
    done = man.spots[man.spots["status"] == "complete"]
    pano_canonical: dict[str, str] = {}
    for pid, s0 in zip(done["pano_id"].fillna("").astype(str), done["spot_id"].astype(str)):
        if pid:
            pano_canonical.setdefault(pid, s0)
    consecutive = 0
    for _, s in todo.iterrows():
        sid, lat, lon = str(s["spot_id"]), float(s["lat_req"]), float(s["lon_req"])
        ck = (round(lat, 7), round(lon, 7))
        if dedup_coords and ck in canonical:
            stats["spots_dup"] += 1
            man.upsert_spot({**{k: s[k] for k in SPOT_COLUMNS}, "status": "dup", "dup_of": canonical[ck],
                             "dup_reason": "coord", "n_views_ok": 0, "provider": cam["provider"], "camera_fingerprint": fp,
                             "queried_at_utc": utcnow()})
            continue
        canonical[ck] = sid
        sdir = folder / sid
        paths = {h: sdir / st["view_filename"].format(heading=h) for h in hs}
        if all(man.valid_view(sid, h, paths[h], st["min_valid_bytes"]) for h in hs):
            stats["views_skipped_valid"] += len(hs)
            continue
        stats["spots_attempted"] += 1
        meta, matt = client.metadata(lat, lon)
        loc = meta.get("location", {}) or {}
        spot_rec = {**{k: s[k] for k in SPOT_COLUMNS}, "pano_id": meta.get("pano_id", ""),
                    "lat_pano": loc.get("lat", np.nan), "lon_pano": loc.get("lng", np.nan),
                    "capture_date_if_known": meta.get("date", ""), "metadata_status": meta.get("status", ""),
                    "metadata_attempts": matt, "provider": cam["provider"], "camera_fingerprint": fp,
                    "queried_at_utc": utcnow()}
        if meta.get("status") != "OK":
            man.incident(sid, "metadata", str(meta.get("status")))
            if skip_no_coverage and meta.get("status") == "ZERO_RESULTS":
                stats["spots_no_coverage"] += 1
                man.upsert_spot({**spot_rec, "n_views_ok": 0, "status": "no_coverage"})
                man.save()
                continue
            if str(meta.get("status", "")).startswith("HTTP_") or meta.get("status") in ("REQUEST_DENIED",
                                                                                          "OVER_QUERY_LIMIT"):
                consecutive += 1
        pid = str(spot_rec["pano_id"] or "")
        if dedup_pano and pid and pano_canonical.get(pid, sid) != sid:
            stats["spots_dup"] += 1
            man.upsert_spot({**spot_rec, "status": "dup", "dup_of": pano_canonical[pid], "dup_reason": "pano",
                             "n_views_ok": 0})
            man.save()
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for h in hs:
            if man.valid_view(sid, h, paths[h], st["min_valid_bytes"]):
                n_ok += 1
                stats["views_skipped_valid"] += 1
                continue
            if budget is not None and not budget.can_spend(worst_case):
                stats["budget_stopped"] = True
                break
            status, blob, att = client.image(lat, lon, h, spot_rec["pano_id"] or None)
            if budget is not None:
                budget.charge(att)
            rec = {"spot_id": sid, "view_id": f"{sid}_h{h:03d}", "heading_deg": h, "pitch_deg": cam["pitch"],
                   "fov_deg": cam["fov"], "path": paths[h].relative_to(folder).as_posix(),
                   "image_width": np.nan, "image_height": np.nan, "bytes": len(blob), "sha256": "",
                   "provider": cam["provider"], "pano_id": spot_rec["pano_id"],
                   "capture_date_if_known": spot_rec["capture_date_if_known"], "download_status": "failed",
                   "retrieval_attempts": att, "http_status": status, "requested_at_utc": utcnow()}
            info = inspect_image(blob) if status == 200 else {"ok": False, "reason": f"http_{status}"}
            if status == 200 and info["ok"] and len(blob) >= st["min_valid_bytes"]:
                tmp = paths[h].with_suffix(".part")
                tmp.write_bytes(blob)
                os.replace(tmp, paths[h])
                rec.update(download_status="ok", sha256=sha256(blob), image_width=info["width"],
                           image_height=info["height"])
                n_ok += 1
                consecutive = 0
                stats["views_downloaded"] += 1
            else:
                reason = info.get("reason") or f"too_small:{len(blob)}"
                rec["download_status"] = reason
                man.incident(sid, "view", f"h={h} {reason}")
                stats["views_failed"] += 1
                consecutive += 1
            man.upsert_view(rec)
            if consecutive >= cfg["network"]["max_consecutive_errors"]:
                stats["aborted"] = True
                break
        man.upsert_spot({**spot_rec, "n_views_ok": n_ok,
                         "status": "complete" if n_ok == len(hs) else ("partial" if n_ok else "failed")})
        if n_ok == len(hs) and pid:
            pano_canonical.setdefault(pid, sid)
        man.save()
        if budget is not None:
            budget.commit()
        if stats["aborted"]:
            log.error("abortado: %d errores consecutivos (¿clave/cuota?)", consecutive)
            break
        if stats["budget_stopped"]:
            log.warning("tope de presupuesto alcanzado: quedan %d imágenes", budget.remaining())
            break
    man.save()
    if budget is not None:
        stats["billable_images_this_run"] = budget.used
        stats["billable_images_cumulative"] = budget.prior + budget.used
        budget.commit()
    return stats


# ─── QA ───────────────────────────────────────────────────────────────────────

def average_hash(blob: bytes, size: int = 8) -> int:
    """aHash 8×8 (sin dependencias): bit = píxel > media. Distancia de Hamming ≤ 5 ≈ casi duplicado."""
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("L").resize((size, size))
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def count_near_duplicates(hashes: np.ndarray, groups: np.ndarray, max_hamming: int) -> int:
    """Pares (i < j) de grupos distintos con distancia de Hamming ≤ max_hamming entre hashes de 64 bits."""
    n, total = len(hashes), 0
    for i in range(n - 1):
        x = np.bitwise_xor(hashes[i + 1:], hashes[i])
        d = _POPCOUNT8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        total += int(((d <= max_hamming) & (groups[i + 1:] != groups[i])).sum())
    return total


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def verify(cfg: dict, spots_plan: pd.DataFrame, folder: Path, near_dup_hamming: int = 5,
           blank_std: float = 3.0) -> dict:
    """QA del dataset descargado. Lee SOLO archivos reales del disco y los manifiestos."""
    hs = headings(cfg["camera"])
    man = Manifest(folder)
    V, S = man.views.copy(), man.spots.copy()
    rep: dict = {"dataset": folder.name, "n_spots_planned": int(len(spots_plan)),
                 "n_spots_planned_in_district": int(spots_plan["in_district"].astype(bool).sum()),
                 "n_spots_in_manifest": int(len(S)), "expected_headings": hs}
    if not len(V):
        rep["status"] = "SIN_DATOS: no hay vistas descargadas (¿falta la clave o no se ejecutó collect?)"
        return rep
    ok = V[V["download_status"] == "ok"].copy()
    rep["views_ok"] = int(len(ok))
    per_spot = ok.groupby("spot_id")["heading_deg"].nunique()
    rep["spots_12_of_12"] = int((per_spot == len(hs)).sum())
    rep["spots_partial"] = int(((per_spot > 0) & (per_spot < len(hs))).sum())
    rep["spots_no_coverage"] = int((S["status"] == "no_coverage").sum())
    complete = set(per_spot[per_spot == len(hs)].index.astype(str))
    dups = S[S["status"] == "dup"]
    rep["spots_dup_of"] = int(len(dups))
    rep["spots_dup_of_complete"] = int(dups["dup_of"].astype(str).isin(complete).sum())
    rep["coverage_rate_vs_planned_in_district"] = round(
        (rep["spots_12_of_12"] + rep["spots_dup_of_complete"]) / max(1, rep["n_spots_planned_in_district"]), 4)
    rep["headings_outside_expected"] = int((~ok["heading_deg"].astype(int).isin(hs)).sum())
    rep["duplicate_spot_view_ids"] = int(ok["view_id"].duplicated().sum())
    rep["spots_outside_district"] = int((~S["in_district"].astype(bool)).sum()) if len(S) else 0
    # recorrido de archivos reales
    sizes, blanks, missing, corrupt, ahash = set(), [], [], [], {}
    for _, r in ok.iterrows():
        p = folder / r["path"]
        if not p.exists():
            missing.append(r["view_id"])
            continue
        blob = p.read_bytes()
        info = inspect_image(blob)
        if not info["ok"]:
            corrupt.append(r["view_id"])
            continue
        sizes.add((info["width"], info["height"], info["mode"]))
        if info["std"] < blank_std:
            blanks.append(r["view_id"])
        ahash[r["view_id"]] = average_hash(blob)
    rep.update(files_missing=len(missing), files_corrupt=len(corrupt), near_uniform_images=len(blanks),
               distinct_size_mode=sorted(map(list, sizes)))
    rep["exact_duplicate_hashes_across_spots"] = int(
        ok.groupby("sha256")["spot_id"].nunique().gt(1).sum())
    # casi duplicados ENTRE spots distintos (dentro del mismo spot el solape angular es esperado)
    keys = list(ahash)
    spot_of = ok.set_index("view_id")["spot_id"].astype(str)
    rep["near_duplicate_view_pairs_across_spots"] = count_near_duplicates(
        np.array([ahash[k] for k in keys], dtype=np.uint64),
        spot_of.loc[keys].to_numpy(), near_dup_hamming)
    # reutilización de pano entre spots
    Sp = S[S["pano_id"].fillna("") != ""]
    reuse = Sp.groupby("pano_id")["spot_id"].nunique()
    rep["panos_unique"] = int(Sp["pano_id"].nunique())
    rep["panos_shared_by_multiple_spots"] = int((reuse > 1).sum())
    if len(Sp):
        d = haversine_m(Sp["lat_req"].astype(float), Sp["lon_req"].astype(float),
                        Sp["lat_pano"].astype(float), Sp["lon_pano"].astype(float))
        rep["pano_offset_m_median"] = round(float(np.nanmedian(d)), 2)
        rep["pano_offset_m_p90"] = round(float(np.nanpercentile(d, 90)), 2)
        dates = Sp["capture_date_if_known"].dropna().astype(str)
        rep["capture_years"] = dates.str[:4].value_counts().sort_index().to_dict()
    rep["status"] = "OK"
    return rep


def mosaic(folder: Path, spot_ids: list[str], hs: list[int], out: Path, view_filename: str) -> Path | None:
    """Mosaico de las vistas REALES de unos spots (sin miniaturas inventadas: falta → recuadro vacío rotulado)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if not spot_ids:
        return None
    fig, axes = plt.subplots(len(spot_ids), len(hs), figsize=(1.6 * len(hs), 1.2 * len(spot_ids)), squeeze=False)
    for r, sid in enumerate(spot_ids):
        for c, h in enumerate(hs):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            p = folder / sid / view_filename.format(heading=h)
            if p.exists():
                ax.imshow(Image.open(p))
            else:
                ax.text(0.5, 0.5, "falta", ha="center", va="center", fontsize=7)
            if r == 0:
                ax.set_title(f"{h}°", fontsize=7)
        axes[r, 0].set_ylabel(sid, fontsize=6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _plan_cached(cfg, district, source, year, out_dir: Path, refresh: bool) -> pd.DataFrame:
    p = out_dir / "spots_plan.csv"
    if p.exists() and not refresh:
        S = pd.read_csv(p, dtype={"spot_id": str, "segment_key": str})
        S["segment_key"] = S["segment_key"].fillna("")
        return S
    S = plan_spots(cfg, district, source, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    S.to_csv(p, index=False)
    return S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["plan", "collect", "verify", "fingerprint"])
    ap.add_argument("--district", choices=["barranco", "la_victoria", "san_isidro", "lince"])
    ap.add_argument("--spot-source", default="street_sampling", choices=["crime_points", "street_sampling", "grid"])
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="planifica y estima costo; no hace peticiones")
    ap.add_argument("--limit", type=int, default=None, help="máximo de spots a recolectar en esta corrida")
    ap.add_argument("--refresh-plan", action="store_true")
    ap.add_argument("--coverage-rate", type=float, default=1.0, help="supuesto de cobertura para el costo")
    ap.add_argument("--include-outside-district", action="store_true",
                    help="pide también spots fuera del polígono (comportamiento literal del histórico)")
    ap.add_argument("--max-billable-images", type=int, default=None,
                    help="OBLIGATORIO en collect real: tope de imágenes facturables de esta corrida "
                         "(además del tope total cost.cap_usd_total del ledger)")
    ap.add_argument("--tramo", default="", help="etiqueta del ledger (T1, T2, ...)")
    ap.add_argument("--no-dedup", action="store_true", help="pide también spots con coordenada o panorama repetidos")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    loaded = load_env_file()
    if loaded:
        log.info("variables cargadas de .env: %s", ", ".join(loaded))

    if args.command == "fingerprint":
        rep = {"camera_fingerprint": camera_fingerprint(cfg["camera"]), "headings": headings(cfg["camera"]),
               "camera": {k: cfg["camera"][k] for k in CAMERA_KEYS}}
        print(json.dumps(rep, indent=2))
        return 0
    if not args.district:
        ap.error("--district es obligatorio")

    if args.spot_source == "crime_points" and not cfg["districts"][args.district].get("crime_name"):
        log.error("%s: no hay registros de crimen en distritos/ — crime_points no es replicable; "
                  "use --spot-source street_sampling o grid", args.district)
        return 2
    root = REPO_ROOT / cfg["storage"]["root"]
    folder = root / dataset_folder(cfg, args.district, args.spot_source, args.year)
    S = _plan_cached(cfg, args.district, args.spot_source, args.year, folder / "_manifest", args.refresh_plan)
    n_in = int(S["in_district"].astype(bool).sum())

    if args.command == "plan" or (args.command == "collect" and args.dry_run):
        rep = {"district": args.district, "spot_source": args.spot_source, "dataset_folder": folder.name,
               "n_spots_planned": int(len(S)), "n_spots_in_district": n_in,
               "n_spots_outside_district": int(len(S) - n_in),
               "camera_fingerprint": camera_fingerprint(cfg["camera"]),
               "cost": estimate_cost(n_in, cfg, args.coverage_rate),
               "api_key_present": bool(os.environ.get(KEY_ENV))}
        (folder / "_manifest" / "dry_run.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    if args.command == "collect":
        key = os.environ.get(KEY_ENV)
        if not key:
            log.error("Falta %s en el entorno: no se hace ninguna petición. Use --dry-run.", KEY_ENV)
            return 2
        if args.max_billable_images is None:
            log.error("Falta --max-billable-images: no se gasta sin un tope explícito por corrida.")
            return 2
        budget = Budget(REPO_ROOT / cfg["cost"]["ledger"], cap_total_images(cfg), args.max_billable_images,
                        cfg["cost"]["usd_per_1000_static"], tramo=args.tramo, district=args.district,
                        spot_source=args.spot_source)
        log.info("presupuesto: ya gastadas %d · tope total %d · tope de la corrida %d · disponibles %d",
                 budget.prior, budget.cap_total, budget.max_run, budget.remaining())
        client = StreetViewClient(cfg["camera"], cfg["network"], key)
        stats = collect(cfg, S, folder, client, limit=args.limit,
                        include_outside_district=args.include_outside_district, budget=budget,
                        dedup_coords=not args.no_dedup, dedup_pano=not args.no_dedup)
        print(json.dumps(stats, indent=2))
        return 3 if stats["aborted"] else (4 if stats["budget_stopped"] else 0)

    rep = verify(cfg, S, folder)
    (folder / "_manifest" / "qa_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    man = Manifest(folder)
    done = man.spots[man.spots.get("status", pd.Series(dtype=str)) == "complete"]["spot_id"].astype(str).tolist()
    if done:
        rng = np.random.default_rng(42)
        pick = sorted(rng.choice(done, size=min(6, len(done)), replace=False).tolist())
        mosaic(folder, pick, headings(cfg["camera"]), folder / "_manifest" / "qa_mosaic.png",
               cfg["storage"]["view_filename"])
    print(json.dumps(rep, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
