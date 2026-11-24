"""
Cliente mínimo de la Graph API de Mapillary (solo METADATA de imágenes; no descarga imágenes).

POR QUE
-------
EXP-046 necesita saber cuántas imágenes de Mapillary hay por segmento de calle (en Place Pulse y en los 4 distritos
de Lima) antes de decidir si se densifica el contexto de PP con ellas (D6 del prompt maestro 2026-10-29). La API
`/images` devuelve como mucho `limit` (≤ 2,000) imágenes por bbox, así que se tesela el área y una tesela que vuelve
llena se divide en 4 hasta `max_depth` (si aún vuelve llena se registra como TRUNCADA: el conteo es una cota inferior).

    python image-extraction/mapillary_client.py bbox --bbox=-77.040,-12.095,-77.030,-12.085 --out data/_mapillary/x.csv.gz

El token se lee SOLO de MAPILLARY_ACCESS_TOKEN (o de image-extraction/.env, ignorado por Git) y nunca se escribe.
Licencia de las imágenes: CC-BY-SA 4.0 (atribución a los autores de Mapillary si alguna vez se publican).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streetview_collector import RateLimiter, load_env_file  # noqa: E402

IMAGES_URL = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_ACCESS_TOKEN"
FIELDS = ["id", "geometry", "captured_at", "compass_angle", "is_pano", "camera_type", "sequence", "quality_score"]
COLUMNS = ["id", "lon", "lat", "captured_at", "year", "compass_angle", "is_pano", "camera_type", "sequence",
           "quality_score"]

log = logging.getLogger("mapillary_client")


@dataclass
class MapillaryClient:
    token: str
    session: object = None
    per_s: float = 5.0
    limit: int = 2000
    max_retries: int = 2
    backoff_base_s: float = 2.0
    timeout_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    limiter: RateLimiter = field(init=False)
    truncated: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    requests_made: int = 0

    def __post_init__(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        self.limiter = RateLimiter(self.per_s, sleep=self.sleep)

    def _get(self, params: dict) -> list[dict]:
        last = ""
        for attempt in range(1, self.max_retries + 2):
            self.limiter.wait()
            self.requests_made += 1
            try:
                r = self.session.get(IMAGES_URL, params={**params, "access_token": self.token},
                                     timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.json().get("data", [])
                last = f"HTTP_{r.status_code}"
                if r.status_code in (400, 401, 403):
                    break
            except Exception as e:                          # timeout / conexión / JSON
                last = type(e).__name__
            self.sleep(self.backoff_base_s ** attempt)
        raise RuntimeError(f"Mapillary /images falló tras reintentos: {last}")

    def images_in_bbox(self, bbox: tuple[float, float, float, float], depth: int = 0, max_depth: int = 6) -> list[dict]:
        """Todas las imágenes de un bbox (min_lon, min_lat, max_lon, max_lat), dividiendo teselas llenas.

        Un HTTP 5xx persistente en una tesela densa (la API no llega a responder) también se resuelve dividiéndola;
        si ni a `max_depth` responde, la tesela queda en `failed` (conteo = cota inferior) y se sigue.
        """
        try:
            data = self._get({"bbox": ",".join(f"{v:.6f}" for v in bbox), "fields": ",".join(FIELDS),
                              "limit": self.limit})
        except RuntimeError as e:
            if "HTTP_4" in str(e):
                raise
            if depth >= max_depth:
                self.failed.append(bbox)
                return []
            data = [None] * self.limit                     # tratar como tesela llena → dividir
        if len(data) < self.limit:
            return data
        if depth >= max_depth:
            self.truncated.append(bbox)
            return data
        x0, y0, x1, y1 = bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        out = []
        for q in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            out += self.images_in_bbox(q, depth + 1, max_depth)
        return out


def to_frame(data: list[dict]) -> pd.DataFrame:
    rows = []
    for d in data:
        lon, lat = (d.get("geometry") or {}).get("coordinates", (math.nan, math.nan))
        ts = d.get("captured_at")
        rows.append({"id": str(d.get("id")), "lon": lon, "lat": lat, "captured_at": ts,
                     "year": pd.to_datetime(ts, unit="ms", utc=True).year if ts else pd.NA,
                     "compass_angle": d.get("compass_angle"), "is_pano": d.get("is_pano"),
                     "camera_type": d.get("camera_type"), "sequence": d.get("sequence"),
                     "quality_score": d.get("quality_score")})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.drop_duplicates("id").reset_index(drop=True)


def tiles_around(lonlat: list[tuple[float, float]], tile_deg: float, halo: int = 1) -> list[tuple[float, float, float, float]]:
    """Teselas de una rejilla de `tile_deg` que contienen algún punto, más `halo` teselas alrededor."""
    keys = set()
    for lon, lat in lonlat:
        i, j = math.floor(lon / tile_deg), math.floor(lat / tile_deg)
        for di in range(-halo, halo + 1):
            for dj in range(-halo, halo + 1):
                keys.add((i + di, j + dj))
    return [(i * tile_deg, j * tile_deg, (i + 1) * tile_deg, (j + 1) * tile_deg) for i, j in sorted(keys)]


def tiles_bbox(bbox: tuple[float, float, float, float], tile_deg: float) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = bbox
    return [(max(x0, i * tile_deg), max(y0, j * tile_deg), min(x1, (i + 1) * tile_deg), min(y1, (j + 1) * tile_deg))
            for i in range(math.floor(x0 / tile_deg), math.ceil(x1 / tile_deg))
            for j in range(math.floor(y0 / tile_deg), math.ceil(y1 / tile_deg))]


def fetch_tiles(client: MapillaryClient, tiles: list, out: Path, checkpoint_every: int = 10) -> pd.DataFrame:
    """Recorre teselas con reanudación: `<out>.tiles.json` guarda las teselas hechas; escritura atómica."""
    done_path = out.with_suffix(".tiles.json")
    done = set(map(tuple, json.loads(done_path.read_text()))) if done_path.exists() else set()
    parts = [pd.read_csv(out, dtype={"id": str})] if out.exists() and done else []
    buf = []

    def flush():
        nonlocal parts, buf
        if buf:
            parts.append(to_frame(buf))
            buf = []
        df = pd.concat(parts, ignore_index=True).drop_duplicates("id") if parts else pd.DataFrame(columns=COLUMNS)
        parts = [df]
        tmp = out.with_suffix(".tmp.csv.gz")
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, out)
        tmpd = done_path.with_suffix(".tmp")
        tmpd.write_text(json.dumps(sorted(done)))
        os.replace(tmpd, done_path)
        return df

    todo = [t for t in tiles if tuple(round(v, 6) for v in t) not in done]
    log.info("%s: %d teselas (%d ya hechas)", out.name, len(tiles), len(tiles) - len(todo))
    for n, t in enumerate(todo, 1):
        buf += client.images_in_bbox(t)
        done.add(tuple(round(v, 6) for v in t))
        if n % checkpoint_every == 0:
            flush()
            log.info("  %d/%d teselas, %d peticiones", n, len(todo), client.requests_made)
    df = flush()
    if client.truncated:
        (out.with_suffix(".truncated.json")).write_text(json.dumps(client.truncated))
    if client.failed:
        (out.with_suffix(".failed.json")).write_text(json.dumps(client.failed))
    return df


def client_from_env(**kw) -> MapillaryClient:
    load_env_file()
    tok = os.environ.get(TOKEN_ENV)
    if not tok:
        raise SystemExit(f"Falta {TOKEN_ENV} (entorno o image-extraction/.env)")
    return MapillaryClient(tok, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["bbox"])
    ap.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    ap.add_argument("--tile-deg", type=float, default=0.005)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bbox = tuple(float(v) for v in args.bbox.split(","))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_tiles(client_from_env(), tiles_bbox(bbox, args.tile_deg), args.out)
    print(json.dumps({"images": int(len(df)), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
