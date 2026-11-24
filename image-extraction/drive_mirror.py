"""
Espejo local de las imágenes HISTÓRICAS de Barranco / La Victoria desde la carpeta pública de Google Drive del autor.

POR QUE
-------
Las vistas de 2016 de B/LV solo estaban en local como embeddings (lima_views.py). La carpeta de Drive
"Perpection" contiene `Inseguros-Barranco-GGZ-2016/<spot>/heading_*.jpg` e `Inseguros-La_Victoria-GGZ-2016/...`,
las mismas rutas que `external_assets/embeddings_export/metadata.json`. Con ellas se pueden extraer DINOv2 / BoVW de
parches en los 4 distritos y usar B/LV como estímulos de la encuesta.

    python image-extraction/drive_mirror.py list      # árbol → data/_drive/<carpeta>_files.csv
    python image-extraction/drive_mirror.py download  # descarga reanudable y paralela → data/<carpeta>/<spot>/heading_*.jpg
    (--headings 0,90,180,270 --spots-per-district N descarga solo un subconjunto)

Solo `heading_*.jpg`; escritura atómica; un archivo existente con tamaño > 0 que abre como JPEG no se vuelve a bajar.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT_ID = "1TUr3R36_8z27b82jhIep_I_eGS7ElSGH"
FOLDERS = {"Inseguros-Barranco-GGZ-2016": "1SSeRP6g7_fhhenOkitkypZJ9IdM24wgl",
           "Inseguros-La_Victoria-GGZ-2016": "1SG78wsa3pAy7f9ZPxz8NlUQMD3qjnbvw"}
DATA = Path(__file__).resolve().parent / "data"
LISTS = DATA / "_drive"
log = logging.getLogger("drive_mirror")


def session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def list_folder(s, fid: str):
    import gdown  # noqa: F401  (registra el submódulo)
    mod = sys.modules["gdown.download_folder"]
    for attempt in range(4):
        try:
            return mod._parse_embedded_folder_view(sess=s, folder_id=fid, verify=True, timeout=30)[1]
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"no se pudo listar {fid}")


def cmd_list(threads: int) -> None:
    LISTS.mkdir(parents=True, exist_ok=True)
    s = session()
    for folder, fid in FOLDERS.items():
        spots = [(i, n) for i, n, t in list_folder(s, fid) if t.endswith("folder")]
        rows = []
        with ThreadPoolExecutor(threads) as ex:
            futs = {ex.submit(list_folder, session(), sid): name for sid, name in spots}
            for f in as_completed(futs):
                for i, n, t in f.result():
                    if n.startswith("heading_") and n.lower().endswith(".jpg"):
                        rows.append({"folder": folder, "spot_id": futs[f], "file": n, "file_id": i,
                                     "heading": int(n.split("_")[1].split(".")[0])})
        df = pd.DataFrame(rows).sort_values(["spot_id", "heading"])
        tmp = LISTS / f"{folder}_files.tmp.csv"
        df.to_csv(tmp, index=False)
        os.replace(tmp, LISTS / f"{folder}_files.csv")
        log.info("%s: %d spots, %d vistas heading_*", folder, df["spot_id"].nunique(), len(df))


def fetch(s, row) -> str:
    from PIL import Image
    out = DATA / row.folder / row.spot_id / row.file
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    url = f"https://drive.google.com/uc?export=download&id={row.file_id}"
    for attempt in range(4):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff":
                Image.open(io.BytesIO(r.content)).verify()
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                os.replace(tmp, out)
                return "ok"
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return "fail"


def cmd_download(threads: int, headings: list[int] | None, n_spots: int | None, seed: int) -> None:
    import threading
    local = threading.local()

    def sess():
        if not hasattr(local, "s"):
            local.s = session()
        return local.s

    for folder in FOLDERS:
        F = pd.read_csv(LISTS / f"{folder}_files.csv", dtype={"spot_id": str, "file_id": str})
        if headings:
            F = F[F["heading"].isin(headings)]
        if n_spots:
            keep = pd.Series(F["spot_id"].unique()).sample(min(n_spots, F["spot_id"].nunique()), random_state=seed)
            F = F[F["spot_id"].isin(set(keep))]
        cnt, t0 = {"ok": 0, "skip": 0, "fail": 0}, time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            for k, st in enumerate(as_completed([ex.submit(lambda r: fetch(sess(), r), r) for r in F.itertuples()]), 1):
                cnt[st.result()] += 1
                if k % 500 == 0:
                    log.info("  %s %d/%d %s %.1f min", folder, k, len(F), cnt, (time.perf_counter() - t0) / 60)
        log.info("%s: %s en %.1f min", folder, cnt, (time.perf_counter() - t0) / 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "download"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--headings", default=None, help="p. ej. 0,90,180,270")
    ap.add_argument("--spots-per-district", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.command == "list":
        cmd_list(a.threads)
    else:
        cmd_download(a.threads, [int(h) for h in a.headings.split(",")] if a.headings else None,
                     a.spots_per_district, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
