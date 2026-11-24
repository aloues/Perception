"""
EXP-048 (E4) — Place Pulse densificado con Mapillary en Santiago: ¿al grafo le faltaban datos o capacidad?

Diseño registrado ANTES de ejecutar (docs/EXPERIMENT_REGISTRY.md, 2026-10-29).

EXP-042 dio nulo con ~1 imagen de PP cada 150 m. Aquí el contexto de cada imagen de PP de Santiago son imágenes de
Mapillary (sin etiquetas) de su segmento y de los segmentos a 1–2 saltos; la etiqueta sigue siendo el voto de PP.

  select    snap de la metadata Mapillary de Santiago (EXP-046) a los segmentos; de perspectiva, ≤ 3 por secuencia,
            primero las más antiguas; ≤ 6 por segmento en el ego-1 de los segmentos con imágenes PP y ≤ 2 en los que
            solo están a 2 saltos (contexto de la 2.ª capa del GCN)
  download  URL de miniatura (Graph API, lotes de ids) y descarga a 256 px de lado corto, reanudable, 8 hilos
  embed     DINOv2 ViT-B/14 CLS con la misma transformación que PP (encoders.load_dinov2), L2
  models    E0 · M-ego0/1/2 · M-rand · M-radius · G1 GCN (+ sin aristas) · G2 CRF, mismo entrenador y pares que EXP-042

Uso:  python exp048_mapillary_context.py --stage select|download|embed|models
Imágenes en image-extraction/data/_mapillary_img/Santiago/ (CC-BY-SA 4.0, no Git). Salidas: outputs/exp048/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from utils import OUTPUTS, REPO_ROOT, log, set_seed

sys.path.insert(0, str(REPO_ROOT / "image-extraction"))

SEED = 42
CITY = os.environ.get("EXP_CITY", "Santiago")          # EXP-062: réplica en otras ciudades (EXP_CITY=Atlanta)
OUT = OUTPUTS / ("exp048" if CITY == "Santiago" else f"exp062_{CITY.lower()}")
MLY = REPO_ROOT / "image-extraction" / "data" / "_mapillary"
IMG = REPO_ROOT / "image-extraction" / "data" / "_mapillary_img" / CITY
PER_SEG_EGO1, PER_SEG_EGO2, PER_SEQUENCE = 6, 2, 3     # tope por segmento: ego ≤ 1 / solo ego-2
SHORT_SIDE = 256


def city_graph():
    import osmnx as ox

    from exp042_pp_segment_graph_models import GRAPHS
    from segment_graph import dual_adjacency, segments_table
    Gp = ox.project_graph(ox.convert.to_undirected(ox.load_graphml(GRAPHS / f"{CITY}_drive.graphml")))
    seg = segments_table(Gp)
    return Gp, seg, dual_adjacency(seg)


def stage_select() -> None:
    from exp046_mapillary_density import pp_pilot, snap_points
    from segment_graph import reach_matrices
    OUT.mkdir(parents=True, exist_ok=True)
    Gp, seg, A = city_graph()
    pp = pp_pilot()
    pp = pp[pp["city"] == CITY].reset_index(drop=True)
    seg_pp, _ = snap_points(seg["geometry"], pp[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
    src = np.unique(seg_pp[seg_pp >= 0])
    R = reach_matrices(A, src, 2)
    ego1, ego = np.unique(R[1].indices), np.unique(R[2].indices)
    M = pd.read_csv(MLY / f"{CITY}.csv.gz", dtype={"id": str, "sequence": str}).dropna(subset=["lon", "lat"])
    M = M[~M["is_pano"].astype(bool)].reset_index(drop=True)
    seg_m, _ = snap_points(seg["geometry"], M[["lon", "lat"]].to_numpy(), Gp.graph["crs"])
    M["seg_idx"] = seg_m
    M = M[np.isin(M["seg_idx"], ego)]
    M = M.sort_values(["seg_idx", "captured_at", "id"])
    M = M.groupby(["seg_idx", "sequence"], dropna=False).head(PER_SEQUENCE)
    M["rank"] = M.groupby("seg_idx").cumcount()
    M = M[M["rank"] < np.where(np.isin(M["seg_idx"], ego1), PER_SEG_EGO1, PER_SEG_EGO2)].drop(columns="rank")
    M = M.reset_index(drop=True)
    M.to_csv(OUT / "mapillary_selection.csv", index=False)
    info = dict(pp_images=len(pp), pp_snapped=int((seg_pp >= 0).sum()), pp_segments=len(src),
                ego1_segments=len(ego1), ego2_segments=len(ego), segments_with_selected=int(M["seg_idx"].nunique()),
                selected_images=len(M), year_median=float(M["year"].median()),
                pct_before_2016=100 * float((M["year"] < 2016).mean()))
    (OUT / "selection_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(info)


def stage_download(url_workers: int = 6, workers: int = 16) -> None:
    """Fase 1: URL de miniatura por lotes de 100 ids (paralelo, cacheadas en urls.csv). Fase 2: descarga concurrente."""
    from concurrent.futures import as_completed
    from io import BytesIO

    import requests
    from PIL import Image

    import mapillary_client as mc
    mc.load_env_file()
    token = os.environ[mc.TOKEN_ENV]
    S = pd.read_csv(OUT / "mapillary_selection.csv", dtype={"id": str})
    IMG.mkdir(parents=True, exist_ok=True)
    todo = [i for i in S["id"] if not (IMG / f"{i}.jpg").exists()]
    log(f"{len(S):,} seleccionadas, {len(todo):,} por descargar")
    t0 = time.perf_counter()
    upath = OUT / "thumb_urls.csv"
    urls = pd.read_csv(upath, dtype=str).set_index("id")["url"].to_dict() if upath.exists() else {}

    def thumbs(ids):
        for attempt in range(4):
            try:
                r = requests.get(mc.IMAGES_URL, params={"image_ids": ",".join(ids), "fields": "id,thumb_1024_url",
                                                        "access_token": token}, timeout=90)
                if r.status_code == 200:
                    return {d["id"]: d.get("thumb_1024_url") for d in r.json().get("data", [])}
            except Exception:
                pass
            time.sleep(2 ** attempt)
        return {}

    need = [i for i in todo if i not in urls]
    batches = [need[b:b + 100] for b in range(0, len(need), 100)]
    with ThreadPoolExecutor(url_workers) as ex:
        for k, got in enumerate(ex.map(thumbs, batches), 1):
            urls.update({i: u for i, u in got.items() if u})
            if k % 50 == 0 or k == len(batches):
                tmp = upath.with_suffix(".tmp")
                pd.DataFrame({"id": list(urls), "url": list(urls.values())}).to_csv(tmp, index=False)
                os.replace(tmp, upath)
                log(f"  URL: lote {k}/{len(batches)} · {len(urls):,} URL · {(time.perf_counter() - t0) / 60:.1f} min")

    def fetch(i):
        url = urls.get(i)
        if not url:
            return i, False
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    with Image.open(BytesIO(r.content)) as im:
                        im = im.convert("RGB")
                        s = SHORT_SIDE / min(im.size)
                        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
                        tmp = IMG / f"{i}.tmp.jpg"
                        im.save(tmp, quality=90)
                    os.replace(tmp, IMG / f"{i}.jpg")
                    return i, True
            except Exception:
                pass
            time.sleep(2 ** attempt)
        return i, False

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            i, ok = f.result()
            if not ok:
                failed.append(i)
            if k % 2000 == 0:
                log(f"  imágenes: {k:,}/{len(todo):,} · fallidas {len(failed)} · {(time.perf_counter() - t0) / 60:.1f} min")
    (OUT / "download_failed.json").write_text(json.dumps(failed), encoding="utf-8")
    log(f"descarga lista: {len(todo) - len(failed):,} nuevas, {len(failed)} fallidas, {(time.perf_counter() - t0) / 60:.1f} min")


def stage_embed(threads: int = 10, batch: int = 32) -> None:
    import torch
    from PIL import Image

    from encoders import load_dinov2
    from visual_repr import l2n
    torch.set_num_threads(threads)
    S = pd.read_csv(OUT / "mapillary_selection.csv", dtype={"id": str})
    ids = [i for i in S["id"] if (IMG / f"{i}.jpg").exists()]
    out = OUT / "mapillary_dinov2.npz"
    done = dict(np.load(out)) if out.exists() else {"ids": np.array([], str), "cls": np.zeros((0, 768), np.float32)}
    have = set(done["ids"].tolist())
    todo = [i for i in ids if i not in have]
    log(f"{len(ids):,} imágenes, {len(todo):,} por extraer")
    spec = load_dinov2("cpu")
    got_ids, got = list(done["ids"]), [done["cls"]]
    t0 = time.perf_counter()
    with torch.no_grad():
        for a in range(0, len(todo), batch):
            part = todo[a:a + batch]
            tens = []
            for i in part:
                with Image.open(IMG / f"{i}.jpg") as im:
                    tens.append(spec.transform(im.convert("RGB")))
            f = spec.model.forward_features(torch.stack(tens)).float().numpy()
            got.append(l2n(f[:, 0]).astype(np.float32))
            got_ids += part
            if (a // batch) % 100 == 99 or a + batch >= len(todo):          # checkpoint atómico
                tmp = out.with_suffix(".tmp.npz")
                np.savez(tmp, ids=np.array(got_ids), cls=np.concatenate(got))
                os.replace(tmp, out)
                log(f"  {len(got_ids):,}/{len(ids):,} · {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "download", "embed", "models"], required=True)
    a = ap.parse_args()
    set_seed(SEED)
    if a.stage == "select":
        stage_select()
    elif a.stage == "download":
        stage_download()
    elif a.stage == "embed":
        stage_embed()
    else:
        from exp048_models import main as models_main
        models_main()
