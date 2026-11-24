"""
EXP-050 — Publica la encuesta abierta en GitHub Pages (repo público `aloues/calles-lima`).

Copia al repo del sitio solo lo que ve el participante: index.html (con la URL del backend), los 400 estímulos
anónimos sNNN.jpg y una portada para la vista previa de enlaces. El manifiesto estímulo → distrito/spot NO se publica.

Uso:  python publish_site.py [--endpoint https://script.google.com/macros/s/<id>/exec] [--live-key <clave de navegador>] [--no-push]

Con --live-key (recomendado, 06/11): NO se copian imágenes de Street View al sitio; la página pide cada foto a la Street View
Static API con una clave de NAVEGADOR restringida al dominio aloues.github.io (no la del .env). Mapeo sNNN → pano y
direcciones desde outputs/survey/stimuli_live.csv (survey_live_panos.py).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STIMULI = HERE.parent / "outputs" / "survey" / "stimuli"
SITE = HERE.parents[3] / "calles-lima"          # fuera del repo de la tesis: C:/Users/.../perception/calles-lima
REPO = "aloues/calles-lima"
COVER_STIMS = ("s003", "s002")                  # un punto de Barranco y uno de Lince


def build(endpoint: str, live_key: str | None = None) -> None:
    SITE.mkdir(exist_ok=True)
    html = (HERE / "index.html").read_text(encoding="utf-8")
    html, k = re.subn(r'const ENDPOINT = "[^"]*";', f'const ENDPOINT = "{endpoint}";', html)
    assert k == 1
    if live_key:
        import json

        import pandas as pd
        L = pd.read_csv(HERE.parent / "outputs" / "survey" / "stimuli_live.csv", dtype=str)
        assert len(L) == 400 and L["pano_id"].notna().all(), "stimuli_live.csv incompleto"
        live = {"key": live_key, "s": {r.stim_id: [r.pano_id, *map(int, r.headings.split("|"))] for r in L.itertuples()}}
        html, k = re.subn(r"const LIVE = null;", "const LIVE = " + json.dumps(live, separators=(",", ":")) + ";", html)
        assert k == 1
    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    (SITE / "README.md").write_text("Encuesta de tesis (UTEC): ¿qué calles de Lima se ven más seguras? "
                                    "https://aloues.github.io/calles-lima/ · Imágenes: Google Street View.\n",
                                    encoding="utf-8")
    if live_key:                                    # sin imágenes de Street View en el sitio
        shutil.rmtree(SITE / "stimuli", ignore_errors=True)
        from PIL import ImageDraw
        cover = Image.new("RGB", (1200, 630), (27, 42, 58))
        ImageDraw.Draw(cover).text((80, 280), "¿Qué calles de Lima se ven más seguras?  ·  Encuesta de tesis (UTEC)", fill=(255, 255, 255))
        cover.save(SITE / "cover.jpg", quality=85)
        return
    shutil.copytree(STIMULI, SITE / "stimuli", dirs_exist_ok=True)
    cover = Image.new("RGB", (1200, 630), (245, 246, 244))
    for i, s in enumerate(COVER_STIMS):
        with Image.open(STIMULI / f"{s}.jpg") as im:
            cover.paste(im.resize((576, 360), Image.LANCZOS), (16 + i * 592, 135))
    cover.save(SITE / "cover.jpg", quality=85)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=SITE, check=True, capture_output=True, text=True).stdout


def push() -> None:
    if not (SITE / ".git").exists():
        git("init", "-b", "main")
    git("add", "-A")
    if git("status", "--porcelain").strip():
        git("commit", "-m", "Actualiza la encuesta")
    if not git("remote").strip():
        subprocess.run(["gh", "repo", "create", REPO, "--public", "--source", str(SITE), "--push",
                        "--description", "Encuesta de tesis: ¿qué calles de Lima se ven más seguras?"], check=True)
        subprocess.run(["gh", "api", "-X", "POST", f"repos/{REPO}/pages", "-f", "build_type=legacy",
                        "-f", "source[branch]=main", "-f", "source[path]=/"], check=True)
    else:
        git("push", "origin", "main")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None, help="URL /exec del Apps Script; por defecto conserva la publicada")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--live-key", default=None, help="clave de navegador restringida por dominio (modo en vivo)")
    a = ap.parse_args()
    endpoint = a.endpoint
    if endpoint is None:
        cur = (SITE / "index.html").read_text(encoding="utf-8") if (SITE / "index.html").exists() else ""
        m = re.search(r'const ENDPOINT = "([^"]*)";', cur)
        endpoint = m.group(1) if m else ""
    assert endpoint == "" or re.fullmatch(r"https://script\.google\.com/macros/s/[\w-]+/exec", endpoint), endpoint
    build(endpoint, a.live_key)
    if not a.no_push:
        push()
    print("sitio:", SITE, "· backend:", endpoint or "(sin configurar: la encuesta se muestra cerrada)")
