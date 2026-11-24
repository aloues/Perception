"""
Corre el probe de los TRES encoders sobre el subconjunto del smoke test.

OURS e ImageNet no necesitan re-extraccion: sus features completas ya existen y
los splits restringidos del subconjunto solo seleccionan pares cuyas dos
imagenes estan en el subconjunto, asi que el probe solo ve esas imagenes.
DINOv2 usa outputs/features_smoke/.

Los run_id llevan el prefijo `smoke__` para que collect_results.py los etiquete
como PRELIMINAR / SMOKE TEST y no los mezcle con la corrida completa.

Uso:
    python run_smoke_probes.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from utils import FEATURES, OUTPUTS, PKG_ROOT, log

HERE = Path(__file__).parent
SMOKE_SPLITS = PKG_ROOT / "data" / "smoke" / "splits"
FEATURES_SMOKE = OUTPUTS / "features_smoke"
PROTOCOLS = ["image_disjoint", "city_holdout", "paper_protocol"]  # prioridad del brief
ENCODERS = [
    ("ours_simclr_vit_b16", None),        # features completas
    ("imagenet_vit_b16", None),
    ("dinov2_vit_b14", FEATURES_SMOKE),   # features del subconjunto
]


def available(enc: str, root: Path | None) -> bool:
    d = (root or FEATURES) / enc
    idx = d / "index.json"
    if not idx.exists():
        log(f"  {enc}: sin index.json en {d} -> se omite")
        return False
    meta = json.loads(idx.read_text(encoding="utf-8"))
    if root is FEATURES_SMOKE and not meta.get("complete", False):
        log(f"  {enc}: extraccion del subconjunto INCOMPLETA "
            f"({meta['n_images']:,}/{meta['n_subset']:,}) -> se omite")
        return False
    return True


def main() -> None:
    ok, failed = [], []
    for enc, root in ENCODERS:
        if not available(enc, root):
            failed.append((enc, "features no disponibles/incompletas"))
            continue
        for proto in PROTOCOLS:
            run_id = f"smoke__{enc}__{proto}__linear__seed42"
            cmd = [sys.executable, str(HERE / "train_probe.py"),
                   "--encoder", enc, "--protocol", proto,
                   "--splits-root", str(SMOKE_SPLITS),
                   "--run-id", run_id]
            if root is not None:
                cmd += ["--features-root", str(root)]
            log(f"\n=== {run_id} ===")
            p = subprocess.run(cmd, cwd=HERE)
            (ok if p.returncode == 0 else failed).append(
                (run_id, "ok" if p.returncode == 0 else f"exit {p.returncode}"))

    log(f"\nRuns OK: {len(ok)}")
    for r, _ in ok:
        log(f"  {r}")
    if failed:
        log(f"Fallidos/omitidos: {len(failed)}")
        for r, why in failed:
            log(f"  {r}: {why}")


if __name__ == "__main__":
    main()
