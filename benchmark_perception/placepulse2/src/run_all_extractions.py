"""
Corre extract_features.py para los 3 encoders, en orden, con reintentos.

Pensado para dejarse corriendo toda la noche sin supervision: si un encoder
falla o se interrumpe, extract_features.py ya reanuda desde el ultimo shard
completo (ver su docstring), asi que basta con re-lanzar este script.

Escribe un log de progreso en outputs/extraction_run.log ademas de stdout,
para poder revisar el avance aunque la terminal que lo lanzo ya no este
visible.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
LOG_PATH = HERE.parent / "outputs" / "extraction_run.log"

ENCODERS = ["ours", "imagenet", "dinov2"]
BATCH_SIZE = 32


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(encoder: str, max_retries: int = 3) -> bool:
    for attempt in range(1, max_retries + 1):
        log(f"=== {encoder}: intento {attempt}/{max_retries} ===")
        proc = subprocess.run(
            [sys.executable, str(HERE / "extract_features.py"),
             "--encoder", encoder, "--batch-size", str(BATCH_SIZE)],
            cwd=HERE,
        )
        if proc.returncode == 0:
            log(f"{encoder}: completado OK")
            return True
        log(f"{encoder}: fallo (codigo {proc.returncode}), reintentando en 30s...")
        time.sleep(30)
    log(f"{encoder}: FALLO tras {max_retries} intentos, se continua con el siguiente")
    return False


def main() -> None:
    log(f"Iniciando extraccion completa: {ENCODERS}")
    t0 = time.perf_counter()
    results = {enc: run_one(enc) for enc in ENCODERS}
    elapsed_h = (time.perf_counter() - t0) / 3600
    log(f"TERMINADO en {elapsed_h:.2f}h. Resultados: {results}")


if __name__ == "__main__":
    main()
