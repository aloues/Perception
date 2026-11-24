"""Utilidades compartidas: paths, semillas, entorno y registro de runs."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

PP_DIR = REPO_ROOT / "external_assets" / "placepulse2"
EXTRACTED = PP_DIR / "extracted"
ZIP_PATH = PP_DIR / "raw" / "place-pulse-2.0.zip"
CKPT_DIR = REPO_ROOT / "external_assets" / "checkpoints"

MANIFESTS = PKG_ROOT / "data" / "manifests"
SPLITS = PKG_ROOT / "data" / "splits"
OUTPUTS = PKG_ROOT / "outputs"
FEATURES = OUTPUTS / "features"
RUNS = OUTPUTS / "runs"

SAFE_STUDY_QUESTIONS = {"safer", "safe"}
VALID_CHOICES = ("left", "right", "equal")
TIE_LABEL = "equal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch"] = None
    return info


def new_run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "environment.txt").write_text(
        json.dumps(environment_info(), indent=2), encoding="utf-8"
    )
    return d


def resolve_safe_study_id() -> str:
    """Identifica el study_id de safe desde studies.tsv, sin hardcodear."""
    import pandas as pd

    studies = pd.read_csv(EXTRACTED / "studies.tsv", sep="\t", dtype=str)
    m = studies[studies["study_question"].str.strip().str.lower().isin(SAFE_STUDY_QUESTIONS)]
    if len(m) != 1:
        raise RuntimeError(f"Se esperaba 1 estudio safe, hay {len(m)}")
    return m.iloc[0]["_id"]


def log(msg: str) -> None:
    # La consola de Windows (cp1252) no codifica λ, ρ, Δ…: sin esto un log mata un proceso de horas (EXP-048).
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(str(msg).encode(enc, errors="replace").decode(enc), flush=True)
    sys.stdout.flush()
