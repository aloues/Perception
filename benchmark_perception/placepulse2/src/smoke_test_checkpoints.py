"""
Fase C — Smoke test de checkpoints (seccion 20 del prompt maestro).

Para cada checkpoint valida:
  - carga de arquitectura y state_dict (strict)
  - inferencia real sobre un batch sintetico
  - shape del embedding (768 esperado)
  - ausencia de NaN/Inf
  - normalizacion L2 downstream (||z||=1)
  - que NO se este usando el projector por error (se verifica que la salida
    tenga 768 dims y no 128, y que projector exista pero quede sin usar)
  - SHA256 del archivo

Salida: outputs/checkpoint_validation.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from encoders import load_ours, embed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = REPO_ROOT / "external_assets" / "checkpoints"
OUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "checkpoint_validation.json"

CHECKPOINTS = {
    "epoch_50": "simclr_perception_224x_224_best_epoch_50.pt",
    "V2_epoch_49": "V2_simclr_perception_224x_224_best_epoch_49.pt",
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(ckpt_path: Path, device: str = "cpu") -> dict:
    result = {"filename": ckpt_path.name, "path": str(ckpt_path)}

    if not ckpt_path.exists():
        return {**result, "status": "MISSING", "error": "archivo no encontrado"}

    result["size_bytes"] = ckpt_path.stat().st_size
    result["sha256"] = sha256_of(ckpt_path)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    result["checkpoint_keys"] = sorted(raw.keys()) if isinstance(raw, dict) else ["<no-dict>"]
    for meta_key in ("epoch", "loss", "best_loss"):
        if isinstance(raw, dict) and meta_key in raw:
            val = raw[meta_key]
            result[f"ckpt_{meta_key}"] = float(val) if isinstance(val, (int, float)) else str(val)
    del raw

    spec = load_ours(str(ckpt_path), device=device)
    result["architecture"] = "timm vit_base_patch16_224 (num_classes=0) + SimCLR projector"
    result["state_dict_loaded_strict"] = True

    torch.manual_seed(0)
    batch = torch.randn(4, 3, spec.image_size, spec.image_size, device=device)

    raw_feats = spec.model(batch)
    z = embed(spec, batch)

    result["embedding_shape"] = list(z.shape)
    result["embedding_dim"] = int(z.shape[-1])
    result["embedding_dim_is_768"] = int(z.shape[-1]) == 768
    result["raw_backbone_dim"] = int(raw_feats.shape[-1])

    # Verificacion anti-projector: la salida del projector seria 128-dim.
    proj_out_dim = spec.model.projector[-1].out_features
    result["projector_out_dim"] = int(proj_out_dim)
    result["projector_bypassed"] = int(z.shape[-1]) != int(proj_out_dim)

    zn = z.detach().cpu().numpy()
    result["has_nan"] = bool(np.isnan(zn).any())
    result["has_inf"] = bool(np.isinf(zn).any())

    norms = np.linalg.norm(zn, axis=1)
    result["l2_norms"] = [round(float(n), 6) for n in norms]
    result["l2_normalized"] = bool(np.allclose(norms, 1.0, atol=1e-5))

    result["embedding_std"] = round(float(zn.std()), 6)
    result["all_rows_identical"] = bool(np.allclose(zn, zn[0], atol=1e-7))

    checks = [
        result["embedding_dim_is_768"],
        result["projector_bypassed"],
        not result["has_nan"],
        not result["has_inf"],
        result["l2_normalized"],
        not result["all_rows_identical"],
    ]
    result["status"] = "PASS" if all(checks) else "FAIL"
    return result


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "checkpoints": {},
    }

    for label, filename in CHECKPOINTS.items():
        print(f"\n=== Validando {label}: {filename} ===")
        res = validate(CKPT_DIR / filename, device=device)
        report["checkpoints"][label] = res
        print(f"  status={res['status']}")
        for k in ("embedding_dim", "l2_normalized", "projector_bypassed", "has_nan", "sha256"):
            if k in res:
                print(f"  {k}: {res[k]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGuardado: {OUT_PATH}")

    return 0 if all(c.get("status") == "PASS" for c in report["checkpoints"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
