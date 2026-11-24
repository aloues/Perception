"""
EXP-064 — Extracción de embeddings CONGELADOS de DINOv3 e I-JEPA para las 110,987 imágenes de Place Pulse 2.0.
Pedido R3 de la asesoría del 12/11/2026 ("pruebas con JEPA y DINOv3").

Encoders (timm, pesos oficiales, sin fine-tuning, eval + no_grad):
    dinov3_b16 -> "vit_base_patch16_dinov3.lvd1689m"   ViT-B/16, 768-d, 1 CLS + 4 registros (licencia DINOv3)
                  salidas: CLS (tras la norma final)  y  media de los 196 tokens de parche (sin CLS ni registros)
    ijepa_h14  -> "vit_huge_patch14_gap_224.in1k_ijepa" ViT-H/14, 1280-d, SIN token CLS (CC-BY-NC 4.0)
                  salida: media de los 256 tokens de parche tras la norma final (= pooling 'avg' de timm)

MISMO PROTOCOLO que DINOv2 en RQ1 (encoders.py): Resize((224, 224)) sin recorte, normalización ImageNet, L2 después del
forward. Diferencias inevitables (se declaran en la tesis): parche 16 (196 tokens) en DINOv3 frente a 14 (256 tokens)
en DINOv2/I-JEPA; I-JEPA no tiene CLS, por eso su único pooling posible es la media de parches; dimensiones 768 vs 1280.

Patrón M03: un SUBPROCESO por bloque de 8,192 imágenes, zip abierto y cerrado dentro del bloque, escritura atómica,
reanudable. Lectura de JPEG con un pool de hilos (sin DataLoader con procesos: EXP-061 falló con WinError 1455).
GPU con autocast fp16 si hay CUDA.

Uso (venv con CUDA):
    python src/exp064_extract_ssl.py --model dinov3_b16 --smoke     # 10 imágenes ×2: determinismo, forma, NaN
    python src/exp064_extract_ssl.py --model dinov3_b16             # lanza / reanuda y consolida
    python src/exp064_extract_ssl.py --model ijepa_h14 --status
Salida: outputs/features/<nombre>/{features.npy, index.json} (no versionado) + outputs/exp064/extract_<modelo>.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURES, MANIFESTS, OUTPUTS, ZIP_PATH, environment_info, log  # noqa: E402

BLOCK = 8192
IMG = 224
MODELS = {
    # clave: (nombre timm, {salida: carpeta de features})
    "dinov3_b16": ("vit_base_patch16_dinov3.lvd1689m",
                   {"cls": "dinov3_vitb16_cls", "meanpatch": "dinov3_vitb16_meanpatch"}),
    "ijepa_h14": ("vit_huge_patch14_gap_224.in1k_ijepa",
                  {"meanpatch": "ijepa_vith14_meanpatch"}),
    # EXP-067 (pre-registrado 13/11 noche): datos (I-JEPA IN-22k), JEPA vigente (V-JEPA 2) y escala (DINOv2 ViT-L)
    "ijepa_h14_in22k": ("vit_huge_patch14_gap_224.in22k_ijepa",
                        {"meanpatch": "ijepa_vith14_in22k_meanpatch"}),
    "dinov2_l14": ("vit_large_patch14_dinov2.lvd142m",
                   {"cls": "dinov2_vitl14_cls", "meanpatch": "dinov2_vitl14_meanpatch"}),
    "vjepa2_l": ("hf:facebook/vjepa2-vitl-fpc64-256",
                 {"meanpatch": "vjepa2_vitl16_meanpatch"}),
}
EXP_DIR = OUTPUTS / "exp064"


def all_ids_paths() -> tuple[list[str], dict[str, str]]:
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    return man["location_id"].tolist(), dict(zip(man["location_id"], man["image_path"]))


IJEPA_HF = {"ijepa_h14": ("facebook/ijepa_vith14_1k", "model.safetensors"),
            "ijepa_h14_in22k": ("facebook/ijepa_vith14_22k", "model.safetensors")}


class VJEPA2Image:
    """V-JEPA 2 (transformers) usado como encoder de IMAGEN: la imagen se replica en 2 cuadros (= 1 tubelet temporal),
    se toma la salida del encoder (sin predictor) tras su norma final; RoPE 3D permite 224 px (16×16 → 196 tokens)."""

    num_prefix_tokens = 0

    def __init__(self, hf_id: str):
        from transformers import VJEPA2Model
        self.m = VJEPA2Model.from_pretrained(hf_id, dtype="float32")

    def to(self, dev):
        self.m = self.m.to(dev)
        return self

    def half(self):
        self.m = self.m.half()
        return self

    def float(self):
        self.m = self.m.float()
        return self

    def eval(self):
        self.m.eval()
        return self

    def forward_features(self, x):
        v = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)          # (B, T=2, C, H, W)
        out = self.m(pixel_values_videos=v, skip_predictor=True)
        return out.last_hidden_state


def ijepa_state_dict_from_hf(model_key: str = "ijepa_h14") -> dict:
    """Pesos HF `facebook/ijepa_vith14_1k` (conversión del checkpoint oficial IN1K-vit.h.14-300e, entrada `encoder`
    = encoder de CONTEXTO, igual que timm; el paper evalúa con el target encoder → limitación declarada) mapeados a las
    claves del ViT de timm. Se usa porque el .pth.tar oficial completo (> 6 GB con predictor) no cabe en disco."""
    import torch
    from huggingface_hub import hf_hub_download

    def read_safetensors_no_mmap(path: str) -> dict:
        """Lector mínimo del formato safetensors (8 bytes de longitud + cabecera JSON + datos) con lecturas por offset,
        sin mmap: safetensors.load_file fallaba con 0xC0000005 en esta máquina bajo presión de memoria."""
        dt = {"F32": np.float32, "F16": np.float16}
        out = {}
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
            base = 8 + n
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                fh.seek(base + a)
                arr = np.frombuffer(fh.read(b - a), dtype=dt[meta["dtype"]]).reshape(meta["shape"])
                out[k] = torch.from_numpy(arr.copy())
        return out
    sd = read_safetensors_no_mmap(hf_hub_download(*IJEPA_HF[model_key]))
    out = {"patch_embed.proj.weight": sd["embeddings.patch_embeddings.projection.weight"],
           "patch_embed.proj.bias": sd["embeddings.patch_embeddings.projection.bias"],
           "pos_embed": sd["embeddings.position_embeddings"],
           "norm.weight": sd["layernorm.weight"], "norm.bias": sd["layernorm.bias"]}
    n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer."))
    for i in range(n):
        p = f"encoder.layer.{i}."
        a = p + "attention.attention."
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([sd[a + f"{t}.weight"] for t in ("query", "key", "value")])
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([sd[a + f"{t}.bias"] for t in ("query", "key", "value")])
        for dst, src in (("attn.proj", "attention.output.dense"), ("norm1", "layernorm_before"),
                         ("norm2", "layernorm_after"), ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
            out[f"blocks.{i}.{dst}.weight"] = sd[p + src + ".weight"]
            out[f"blocks.{i}.{dst}.bias"] = sd[p + src + ".bias"]
    return out


def build_model(model_key: str):
    import timm
    import torch
    timm_name, _ = MODELS[model_key]
    if timm_name.startswith("hf:"):
        model = VJEPA2Image(timm_name[3:])
    elif model_key in IJEPA_HF:
        # Memoria: ViT-H fp32 = 2.5 GB; assign=True evita una segunda copia y se pasa a fp16 antes de la GPU
        # (el primer intento murió con código 139 sin traza, con ~4 GB de RAM libre).
        # load_file (mmap) de safetensors falló de forma intermitente con 0xC0000005 bajo presión de memoria:
        # se convierte UNA vez a un state_dict timm fp16 (torch.save) y los workers cargan esa copia.
        model = timm.create_model(timm_name, pretrained=False, num_classes=0, img_size=IMG)
        # (sin copia en disco: el disco C: quedó lleno; fp16 en memoria)
        model = model.half()
        sd = {k: v.half() for k, v in ijepa_state_dict_from_hf(model_key).items()}
        model.load_state_dict(sd, strict=True)
        del sd
        import gc
        gc.collect()
        if not torch.cuda.is_available():
            model = model.float()          # en CPU: fp32 (los pesos se leyeron en fp16)
    else:
        model = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=IMG)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev), dev


def pool(model, x):
    """Devuelve {salida: tensor (B, d)} a partir de forward_features (tokens tras la norma final)."""
    import torch.nn.functional as F
    tok = model.forward_features(x)                      # (B, prefix + N, d)
    npref = getattr(model, "num_prefix_tokens", 0)
    out = {"meanpatch": tok[:, npref:].mean(dim=1)}
    if npref > 0:
        out["cls"] = tok[:, 0]
    return {k: F.normalize(v.float(), dim=-1) for k, v in out.items()}


def load_batch(zf_path: str, paths: list[str], pool_exec) -> "torch.Tensor":
    import torch
    from PIL import Image
    from encoders import _transform
    tf = _transform(IMG)
    zf = zipfile.ZipFile(zf_path)
    raws = [zf.read(p) for p in paths]
    zf.close()

    def dec(b):
        with Image.open(io.BytesIO(b)) as im:
            return tf(im.convert("RGB"))
    return torch.stack(list(pool_exec.map(dec, raws)))


def worker(model_key: str, block: int, batch_size: int, ids_override: list[str] | None = None,
           out_override: Path | None = None) -> dict:
    import torch
    _, outs = MODELS[model_key]
    ids, path_of = all_ids_paths()
    ids = ids_override if ids_override is not None else ids[block * BLOCK:(block + 1) * BLOCK]
    model, dev = build_model(model_key)
    vecs = {k: [] for k in outs}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as ex, torch.no_grad():
        for i in range(0, len(ids), batch_size):
            x = load_batch(str(ZIP_PATH), [path_of[l] for l in ids[i:i + batch_size]], ex).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                p = pool(model, x)
            for k in outs:
                vecs[k].append(p[k].cpu().numpy().astype(np.float32))
    dt = time.perf_counter() - t0
    res = {}
    for k, folder in outs.items():
        X = np.concatenate(vecs[k])
        assert X.shape[0] == len(ids) and np.isfinite(X).all(), f"{k}: forma/NaN"
        res[k] = X
        if out_override is None:
            stage = FEATURES / folder / "blocks"
            stage.mkdir(parents=True, exist_ok=True)
            tmp = stage / f"block_{block:03d}.tmp.npy"
            np.save(tmp, X)
            os.replace(tmp, stage / f"block_{block:03d}.npy")
            (stage / f"block_{block:03d}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    if out_override is None:
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        log(f"{model_key} bloque {block}: {len(ids)} imgs en {dt / 60:.1f} min ({len(ids) / dt:.1f} img/s, "
            f"VRAM pico {mem:.2f} GB)")
    return {"res": res, "seconds": dt, "device": dev}


def smoke(model_key: str, batch_size: int) -> None:
    """10 imágenes idénticas dos veces: forma, NaN, norma 1, determinismo (R08 / §6.2 del prompt)."""
    ids, _ = all_ids_paths()
    sub = ids[:10]
    a = worker(model_key, 0, batch_size, ids_override=sub, out_override=Path("."))
    b = worker(model_key, 0, batch_size, ids_override=sub, out_override=Path("."))
    rep = {"model": model_key, "timm": MODELS[model_key][0], "n": len(sub), "device": a["device"]}
    for k in a["res"]:
        Xa, Xb = a["res"][k], b["res"][k]
        rep[k] = {"shape": list(Xa.shape), "finite": bool(np.isfinite(Xa).all()),
                  "norm_min": float(np.linalg.norm(Xa, axis=1).min()),
                  "max_abs_diff_rerun": float(np.abs(Xa - Xb).max()),
                  "mean_offdiag_cos": float((Xa @ Xa.T)[~np.eye(len(Xa), dtype=bool)].mean())}
    rep["sec_per_img_first_run"] = a["seconds"] / len(sub)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / f"smoke_{model_key}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(json.dumps(rep, indent=2))


def consolidate(model_key: str, ids: list[str], n_blocks: int) -> None:
    timm_name, outs = MODELS[model_key]
    for k, folder in outs.items():
        out = FEATURES / folder
        stage = out / "blocks"
        X = np.concatenate([np.load(stage / f"block_{b:03d}.npy") for b in range(n_blocks)])
        got = sum((json.loads((stage / f"block_{b:03d}.ids.json").read_text()) for b in range(n_blocks)), [])
        assert got == ids and X.shape[0] == len(ids) and len(set(got)) == len(got)
        assert np.isfinite(X).all()
        tmp = out / "features.tmp.npy"
        np.save(tmp, X)
        os.replace(tmp, out / "features.npy")
        sha = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        (out / "index.json").write_text(json.dumps({
            "encoder": f"timm {timm_name}", "pooling": k, "embed_dim": int(X.shape[1]), "image_size": IMG,
            "preprocessing": "Resize((224,224)) sin recorte + Normalize(ImageNet) + L2", "n_images": len(ids),
            "l2_normalized": True, "sha256_16": sha, "location_ids": ids}), encoding="utf-8")
        log(f"consolidado {folder}: {X.shape} sha {sha}")
        # disco casi lleno: tras verificar ids, forma, NaN y escribir el consolidado, se borran los bloques intermedios
        for b in range(n_blocks):
            for suf in (".npy", ".ids.json"):
                (stage / f"block_{b:03d}{suf}").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=None, help="detenerse tras N bloques nuevos")
    args = ap.parse_args()
    if args.smoke:
        smoke(args.model, args.batch_size)
        return
    if args.worker is not None:
        worker(args.model, args.worker, args.batch_size)
        return
    _, outs = MODELS[args.model]
    first = FEATURES / next(iter(outs.values())) / "blocks"
    first.mkdir(parents=True, exist_ok=True)
    ids, _ = all_ids_paths()
    n_blocks = (len(ids) + BLOCK - 1) // BLOCK
    done = [b for b in range(n_blocks) if all((FEATURES / f / "blocks" / f"block_{b:03d}.npy").exists()
                                             for f in outs.values())]
    log(f"{args.model}: {len(ids):,} imágenes, {n_blocks} bloques, {len(done)} hechos")
    if args.status:
        return
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "timm": MODELS[args.model][0], "environment": environment_info(),
            "batch_size": args.batch_size, "blocks": []}
    new = 0
    for b in range(n_blocks):
        if b in done:
            continue
        t0 = time.time()
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                        "--worker", str(b), "--batch-size", str(args.batch_size)], check=True)
        meta["blocks"].append({"block": b, "seconds": time.time() - t0})
        (EXP_DIR / f"extract_{args.model}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        new += 1
        if args.max_blocks and new >= args.max_blocks:
            log("max-blocks alcanzado")
            return
    consolidate(args.model, ids, n_blocks)
    meta["consolidated"] = True
    (EXP_DIR / f"extract_{args.model}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
