"""
EXP-065 — LeJEPA in-domain (factibilidad, P1). SIGReg y pérdida LeJEPA reimplementados a partir del código oficial
https://github.com/galilai-group/lejepa (CC BY-NC 4.0; Balestriero & LeCun, arXiv:2511.08544):
  - lejepa/univariate/epps_pulley.py  EppsPulley(t_max=3, n_points=17), trapecio, ventana exp(-t²/2)
  - lejepa/multivariate/slicing.py    SlicingUnivariateTest: proyecciones aleatorias N(0,1) normalizadas, nuevas por paso
  - MINIMAL.md / Algoritmo 2          L = (1-λ)·L_pred + λ·mean_v SIGReg(proj_v);  L_pred = ||centro(globales) - proj||²
Ver outputs/exp065/LITERATURE_NOTES.md (qué está verificado y qué es inferencia).

Piloto (solo si la GPU está libre; nunca junto a EXP-061/EXP-064):
  ResNet-18 desde cero, 4 vistas 128 px (RandomResizedCrop 0.3–1, flip, ColorJitter suave), proyector 512→2048→2048→64
  con BatchNorm, λ = 0.05, AdamW lr 5e-4 wd 5e-4, warmup + coseno, fp16 + GradScaler. Datos: SOLO imágenes de train de
  `image_disjoint` (submuestra declarada). Sin etiquetas.

Uso:
  python src/exp065_lejepa.py --sanity                         # pruebas del regularizador → outputs/exp065/sanity.json
  python src/exp065_lejepa.py --train --n-images 20000 --epochs 10 --batch 64
  python src/exp065_lejepa.py --extract --ckpt outputs/exp065/pilot/last.pt   # embeddings 512-d de las 110,987
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURES, MANIFESTS, OUTPUTS, SPLITS, ZIP_PATH, environment_info, log, set_seed  # noqa: E402

OUT = OUTPUTS / "exp065"


# =============================================================== SIGReg (fiel a la librería oficial)
class EppsPulley(nn.Module):
    """Estadístico de Epps–Pulley por slice: N · ∫ |ECF(t) − φ_N(0,1)(t)|² w(t) dt, t ∈ [0, t_max] usando simetría."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        w = torch.full((n_points,), 2 * dt)
        w[0] = w[-1] = dt
        phi = torch.exp(-t ** 2 / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (N, K)
        n = x.shape[-2]
        xt = x.float().unsqueeze(-1) * self.t                 # (N, K, T)
        err = (torch.cos(xt).mean(-3) - self.phi) ** 2 + torch.sin(xt).mean(-3) ** 2   # (K, T)
        return (err @ self.weights) * n                       # (K,)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization: media del estadístico EP sobre `num_slices` direcciones aleatorias."""

    def __init__(self, num_slices: int = 256, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley(t_max, n_points)
        self.step = 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:      # z: (N, D)
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(self.step)
            A = torch.randn(z.shape[-1], self.num_slices, generator=g).to(z.device)
            A = A / A.norm(dim=0, keepdim=True)
            self.step += 1
        return self.test(z.float() @ A).mean()


def lejepa_loss(proj: torch.Tensor, sigreg: SIGReg, lam: float = 0.05, n_global: int | None = None):
    """proj: (V, N, K). Todas las vistas predicen el centro de las globales (ResNet: globales = todas)."""
    V = proj.shape[0]
    ng = n_global or V
    centers = proj[:ng].mean(0)
    l_pred = (centers.unsqueeze(0) - proj).pow(2).mean()
    l_sig = torch.stack([sigreg(proj[v]) for v in range(V)]).mean()
    return (1 - lam) * l_pred + lam * l_sig, l_pred.detach(), l_sig.detach()


def effective_rank(z: np.ndarray) -> float:
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


# =============================================================== pruebas del regularizador
def sanity() -> dict:
    torch.manual_seed(0)
    reg = SIGReg(num_slices=256)
    N, D = 512, 64
    cases = {
        "isotropic_gaussian": torch.randn(N, D),
        "anisotropic_gaussian(scale 0.1..3)": torch.randn(N, D) * torch.linspace(0.1, 3, D),
        "low_rank(rank 4)": torch.randn(N, 4) @ torch.randn(4, D) / 2,
        "collapsed(constant + 1e-3 noise)": torch.ones(N, D) + 1e-3 * torch.randn(N, D),
        "uniform[-√3,√3] (var 1, no gaussiana)": (torch.rand(N, D) * 2 - 1) * math.sqrt(3),
    }
    res = {k: float(reg(v)) for k, v in cases.items()}
    # gradiente: optimizar SIGReg desde un embedding colapsado debe reducirlo y elevar el rango efectivo
    z = (torch.ones(N, D) + 0.01 * torch.randn(N, D)).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    r0 = effective_rank(z.detach().numpy())
    l0 = float(reg(z))
    for _ in range(300):
        loss = reg(z)
        opt.zero_grad(); loss.backward(); opt.step()
    out = {"statistic_by_case": res, "grad_test": {"sigreg_start": l0, "sigreg_end": float(reg(z)),
                                                   "eff_rank_start": r0, "eff_rank_end": effective_rank(z.detach().numpy())},
           "lejepa_loss_finite": None}
    proj = torch.randn(4, 64, 16, requires_grad=True)
    L, lp, ls = lejepa_loss(proj, reg, 0.05)
    L.backward()
    out["lejepa_loss_finite"] = bool(torch.isfinite(L).item() and torch.isfinite(proj.grad).all().item())
    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.float16):
            Lc, _, _ = lejepa_loss(torch.randn(4, 64, 16, device="cuda"), reg.cuda(), 0.05)
        out["amp_fp16_finite"] = bool(torch.isfinite(Lc).item())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps(out, indent=2))
    return out


# =============================================================== piloto
class Net(nn.Module):
    def __init__(self, proj_dim: int = 64):
        super().__init__()
        import timm
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.proj = nn.Sequential(nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                  nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, proj_dim))

    def forward(self, x):
        h = self.backbone(x)
        return h, self.proj(h)


def train_images(n: int, seed: int) -> list[str]:
    import pandas as pd
    split = json.loads((SPLITS / "image_disjoint.json").read_text())
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str, usecols=["left", "right"])
    ids = sorted(set(pairs.iloc[split["train"]][["left", "right"]].to_numpy().ravel()))
    rng = np.random.default_rng(seed)
    return list(rng.choice(ids, min(n, len(ids)), replace=False))


class ViewsDataset(torch.utils.data.Dataset):
    def __init__(self, ids, V=4, res=128):
        import pandas as pd
        from torchvision import transforms as T
        man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).set_index("location_id")
        self.paths = [man.loc[i, "image_path"] for i in ids]
        self.V, self.zf = V, None
        self.tf = T.Compose([T.RandomResizedCrop(res, scale=(0.3, 1.0)), T.RandomHorizontalFlip(),
                             T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        if self.zf is None:
            self.zf = zipfile.ZipFile(ZIP_PATH)
        im = Image.open(io.BytesIO(self.zf.read(self.paths[i])))
        im.draft("RGB", (256, 256))
        im = im.convert("RGB")
        return torch.stack([self.tf(im) for _ in range(self.V)])


def train(args) -> None:
    set_seed(args.seed)
    od = OUT / (args.out_name)
    od.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = train_images(args.n_images, args.seed)
    (od / "train_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    ds = ViewsDataset(ids, V=4, res=args.res)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    net = Net().to(dev)
    reg = SIGReg(num_slices=256).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    total = args.epochs * len(dl)
    warm = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.999 + 0.001)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    hist, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        net.train()
        for x in dl:
            B = x.shape[0]
            x = x.to(dev, non_blocking=True).flatten(0, 1)      # (B·V, 3, H, W)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                _, p = net(x)
            p = p.float().view(B, 4, -1).transpose(0, 1)       # (V, B, K)
            L, lp, ls = lejepa_loss(p, reg, args.lam)
            opt.zero_grad(set_to_none=True)
            scaler.scale(L).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if not torch.isfinite(L):
                raise RuntimeError(f"pérdida no finita en paso {step}")
            if step % 20 == 0:
                hist.append({"step": step, "epoch": ep, "loss": float(L), "l_pred": float(lp), "sigreg": float(ls),
                             "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60})
                log(f"ep {ep} paso {step}/{total} L {float(L):.4f} pred {float(lp):.4f} sig {float(ls):.3f} "
                    f"{(time.time() - t0) / 60:.1f} min VRAM {(torch.cuda.max_memory_allocated() / 2**30) if dev == 'cuda' else 0:.2f} GB")
            step += 1
        torch.save({"model": net.state_dict(), "epoch": ep, "step": step, "args": vars(args)}, od / "last.pt")
        (od / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    (od / "train_meta.json").write_text(json.dumps({"args": vars(args), "n_images": len(ids), "steps": step,
                                                    "minutes": (time.time() - t0) / 60,
                                                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0,
                                                    "environment": environment_info()}, indent=2), encoding="utf-8")


@torch.no_grad()
def extract(args) -> None:
    """Embeddings 512-d (backbone, no proyector) de las 110,987 imágenes, 128 px Resize, L2 → features/<name>."""
    import pandas as pd
    from PIL import Image
    from torchvision import transforms as T
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"]
    ids, paths = man["location_id"].tolist(), man["image_path"].tolist()
    set_seed(args.seed)                      # el control "random init" es reproducible con la semilla
    net = Net().cuda().eval()
    name = "lejepa_resnet18_pilot"
    if args.ckpt == "random":
        name = "resnet18_random_init"
    else:
        net.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    tf = T.Compose([T.Resize((args.res, args.res)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    from concurrent.futures import ThreadPoolExecutor

    def dec(b):
        im = Image.open(io.BytesIO(b))
        im.draft("RGB", (2 * args.res, 2 * args.res))     # decodificación JPEG reducida (más rápida; ≥ 2× la salida)
        return tf(im.convert("RGB"))
    with zipfile.ZipFile(ZIP_PATH) as zf, ThreadPoolExecutor(6) as ex:
        for s in range(0, len(paths), 256):
            raws = [zf.read(p) for p in paths[s:s + 256]]
            x = torch.stack(list(ex.map(dec, raws))).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h, _ = net(x)
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
            if s % 25600 == 0:
                log(f"extract {s}/{len(paths)}")
    X = np.concatenate(out).astype(np.float32)
    d = FEATURES / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "features.npy", X)
    (d / "index.json").write_text(json.dumps({"encoder": name, "embed_dim": 512, "image_size": args.res,
                                              "l2_normalized": True, "location_ids": ids}), encoding="utf-8")
    log(f"{name}: {X.shape}, rango efectivo (5k) {effective_rank(X[:5000]):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--ckpt", default=str(OUT / "pilot" / "last.pt"))
    ap.add_argument("--n-images", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="pilot")
    args = ap.parse_args()
    if args.sanity:
        sanity()
    if args.train:
        train(args)
    if args.extract:
        extract(args)


if __name__ == "__main__":
    main()
