"""
EXP-061 (local) — Fine-tuning parcial de DINOv2 ViT-B/14 en Place Pulse 2.0 (safe), en GPU de 4 GB.
Misma lógica que deliverables/colab/build_notebook.py (últimos K bloques + norm entrenables, cabeza lineal sobre CLS L2,
pérdida Bradley–Terry sobre pares de train, OneCycle, mejor punto en val, test una sola vez), adaptada a poca VRAM:
fp16 + GradScaler, bloques congelados bajo no_grad, micro-lotes con acumulación (lote efectivo = 128 pares), imágenes
leídas del zip local con un handle por worker, checkpoint cada N pasos y reanudación automática.

Salida (formato idéntico al notebook, la lee src/exp061_compare_finetune.py):
  deliverables/colab/results/finetune_dinov2_K{K}__{protocolo}__seed42/{predictions.csv, metrics.json}
Checkpoints: outputs/exp061/finetune_local/<run>/last.pt  ·  log: outputs/exp061/finetune_local.log

Uso (venv con CUDA):
  .venv-gpu/Scripts/python.exe src/exp061_finetune_local.py --protocol image_disjoint --epochs 1
  .venv-gpu/Scripts/python.exe src/exp061_finetune_local.py --protocol image_disjoint --smoke
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MANIFESTS, OUTPUTS, REPO_ROOT, SPLITS, ZIP_PATH, environment_info  # noqa: E402

RESULTS = REPO_ROOT / "deliverables" / "colab" / "results"
WORK = OUTPUTS / "exp061" / "finetune_local"
LOG_FILE = OUTPUTS / "exp061" / "finetune_local.log"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_LOGF = None


def plog(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc), flush=True)
    if _LOGF is not None:
        _LOGF.write(line + "\n"); _LOGF.flush()


# ---------------------------------------------------------------- datos
class ZipImages(Dataset):
    """Fila del manifiesto → uint8 HWC 224×224 (convert RGB + resize BILINEAR, igual que el notebook)."""

    def __init__(self, zip_path: str, paths: list[str], img: int):
        self.zip_path, self.paths, self.img = zip_path, paths, img
        self.zf, self.pid = None, None
        self._tl = threading.local()                        # un handle por hilo (modo --thread-workers)

    def __len__(self):
        return len(self.paths)

    def __getstate__(self):
        d = self.__dict__.copy(); d["zf"] = None; d["pid"] = None; d.pop("_tl", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d); self._tl = threading.local()

    def __getitem__(self, i):
        zf = getattr(self._tl, "zf", None)
        if zf is None or getattr(self._tl, "pid", None) != os.getpid():   # un handle por worker/hilo
            zf = self._tl.zf = zipfile.ZipFile(self.zip_path); self._tl.pid = os.getpid()
        with Image.open(io.BytesIO(zf.read(self.paths[i]))) as im:
            a = np.asarray(im.convert("RGB").resize((self.img, self.img), Image.BILINEAR))
        return torch.from_numpy(a.copy())


class ListBatches:
    """batch_sampler mutable: se reasigna .batches antes de cada iter() (sirve con persistent_workers)."""

    def __init__(self):
        self.batches: list[list[int]] = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class ThreadLoader:
    """Alternativa a DataLoader sin procesos hijos (evita WinError 1455 al cargar torch en cada worker en Windows).
    Decodifica con un pool de hilos (PIL libera el GIL) y precarga `prefetch` lotes; mismo orden que batch_sampler."""

    def __init__(self, ds, bsamp, threads, prefetch=6):
        self.ds, self.bs, self.pool, self.prefetch = ds, bsamp, ThreadPoolExecutor(threads), prefetch

    def _load(self, rows):
        x = torch.stack(list(self.pool.map(self.ds.__getitem__, rows)))
        return x.pin_memory() if torch.cuda.is_available() else x

    def __iter__(self):
        batches = list(self.bs.batches)
        with ThreadPoolExecutor(2) as outer:
            futs = [outer.submit(self._load, b) for b in batches[: self.prefetch]]
            for k in range(len(batches)):
                out = futs[k].result(); futs[k] = None
                if k + self.prefetch < len(batches):
                    futs.append(outer.submit(self._load, batches[k + self.prefetch]))
                yield out


def make_loader(ds, workers, threads=0):
    bs = ListBatches()
    if workers == 0 and threads > 0:
        return ThreadLoader(ds, bs, threads), bs
    kw = dict(num_workers=workers, persistent_workers=True, prefetch_factor=4) if workers > 0 else {}
    return DataLoader(ds, batch_sampler=bs, pin_memory=True, **kw), bs


# ---------------------------------------------------------------- modelo
class Ranker(nn.Module):
    def __init__(self, K: int, img: int):
        super().__init__()
        import timm
        self.K = K
        self.enc = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=img)
        for p in self.enc.parameters():
            p.requires_grad = False
        if K > 0:
            for b in self.enc.blocks[-K:]:
                for p in b.parameters():
                    p.requires_grad = True
            for p in self.enc.norm.parameters():
                p.requires_grad = True
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        e, n = self.enc, len(self.enc.blocks)
        with torch.no_grad():                         # parte congelada: sin grafo → poca VRAM
            x = e.norm_pre(e.patch_drop(e._pos_embed(e.patch_embed(x))))
            for blk in e.blocks[: n - self.K]:
                x = blk(x)
            if self.K == 0:
                x = e.norm(x)
        for blk in e.blocks[n - self.K:]:
            x = blk(x)
        if self.K > 0:
            x = e.norm(x)
        z = nn.functional.normalize(e.forward_head(x), dim=-1)   # CLS L2, como las features congeladas
        return self.head(z).squeeze(-1)

    def trainable_state(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}


def to_input(u8, dev, mean, std):
    x = u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return (x - mean) / std


@torch.no_grad()
def scores_for(model, rows, loader, bsamp, bs, dev, mean, std):
    """Puntaje por fila única (ordenada) del manifiesto."""
    model.eval()
    rows = np.asarray(rows)
    bsamp.batches = [rows[a:a + bs].tolist() for a in range(0, len(rows), bs)]
    out = np.zeros(len(rows), np.float32); a = 0
    for u8 in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            s = model(to_input(u8, dev, mean, std))
        out[a:a + len(s)] = s.float().cpu().numpy(); a += len(s)
    return out


def pair_eval(model, idx, Pl, Pr, Y, loader, bsamp, bs, dev, mean, std):
    L, R = Pl[idx], Pr[idx]
    u = np.unique(np.r_[L, R])
    torch.cuda.empty_cache()                              # suelta bloques del entrenamiento antes de lotes grandes
    t = time.time()
    s = scores_for(model, u, loader, bsamp, bs, dev, mean, std)
    torch.cuda.empty_cache()
    plog(f"  eval {len(idx)} pares · {len(u)} imágenes · {len(u) / (time.time() - t):.0f} img/s")
    logit = s[np.searchsorted(u, L)] - s[np.searchsorted(u, R)]
    return logit, Y[idx]


# ---------------------------------------------------------------- entrenamiento
def save_ckpt(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


def run(args) -> None:
    global _LOGF
    dev = "cuda"
    assert torch.cuda.is_available(), "sin CUDA: usar .venv-gpu"
    # Tope de VRAM: en Windows (WDDM) pasar la VRAM física no da OOM sino que desborda a RAM compartida y todo va ~50x
    # más lento (visto en el smoke con micro 64 + eval-bs 256). Con tope, PyTorch libera caché o da OOM explícito.
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / tot))
    CFG = dict(protocols=[args.protocol], K=args.K, epochs=args.epochs, batch_pairs=128, lr_backbone=2e-5, lr_head=1e-3,
               weight_decay=0.05, evals_per_epoch=args.evals_per_epoch, seed=42, img=224,
               runner="local_exp061_finetune_local.py", micro_pairs=args.micro_pairs, max_train_pairs=args.max_train_pairs,
               max_val_pairs=args.max_val_pairs, smoke=args.smoke, gpu=torch.cuda.get_device_name(0))
    name = f"finetune_dinov2_K{CFG['K']}__{args.protocol}__seed{CFG['seed']}"
    out_root = Path(args.out_root) if args.out_root else (OUTPUTS / "exp061" / "smoke_results" if args.smoke else RESULTS)
    work = Path(args.work_dir) if args.work_dir else (OUTPUTS / "exp061" / ("smoke_work" if args.smoke else "finetune_local"))
    out_dir, wdir = out_root / name, work / name
    log_path = Path(args.log_file) if args.log_file else LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOGF = open(log_path, "a", encoding="utf-8")
    plog(f"===== {name} · smoke={args.smoke} · out={out_dir}")
    if (out_dir / "metrics.json").exists() and not args.smoke:
        plog(f"ya existe {out_dir / 'metrics.json'}: salto"); return
    wdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = wdir / "last.pt"

    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", usecols=["left", "right", "label", "city_left", "city_right"])
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str)
    man = man[man["image_in_zip"] == "True"].reset_index(drop=True)
    row_of = {lid: i for i, lid in enumerate(man["location_id"])}
    Pl = pairs["left"].map(row_of).fillna(-1).astype(int).to_numpy()
    Pr = pairs["right"].map(row_of).fillna(-1).astype(int).to_numpy()
    assert (Pl >= 0).all() and (Pr >= 0).all(), "pares con imagen fuera del zip"
    Y = pairs["label"].to_numpy()
    S = json.load(open(SPLITS / f"{args.protocol}.json"))
    test_key = "test_strict" if args.protocol == "city_holdout" else "test"
    tr, va, te = np.array(S["train"]), np.array(S["val"]), np.array(S[test_key])
    sub = np.random.default_rng(CFG["seed"])
    if args.smoke:
        args.max_train_pairs = args.max_train_pairs or 1024
        args.max_val_pairs = args.max_val_pairs or 400
        if args.max_test_pairs != 0:
            te = np.sort(sub.choice(te, args.max_test_pairs or 400, replace=False))
        CFG.update(max_train_pairs=args.max_train_pairs, max_val_pairs=args.max_val_pairs)
    if args.max_train_pairs and args.max_train_pairs < len(tr):
        tr = np.sort(sub.choice(tr, args.max_train_pairs, replace=False))
    if args.max_val_pairs and args.max_val_pairs < len(va):
        va = np.sort(sub.choice(va, args.max_val_pairs, replace=False))
    plog(f"{len(pairs)} pares · {len(man)} imágenes · train {len(tr)} · val {len(va)} · {test_key} {len(te)}")

    ds = ZipImages(str(ZIP_PATH), man["image_path"].tolist(), CFG["img"])
    tr_loader, tr_bs = make_loader(ds, args.workers, args.thread_workers)
    ev_loader, ev_bs = make_loader(ds, args.eval_workers, args.thread_workers)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    model = Ranker(CFG["K"], CFG["img"]).to(dev)
    groups = [{"params": list(model.head.parameters()), "lr": CFG["lr_head"]}]
    bb = [p for p in model.enc.parameters() if p.requires_grad]
    if bb:
        groups.append({"params": bb, "lr": CFG["lr_backbone"]})
    opt = torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])
    spe = len(tr) // CFG["batch_pairs"]                      # pasos por época (descarta el lote incompleto)
    total = CFG["epochs"] * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    every = max(1, spe // CFG["evals_per_epoch"])
    ckpt_every = args.ckpt_every
    best, best_state, hist, step, prior_min = -1.0, None, [], 0, 0.0

    keys = ("K", "epochs", "batch_pairs", "lr_backbone", "lr_head", "weight_decay", "seed", "max_train_pairs", "evals_per_epoch")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        diff = {k: (ck["cfg"].get(k), CFG[k]) for k in keys if ck["cfg"].get(k) != CFG[k]}
        if diff:
            raise SystemExit(f"checkpoint {ckpt_path} con otra config {diff}: bórralo o usa la misma config")
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        best, best_state, hist, step, prior_min = ck["best"], ck["best_state"], ck["log"], ck["step"], ck["minutes"]
        assert np.array_equal(ck["tr"], tr), "subconjunto de train distinto al del checkpoint"
        plog(f"REANUDO desde paso {step}/{total} (best val {best:.4f}, {prior_min:.1f} min previos)")
    plog(f"pasos/época {spe} · total {total} · eval cada {every} · ckpt cada {ckpt_every} · micro {args.micro_pairs} pares")

    def checkpoint(t_sess):
        save_ckpt(ckpt_path, dict(cfg=CFG, step=step, model=model.trainable_state(), opt=opt.state_dict(),
                                  sched=sched.state_dict(), scaler=scaler.state_dict(), best=best, best_state=best_state,
                                  log=hist, minutes=prior_min + (time.time() - t_sess) / 60, tr=tr))

    t_sess = time.time(); s0 = step; t_win, run_loss, n_win = time.time(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    mp = args.micro_pairs
    while step < total:
        ep = step // spe
        perm = np.random.default_rng(CFG["seed"] * 1000 + ep).permutation(tr)   # determinista → reanudable a mitad de época
        meta, batches = [], []
        for k in range(step % spe, spe):
            b = perm[k * CFG["batch_pairs"]:(k + 1) * CFG["batch_pairs"]]
            for j in range(0, len(b), mp):
                m = b[j:j + mp]
                rows = np.unique(np.r_[Pl[m], Pr[m]])
                batches.append(rows.tolist())
                meta.append((np.searchsorted(rows, Pl[m]), np.searchsorted(rows, Pr[m]), Y[m], j + mp >= len(b)))
        tr_bs.batches = batches
        step_loss = 0.0
        model.train()
        opt.zero_grad(set_to_none=True)
        for u8, (li, ri, y, last) in zip(tr_loader, meta):
            x = to_input(u8, dev, mean, std)
            with torch.autocast("cuda", dtype=torch.float16):
                s = model(x)
                logit = s[torch.from_numpy(li).to(dev)] - s[torch.from_numpy(ri).to(dev)]
                loss = bce(logit.float(), torch.from_numpy(y).to(dev, torch.float32)) / CFG["batch_pairs"]
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
            if not last:
                continue
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            run_loss += step_loss; n_win += 1; last_loss = step_loss; step_loss = 0.0
            if step % args.log_every == 0 or step == total:
                el = time.time() - t_sess
                sps = (step - s0) / el
                pps = n_win * CFG["batch_pairs"] / (time.time() - t_win)
                eta = (total - step) / sps / 60 if sps > 0 else float("nan")
                plog(f"paso {step}/{total} ep {ep} · loss {run_loss / n_win:.4f} · {pps:.1f} pares/s · "
                     f"VRAM pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB · "
                     f"{(prior_min + el / 60):.1f} min · ETA {eta:.1f} min")
                t_win, run_loss, n_win = time.time(), 0.0, 0
            evaled = False
            if step % every == 0:
                t_e = time.time()
                lv, yv = pair_eval(model, va, Pl, Pr, Y, ev_loader, ev_bs, args.eval_bs, dev, mean, std)
                acc = float(((lv > 0) == yv).mean())
                hist.append(dict(step=step, epoch=ep, loss=float(last_loss), val_acc=acc,
                                 min=prior_min + (time.time() - t_sess) / 60))
                if acc > best:
                    best, best_state = acc, model.trainable_state()
                plog(f"VAL {hist[-1]} · mejor {best:.4f} · eval {time.time() - t_e:.0f}s")
                model.train(); evaled = True
            if evaled or step % ckpt_every == 0 or step == total:
                checkpoint(t_sess); plog(f"checkpoint paso {step} → {ckpt_path}")
            if step >= total:
                break

    if best_state is None:                                  # p. ej. smoke sin eval: usa el estado final
        best_state = model.trainable_state()
    model.load_state_dict(best_state, strict=False)
    t_e = time.time()
    lt, yt = pair_eval(model, te, Pl, Pr, Y, ev_loader, ev_bs, args.eval_bs, dev, mean, std)
    corr = ((lt > 0) == yt).astype(float); rng = np.random.default_rng(42)
    boot = [corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000)]
    m = dict(protocol=args.protocol, K=CFG["K"], cfg=CFG, best_val_acc=best, test_split=test_key, n_test=len(te),
             pairwise_accuracy=float(corr.mean()), ci95_low=float(np.percentile(boot, 2.5)),
             ci95_high=float(np.percentile(boot, 97.5)), log=hist)
    P = pairs.iloc[te]
    pred = pd.DataFrame({"split": test_key, "left": P["left"].values, "right": P["right"].values,
                         "city_left": P["city_left"].values, "city_right": P["city_right"].values,
                         "label": yt, "logit": lt, "pred": (lt > 0).astype(int)})
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    json.dump(m, open(out_dir / "metrics.json", "w"), indent=1)
    (wdir / "environment.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
    plog(f"{args.protocol} TEST {m['pairwise_accuracy']:.4f} [{m['ci95_low']:.4f}, {m['ci95_high']:.4f}] "
         f"(test {time.time() - t_e:.0f}s) → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["image_disjoint", "city_holdout", "paper_protocol"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max-train-pairs", type=int, default=None, help="submuestra de train (presupuesto de tiempo)")
    ap.add_argument("--max-val-pairs", type=int, default=None, help="submuestra de val (acelera las evaluaciones)")
    ap.add_argument("--max-test-pairs", type=int, default=None, help="solo smoke: pares de test (def. 400; 0 = test completo)")
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--micro-pairs", type=int, default=32, help="pares por micro-lote (acumulación hasta 128)")
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--vram-cap-gb", type=float, default=3.0, help="tope de VRAM de PyTorch (evita desborde WDDM)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--thread-workers", type=int, default=0,
                    help="con --workers 0: decodifica con N hilos en el proceso principal (sin procesos hijos)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true", help="corrida mínima (1024 pares train, 400 val/test) fuera de results/")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1 if args.epochs == 2 else args.epochs
        args.ckpt_every = min(args.ckpt_every, 4)
        args.log_every = min(args.log_every, 1)
        args.evals_per_epoch = 2 if args.evals_per_epoch == 4 else args.evals_per_epoch
    run(args)


if __name__ == "__main__":
    main()
