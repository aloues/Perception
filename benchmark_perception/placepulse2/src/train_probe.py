"""
Entrena el probe de ranking sobre embeddings CONGELADOS.

Reglas anti-leakage aplicadas aqui (seccion 28):
  - el scaler, si se usa, se ajusta SOLO con train;
  - los hiperparametros y el early stopping se deciden con val;
  - test se evalua una sola vez, al final;
  - el backbone nunca se actualiza (los embeddings entran precomputados).

Cada run guarda (seccion 33):
    outputs/runs/<run_id>/config.yaml
    outputs/runs/<run_id>/metrics.json
    outputs/runs/<run_id>/predictions.csv
    outputs/runs/<run_id>/training_log.csv
    outputs/runs/<run_id>/environment.txt

Uso:
    python train_probe.py --encoder ours_simclr_vit_b16 --protocol image_disjoint
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import evaluate_all, evaluate_per_city
from ranking_head import build_head
from utils import (FEATURES, MANIFESTS, SPLITS, environment_info, log, new_run_dir,
                   set_seed)


def load_features(encoder: str, features_root: Path | None = None
                  ) -> tuple[np.ndarray, dict[str, int]]:
    d = (features_root or FEATURES) / encoder
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))

    single = d / "features.npy"
    if single.exists():
        # formato del subconjunto del smoke test (extract_features_subset.py):
        # un unico array alineado con index["location_ids"]
        X = np.load(single)
        ids = index["location_ids"][: X.shape[0]]
        return X, {lid: i for i, lid in enumerate(ids)}

    shards = sorted(d.glob("shard_*.npy"))
    if not shards:
        raise FileNotFoundError(
            f"No hay features en {d}. Ejecuta extract_features.py --encoder … primero."
        )
    X = np.concatenate([np.load(s) for s in shards], axis=0)
    ids = index["location_ids"][: X.shape[0]]
    if X.shape[0] != len(index["location_ids"]):
        log(f"  AVISO: {X.shape[0]:,} features para {len(index['location_ids']):,} "
            f"ids declarados (extraccion incompleta)")
    return X, {lid: i for i, lid in enumerate(ids)}


def to_tensors(df: pd.DataFrame, idx: list[int], X: np.ndarray,
               id_to_row: dict[str, int], device: str):
    sub = df.loc[idx]
    keep = sub["left"].isin(id_to_row) & sub["right"].isin(id_to_row)
    sub = sub[keep]
    li = sub["left"].map(id_to_row).to_numpy()
    ri = sub["right"].map(id_to_row).to_numpy()
    y = sub["label"].to_numpy().astype(np.float32)
    return (
        torch.from_numpy(X[li]).to(device),
        torch.from_numpy(X[ri]).to(device),
        torch.from_numpy(y).to(device),
        sub,
    )


def run_epoch(head, zl, zr, y, opt=None, batch_size=4096):
    train = opt is not None
    head.train(train)
    crit = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    order = torch.randperm(len(y), device=y.device) if train else torch.arange(len(y), device=y.device)

    for i in range(0, len(order), batch_size):
        b = order[i:i + batch_size]
        with torch.set_grad_enabled(train):
            logits = head(zl[b], zr[b])
            loss = crit(logits, y[b])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += float(loss.detach()) * len(b)
        n += len(b)
    return total / max(n, 1)


@torch.no_grad()
def predict(head, zl, zr) -> np.ndarray:
    head.eval()
    out = []
    for i in range(0, len(zl), 8192):
        out.append(head(zl[i:i + 8192], zr[i:i + 8192]).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--protocol", required=True,
                    choices=["paper_protocol", "image_disjoint", "city_holdout"])
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--subsample-train", type=int, default=None,
                    help="usa solo N pares de train (muestreo aleatorio con la "
                         "semilla del run). Control para separar el efecto del "
                         "LEAKAGE del efecto del TAMANO de train al comparar "
                         "paper_protocol con image_disjoint.")
    ap.add_argument("--features-root", default=None,
                    help="directorio raiz de features (para el smoke test: "
                         "outputs/features_smoke)")
    ap.add_argument("--splits-root", default=None,
                    help="directorio de splits (para el smoke test: "
                         "data/smoke/splits)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = args.run_id or f"{args.encoder}__{args.protocol}__{args.head}__seed{args.seed}"
    run_dir = new_run_dir(run_id)
    log(f"run_id={run_id} device={device}")

    df = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    splits_root = Path(args.splits_root) if args.splits_root else SPLITS
    splits = json.loads((splits_root / f"{args.protocol}.json").read_text())

    features_root = Path(args.features_root) if args.features_root else None
    X, id_to_row = load_features(args.encoder, features_root)
    log(f"features: {X.shape}")

    train_idx = splits["train"]
    if args.subsample_train and args.subsample_train < len(train_idx):
        rng = np.random.default_rng(args.seed)
        train_idx = [train_idx[i] for i in
                     sorted(rng.choice(len(train_idx), args.subsample_train, replace=False))]
        log(f"train submuestreado a {len(train_idx):,} pares (control de tamano)")
    zl_tr, zr_tr, y_tr, _ = to_tensors(df, train_idx, X, id_to_row, device)
    zl_va, zr_va, y_va, _ = to_tensors(df, splits["val"], X, id_to_row, device)
    log(f"train={len(y_tr):,}  val={len(y_va):,}")

    head = build_head(args.head, embed_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(head, zl_tr, zr_tr, y_tr, opt, args.batch_size)
        va_loss = run_epoch(head, zl_va, zr_va, y_va, None, args.batch_size)
        va_acc = evaluate_all(predict(head, zl_va, zr_va), y_va.cpu().numpy())["pairwise_accuracy"]
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_pairwise_accuracy": va_acc})
        log(f"  ep{ep:>3}  train {tr_loss:.4f}  val {va_loss:.4f}  val_acc {va_acc:.4f}")

        if va_loss < best_val - 1e-5:
            best_val, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  early stopping en epoch {ep}")
                break

    if best_state:
        head.load_state_dict(best_state)

    # ── Test: se toca una sola vez, al final ──────────────────────────────
    test_keys = [k for k in splits if k.startswith("test")]
    metrics = {"run_id": run_id, "encoder": args.encoder, "protocol": args.protocol,
               "head": args.head, "frozen_backbone": True, "best_val_loss": best_val,
               "val": {}, "test": {}}
    metrics["val"] = evaluate_all(predict(head, zl_va, zr_va), y_va.cpu().numpy())

    pred_rows = []
    for key in test_keys:
        zl, zr, y, sub = to_tensors(df, splits[key], X, id_to_row, device)
        logits = predict(head, zl, zr)
        yv = y.cpu().numpy()
        metrics["test"][key] = evaluate_all(logits, yv)

        if args.protocol == "city_holdout":
            metrics["test"][key]["by_city"] = evaluate_per_city(
                logits, yv, sub["city_left"].to_numpy(), sub["city_right"].to_numpy()
            )

        for j in range(len(yv)):
            pred_rows.append({
                "split": key, "left": sub["left"].iloc[j], "right": sub["right"].iloc[j],
                "city_left": sub["city_left"].iloc[j], "city_right": sub["city_right"].iloc[j],
                "label": int(yv[j]), "logit": float(logits[j]),
                "pred": int(logits[j] > 0),
            })
        m = metrics["test"][key]
        log(f"  {key}: acc={m['pairwise_accuracy']:.4f} "
            f"[{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]  n={m['n']:,}")

    config = {**vars(args), "device": device, "embed_dim": int(X.shape[1]),
              "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
              "environment": environment_info()}
    (run_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in config.items() if k != "environment")
        + "\nenvironment: " + json.dumps(config["environment"]),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions.csv", index=False)
    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                          "val_pairwise_accuracy"])
        w.writeheader()
        w.writerows(history)

    log(f"\nGuardado en {run_dir}")


if __name__ == "__main__":
    main()
