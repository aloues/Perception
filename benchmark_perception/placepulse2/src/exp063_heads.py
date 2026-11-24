"""
EXP-063 — ¿Qué cabeza / clasificador convierte mejor el embedding en percepción de seguridad? (asesoría 12/11, R1–R2)

Diseño pre-registrado en docs/EXPERIMENT_REGISTRY.md (2026-11-13, antes de ejecutar).

Idea central: TODA cabeza produce un puntaje por imagen g(z) ∈ R. La decisión sobre un par (i, j) es
sign(g_i − g_j) y, para las cabezas entrenadas por pares, P(i ≻ j) = σ(g_i − g_j). Así todas se evalúan sobre
los MISMOS pares de test y la salida es antisimétrica por construcción (intercambiar i y j invierte la decisión).

Tracks (no se mezclan objetivos):
  A  pares   objetivo y_ij (voto)            : lineal-BT (baseline), MLP-BT                       [pérdida BCE]
  B  imagen  objetivo λ_i (BT con votos train): Ridge, KNN-coseno, SVR-RBF, RF, XGBoost, MLP        [MSE]
  C  binaria cuartil sup. vs inf. de λ_train  : logística, SVM lineal, SVM-RBF, KNN, RF, XGBoost, MLP [log-loss]

Selección de hiperparámetros: SOLO exactitud por pares en val. Test se usa una vez por configuración elegida.
Salida incremental (reanudable): outputs/exp063/<protocolo>/results.jsonl + scores_<head>.npy por cabeza elegida.

Uso:
  python src/exp063_heads.py --protocol image_disjoint --tracks A,B,C --reps z,pca128
  python src/exp063_heads.py --protocol image_disjoint --heads ridge,knn --seeds 1,2
  python src/exp063_heads.py --smoke            # 3,000 pares, todas las cabezas, en minutos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURES, MANIFESTS, OUTPUTS, SPLITS, environment_info, log, set_seed  # noqa: E402

OUT = OUTPUTS / "exp063"
N_JOBS = 8
PRIOR_VAR = 1.0
SVM_SUBSAMPLE = 10000


# ============================================================================ datos
def load_Z(encoder: str) -> tuple[np.ndarray, dict[str, int]]:
    d = FEATURES / encoder
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    shards = sorted(d.glob("shard_*.npy"))
    X = np.concatenate([np.load(s) for s in shards]) if shards else np.load(d / "features.npy")
    ids = index["location_ids"][: X.shape[0]]
    return X.astype(np.float32), {lid: i for i, lid in enumerate(ids)}


def pair_arrays(pairs: pd.DataFrame, idx: list[int], row: dict[str, int]):
    sub = pairs.iloc[idx]
    keep = sub["left"].isin(row) & sub["right"].isin(row)
    sub = sub[keep]
    return (sub["left"].map(row).to_numpy(np.int64), sub["right"].map(row).to_numpy(np.int64),
            sub["label"].to_numpy(np.float32))


def bt_fit(li: np.ndarray, ri: np.ndarray, y: np.ndarray, prior_var: float = PRIOR_VAR):
    """Bradley–Terry MAP con prior N(0, prior_var), L-BFGS. Devuelve (rows únicas, λ). Igual que exp038.bt_global."""
    from scipy.optimize import minimize
    from scipy.special import expit
    rows, inv = np.unique(np.r_[li, ri], return_inverse=True)
    a, b = inv[: len(li)], inv[len(li):]
    n = len(rows)
    y = y.astype(float)

    def f(lam):
        p = expit(lam[a] - lam[b])
        ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) - np.sum(lam ** 2) / (2 * prior_var)
        r = y - p
        g = np.bincount(a, r, n) - np.bincount(b, r, n) - lam / prior_var
        return -ll, -g

    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return rows, res.x


# ============================================================================ métricas
def pair_correct(g: np.ndarray, li: np.ndarray, ri: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 si sign(g_i − g_j) coincide con el voto; empate exacto = 0.5 (desempate al azar en esperanza)."""
    d = g[li] - g[ri]
    c = ((d > 0) == (y > 0.5)).astype(np.float64)
    c[d == 0] = 0.5
    return c


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def pair_logloss(g, li, ri, y):
    p = np.clip(sigmoid(g[li] - g[ri]), 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def pigeonhole_ci(ca: np.ndarray, cb: np.ndarray | None, li, ri, B: int = 1000, seed: int = 42,
                  chunk: int = 50) -> tuple[float, float, float]:
    """IC 95 % de mean(ca) (o de mean(ca − cb)) remuestreando IMÁGENES: pesos Poisson(1) por imagen y peso del par =
    w_i·w_j (bootstrap 'pigeonhole' para datos con efectos cruzados, Owen 2007). Respeta que los pares comparten
    imágenes. Devuelve (estimación, lo, hi)."""
    v = ca - cb if cb is not None else ca
    imgs, inv = np.unique(np.r_[li, ri], return_inverse=True)
    a, b = inv[: len(li)], inv[len(li):]
    rng = np.random.default_rng(seed)
    stats = []
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        W = rng.poisson(1.0, size=(m, len(imgs))).astype(np.float32)
        pw = W[:, a] * W[:, b]
        stats.append((pw @ v) / np.maximum(pw.sum(1), 1e-9))
    st = np.concatenate(stats)
    return float(v.mean()), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def pair_boot_ci(v: np.ndarray, B: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    st = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def mcnemar_p(ca: np.ndarray, cb: np.ndarray) -> float:
    from scipy.stats import binomtest
    n01 = int(((ca == 1) & (cb == 0)).sum())
    n10 = int(((ca == 0) & (cb == 1)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


# ============================================================================ cabezas
def _torch_pairwise(kind: str, Xtr_rows, li, ri, y, vli, vri, vy, X, params, seed):
    """Entrena una cabeza de pares (lineal o MLP) con BCE sobre logit = s(z_i) − s(z_j). Early stopping por val loss,
    como train_probe.py. Devuelve g para todas las filas de X."""
    import torch
    from ranking_head import build_head
    from train_probe import run_epoch
    set_seed(seed)
    torch.set_num_threads(N_JOBS)
    Xt = torch.from_numpy(X)
    zl, zr, yt = Xt[li], Xt[ri], torch.from_numpy(y)
    vl, vr, vyt = Xt[vli], Xt[vri], torch.from_numpy(vy)
    kw = {} if kind == "linear" else {"hidden": params["hidden"], "dropout": params["dropout"]}
    head = build_head(kind, embed_dim=X.shape[1], **kw)
    opt = torch.optim.AdamW(head.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("wd", 1e-4))
    best, state, bad, ep_used = np.inf, None, 0, 0
    for ep in range(params.get("epochs", 50)):
        run_epoch(head, zl, zr, yt, opt, 4096)
        v = run_epoch(head, vl, vr, vyt, None, 4096)
        if v < best - 1e-5:
            best, bad, ep_used = v, 0, ep + 1
            state = {k: t.clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= params.get("patience", 5):
                break
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        g = np.concatenate([head.score_images(Xt[i:i + 16384]).numpy() for i in range(0, len(Xt), 16384)])
    return g.astype(np.float64), {"epochs_best": ep_used, "val_loss": float(best)}


def knn_scores(Xtr: np.ndarray, target: np.ndarray, Xq: np.ndarray, ks: list[int], chunk: int = 2048) -> dict:
    """KNN con similitud coseno (filas L2) → media del objetivo de los k vecinos de train. Devuelve {k: scores}."""
    Xtr_n = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    Xq_n = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
    kmax = max(ks)
    out = {k: np.empty(len(Xq)) for k in ks}
    for s in range(0, len(Xq), chunk):
        S = Xq_n[s:s + chunk] @ Xtr_n.T
        top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        T = target[top]
        for k in ks:
            out[k][s:s + chunk] = T[:, :k].mean(1)
    return out


def _mlp_point(Xtr, t, X, params, seed, task: str, val_eval):
    """MLP puntual (regresión MSE sobre λ o clasificación BCE). Early stopping por exactitud de pares en val."""
    import torch
    import torch.nn as nn
    set_seed(seed)
    torch.set_num_threads(N_JOBS)
    net = nn.Sequential(nn.Linear(X.shape[1], params["hidden"]), nn.ReLU(), nn.Dropout(params["dropout"]),
                        nn.Linear(params["hidden"], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=params["wd"])
    crit = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
    Xt, tt, XA = torch.from_numpy(Xtr), torch.from_numpy(t.astype(np.float32)), torch.from_numpy(X)
    best, state, bad, ep_used = -1, None, 0, 0
    for ep in range(100):
        net.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 512):
            b = perm[s:s + 512]
            loss = crit(net(Xt[b]).squeeze(-1), tt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            g = net(XA).squeeze(-1).numpy().astype(np.float64)
        acc = val_eval(g)
        if acc > best + 1e-5:
            best, bad, ep_used, state = acc, 0, ep + 1, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        g = net(XA).squeeze(-1).numpy().astype(np.float64)
    return g, {"epochs_best": ep_used}


# ============================================================================ experimento
class Context:
    """Todo lo que comparten las cabezas de un protocolo: X, índices de pares, λ_train, filas por partición."""

    def __init__(self, protocol: str, encoder: str, rep: str, smoke: bool = False, seed: int = 42):
        self.protocol, self.encoder, self.rep = protocol, encoder, rep
        pairs = pd.read_csv(MANIFESTS / "pairs_safe.csv", dtype=str, usecols=["left", "right", "label",
                                                                             "city_left", "city_right"])
        pairs["label"] = pairs["label"].astype(int)
        split = json.loads((SPLITS / f"{protocol}.json").read_text())
        self.test_key = "test" if "test" in split else "test_strict"
        Z, row = load_Z(encoder)
        tr_idx, va_idx, te_idx = split["train"], split["val"], split[self.test_key]
        if smoke:
            rng = np.random.default_rng(0)
            tr_idx = list(rng.choice(tr_idx, 3000, replace=False))
            va_idx, te_idx = va_idx[:1000], te_idx[:1000]
        self.li, self.ri, self.y = pair_arrays(pairs, tr_idx, row)
        self.vli, self.vri, self.vy = pair_arrays(pairs, va_idx, row)
        self.tli, self.tri, self.ty = pair_arrays(pairs, te_idx, row)
        self.n_train_pairs = len(self.y)
        # filas usadas (se trabaja en un sub-array compacto para no puntuar 110k imágenes en cada cabeza)
        used = np.unique(np.r_[self.li, self.ri, self.vli, self.vri, self.tli, self.tri])
        remap = -np.ones(Z.shape[0], np.int64)
        remap[used] = np.arange(len(used))
        self.global_rows = used
        for a in ("li", "ri", "vli", "vri", "tli", "tri"):
            setattr(self, a, remap[getattr(self, a)])
        X = Z[used]
        self.train_rows = np.unique(np.r_[self.li, self.ri])
        if rep.startswith("pca"):
            from sklearn.decomposition import PCA
            k = int(rep[3:])
            pca = PCA(n_components=k, random_state=seed).fit(X[self.train_rows])   # ajuste SOLO con train
            X = pca.transform(X).astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            self.pca_var = float(pca.explained_variance_ratio_.sum())
        self.X = X
        # λ_BT con votos de TRAIN solamente (objetivo de Tracks B y C)
        rows, lam = bt_fit(self.li, self.ri, self.y)
        self.lam_rows, self.lam = rows, lam
        q25, q75 = np.percentile(lam, [25, 75])
        self.q25, self.q75 = float(q25), float(q75)
        # λ de test con votos de test (solo para evaluar la tarea binaria; nunca para ajustar)
        trows, tlam = bt_fit(self.tli, self.tri, self.ty)
        self.test_lam_rows, self.test_lam = trows, tlam
        self.ids_used = None

    # ---- evaluación común
    def val_acc(self, g):
        return float(pair_correct(g, self.vli, self.vri, self.vy).mean())

    def test_eval(self, g, prob_scale: bool):
        c = pair_correct(g, self.tli, self.tri, self.ty)
        est, lo, hi = pigeonhole_ci(c, None, self.tli, self.tri, B=500)
        out = {"test_acc": est, "test_ci_img_lo": lo, "test_ci_img_hi": hi, "n_test_pairs": int(len(c))}
        if prob_scale:
            out["test_logloss"] = pair_logloss(g, self.tli, self.tri, self.ty)
        return out, c


def run_head(ctx: Context, head: str, seed: int) -> list[dict]:
    """Ajusta la cabeza con su rejilla, elige por val y devuelve [registro] + guarda scores. Un registro por cabeza."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVR, LinearSVC, SVC
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X, lam, lr = ctx.X, ctx.lam, ctx.lam_rows
    Xtr = X[lr]
    rng = np.random.default_rng(seed)
    cands = []   # (params, g, extra)
    t0 = time.perf_counter()
    track = {"linear_bt": "A", "mlp_bt": "A"}.get(head, "C" if head.endswith("_clf") else "B")

    if head == "linear_bt":
        g, ex = _torch_pairwise("linear", None, ctx.li, ctx.ri, ctx.y, ctx.vli, ctx.vri, ctx.vy, X, {}, seed)
        cands.append(({"lr": 1e-3, "wd": 1e-4}, g, ex))
    elif head == "mlp_bt":
        for h in (256, 512):
            for dr in (0.1, 0.3):
                p = {"hidden": h, "dropout": dr}
                g, ex = _torch_pairwise("mlp", None, ctx.li, ctx.ri, ctx.y, ctx.vli, ctx.vri, ctx.vy, X, p, seed)
                cands.append((p, g, ex))
    elif head == "ridge":
        for a in (0.1, 1, 10, 100, 1000):
            m = Ridge(alpha=a).fit(Xtr, lam)
            cands.append(({"alpha": a}, m.predict(X), {}))
    elif head == "knn":
        sc = knn_scores(Xtr, lam, X, [10, 25, 50, 100, 200])
        # una imagen de train es su propio vecino: para train no importa (no se evalúa en train)
        for k, g in sc.items():
            cands.append(({"k": k, "metric": "cosine"}, g, {}))
    elif head == "svr_rbf":
        sub = rng.choice(len(lr), min(SVM_SUBSAMPLE, len(lr)), replace=False)
        for C in (0.3, 1.0):
            m = SVR(kernel="rbf", C=C, epsilon=0.1, gamma="scale", cache_size=2000).fit(Xtr[sub], lam[sub])
            cands.append(({"C": C, "n_train_images": len(sub)}, m.predict(X), {}))
    elif head == "rf":
        for mf in ("sqrt", 0.1):
            for leaf in (5, 20):
                m = RandomForestRegressor(n_estimators=300, max_features=mf, min_samples_leaf=leaf,
                                          n_jobs=N_JOBS, random_state=seed).fit(Xtr, lam)
                cands.append(({"max_features": mf, "min_samples_leaf": leaf, "n_estimators": 300},
                              m.predict(X), {}))
    elif head == "xgb":
        import xgboost as xgb
        for depth in (4, 6):
            m = xgb.XGBRegressor(n_estimators=1000, max_depth=depth, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.5, min_child_weight=5, tree_method="hist",
                                 n_jobs=N_JOBS, random_state=seed).fit(Xtr, lam)
            for it in range(100, 1001, 100):
                cands.append(({"max_depth": depth, "n_rounds": it, "eta": 0.05},
                              m.predict(X, iteration_range=(0, it)).astype(np.float64), {}))
    elif head == "mlp_reg":
        for wd in (1e-4, 1e-3):
            p = {"hidden": 512, "dropout": 0.2, "wd": wd}
            g, ex = _mlp_point(Xtr, lam, X, p, seed, "reg", ctx.val_acc)
            cands.append((p, g, ex))
    elif head.endswith("_clf"):
        pos, neg = lam >= ctx.q75, lam <= ctx.q25
        sel = np.where(pos | neg)[0]
        Xc, yc = Xtr[sel], pos[sel].astype(int)
        base = head[:-4]
        if base == "logreg":
            for C in (0.1, 1, 10):
                m = LogisticRegression(C=C, max_iter=2000).fit(Xc, yc)
                cands.append(({"C": C}, m.decision_function(X), {"proba": m.predict_proba(X)[:, 1]}))
        elif base == "linsvm":
            for C in (0.01, 0.1, 1):
                m = LinearSVC(C=C, max_iter=5000).fit(Xc, yc)
                cands.append(({"C": C}, m.decision_function(X), {}))
        elif base == "svm_rbf":
            sub = rng.choice(len(sel), min(SVM_SUBSAMPLE, len(sel)), replace=False)
            for C in (1.0, 10.0):
                m = SVC(kernel="rbf", C=C, gamma="scale", cache_size=2000).fit(Xc[sub], yc[sub])
                cands.append(({"C": C, "n_train_images": len(sub)}, m.decision_function(X), {}))
        elif base == "knn":
            sc = knn_scores(Xc, yc.astype(float), X, [10, 25, 50, 100, 200])
            for k, g in sc.items():
                cands.append(({"k": k}, g, {"proba": g}))
        elif base == "rf":
            for leaf in (5, 20):
                m = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=leaf,
                                           n_jobs=N_JOBS, random_state=seed).fit(Xc, yc)
                pr = m.predict_proba(X)[:, 1]
                cands.append(({"min_samples_leaf": leaf}, pr, {"proba": pr}))
        elif base == "xgb":
            import xgboost as xgb
            for depth in (4, 6):
                m = xgb.XGBClassifier(n_estimators=1000, max_depth=depth, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.5, min_child_weight=5, tree_method="hist",
                                      n_jobs=N_JOBS, random_state=seed).fit(Xc, yc)
                for it in range(100, 1001, 100):
                    pr = m.predict_proba(X, iteration_range=(0, it))[:, 1].astype(np.float64)
                    cands.append(({"max_depth": depth, "n_rounds": it}, np.log(pr + 1e-9) - np.log(1 - pr + 1e-9),
                                  {"proba": pr}))
        elif base == "mlp":
            p = {"hidden": 512, "dropout": 0.2, "wd": 1e-4}
            g, ex = _mlp_point(Xc, yc, X, p, seed, "clf", ctx.val_acc)
            cands.append((p, g, {**ex, "proba": sigmoid(g)}))
        else:
            raise ValueError(head)
    else:
        raise ValueError(head)

    fit_s = time.perf_counter() - t0
    vals = [ctx.val_acc(g) for _, g, _ in cands]
    best = int(np.argmax(vals))
    params, g, extra = cands[best]
    res, correct = ctx.test_eval(g, prob_scale=(track == "A"))
    rec = {"protocol": ctx.protocol, "encoder": ctx.encoder, "rep": ctx.rep, "head": head, "track": track,
           "seed": seed, "params": params, "val_acc": vals[best], "n_candidates": len(cands),
           "val_grid": [[json.dumps(p), v] for (p, _, _), v in zip(cands, vals)] if len(cands) <= 40 else None,
           "fit_seconds_total": fit_s, "n_train_pairs": ctx.n_train_pairs, "n_train_images": int(len(lr)),
           **res, **{k: v for k, v in extra.items() if k != "proba"}}
    if track == "C":
        rec.update(binary_metrics(ctx, extra.get("proba"), g))
    return rec, g, correct


def binary_metrics(ctx: Context, proba, score) -> dict:
    """Tarea binaria operativa en imágenes de TEST: clases por umbrales de λ_train aplicados a λ_test (votos de test)."""
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
    rows, tl = ctx.test_lam_rows, ctx.test_lam
    pos, neg = tl >= ctx.q75, tl <= ctx.q25
    m = pos | neg
    yt = pos[m].astype(int)
    s = score[rows][m]
    out = {"bin_n_test_images": int(m.sum()), "bin_pos_rate": float(yt.mean()),
           "bin_auroc": float(roc_auc_score(yt, s))}
    if proba is not None:
        p = proba[rows][m]
        out.update({"bin_bal_acc": float(balanced_accuracy_score(yt, p >= 0.5)),
                    "bin_f1": float(f1_score(yt, p >= 0.5)), "bin_brier": float(brier_score_loss(yt, p))})
    else:
        out.update({"bin_bal_acc": float(balanced_accuracy_score(yt, s >= 0)),
                    "bin_f1": float(f1_score(yt, s >= 0))})
    return out


HEADS = {"A": ["linear_bt", "mlp_bt"],
         "B": ["ridge", "knn", "svr_rbf", "rf", "xgb", "mlp_reg"],
         "C": ["logreg_clf", "linsvm_clf", "svm_rbf_clf", "knn_clf", "rf_clf", "xgb_clf", "mlp_clf"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="image_disjoint")
    ap.add_argument("--encoder", default="dinov2_vit_b14")
    ap.add_argument("--tracks", default="A,B,C")
    ap.add_argument("--heads", default=None, help="lista explícita (ignora --tracks)")
    ap.add_argument("--reps", default="z")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    heads = args.heads.split(",") if args.heads else sum((HEADS[t] for t in args.tracks.split(",")), [])
    od = OUT / ("smoke" if args.smoke else args.protocol)
    od.mkdir(parents=True, exist_ok=True)
    (od / "scores").mkdir(exist_ok=True)
    jl = od / "results.jsonl"
    done = set()
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["encoder"], r["rep"], r["head"], r["seed"]))
    for rep in args.reps.split(","):
        ctx = None
        for seed in map(int, args.seeds.split(",")):
            for head in heads:
                key = (args.encoder, rep, head, seed)
                if key in done:
                    log(f"ya hecho {key}")
                    continue
                if ctx is None:
                    t0 = time.perf_counter()
                    ctx = Context(args.protocol, args.encoder, rep, smoke=args.smoke)
                    meta = {"protocol": args.protocol, "encoder": args.encoder, "rep": rep, "test_key": ctx.test_key,
                            "n_train_pairs": ctx.n_train_pairs, "n_val_pairs": int(len(ctx.vy)),
                            "n_test_pairs": int(len(ctx.ty)), "n_train_images_bt": int(len(ctx.lam_rows)),
                            "lambda_train_q25_q75": [ctx.q25, ctx.q75], "lambda_train_sd": float(ctx.lam.std()),
                            "pca_var": getattr(ctx, "pca_var", None), "environment": environment_info(),
                            "context_seconds": time.perf_counter() - t0}
                    (od / f"context_{args.encoder}_{rep}.json").write_text(json.dumps(meta, indent=2),
                                                                         encoding="utf-8")
                    np.save(od / f"rows_{args.encoder}_{rep}.npy", ctx.global_rows)
                    np.savez(od / f"test_pairs_{args.encoder}_{rep}.npz", li=ctx.tli, ri=ctx.tri, y=ctx.ty)
                    log(f"contexto {args.protocol}/{rep}: {meta['n_train_pairs']:,} pares train, "
                        f"{meta['n_test_pairs']:,} test, X {ctx.X.shape}")
                log(f"→ {args.protocol} {args.encoder} {rep} {head} seed {seed}")
                try:
                    rec, g, c = run_head(ctx, head, seed)
                except Exception as e:   # una cabeza que falla no tumba el resto; queda registrada
                    rec = {"protocol": args.protocol, "encoder": args.encoder, "rep": rep, "head": head,
                           "seed": seed, "error": repr(e)}
                    log(f"   ERROR {e!r}")
                    with open(jl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    continue
                np.save(od / "scores" / f"{args.encoder}__{rep}__{head}__s{seed}.npy", g.astype(np.float32))
                np.save(od / "scores" / f"{args.encoder}__{rep}__{head}__s{seed}__correct.npy", c.astype(np.float32))
                with open(jl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                log(f"   val {rec['val_acc']:.4f}  test {rec['test_acc']:.4f} "
                    f"[{rec['test_ci_img_lo']:.4f}, {rec['test_ci_img_hi']:.4f}]  {rec['params']}  "
                    f"{rec['fit_seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
