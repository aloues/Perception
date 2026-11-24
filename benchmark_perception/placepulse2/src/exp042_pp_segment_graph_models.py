"""
EXP-042 — Place Pulse (Atlanta, Berlin, Santiago): ¿aporta el grafo de SEGMENTOS a la percepción
humana? G0 agregación ego-k vs G1 GCN vs G2 CRF gaussiano, contra la imagen sola (E0).

Diseño registrado ANTES de ejecutar en docs/EXPERIMENT_REGISTRY.md (EXP-042, 2026-10-23).

POR QUE
-------
El asesor propone (23/10/2026) que la unidad sea la calle y que una GCN —o, si es pesada, un campo
aleatorio condicional (CRF)— propague información entre calles conectadas. El único target de
percepción LEGÍTIMO del proyecto es el voto humano pareado de Place Pulse; las imágenes de Lima no
tienen etiquetas perceptuales (y el crimen no es percepción, M01). Por eso la comparación de modelos de
grafo se hace aquí, con los grafos `drive` de EXP-025 y el MISMO entrenador para todos:

    E0          s_i = w·z_i
    G0-egoK     s_i = w·[z_i, media de imágenes en segmentos a ≤ K saltos SIN la propia]   (K = 0,1,2)
    G0-rand     igual con imágenes aleatorias de la misma ciudad, count-matched a ego-1
    G0-radius   igual con imágenes a ≤ r m (r ∈ {100,200,300} elegido en val)
    G0-bovw     s_i = w·[z_i, histograma BoVW K=100 del ego-1]   (V3 aplicado al barrio)
    G1-gcn      s_i = w·z_i + GCN(X, Â)[seg(i)]   2 capas 769→64→1, Â = D^-½(A+I)D^-½ del dual
    G1-noedge   igual con Â = I  (misma capacidad, SIN conectividad: aísla el efecto del grafo)
    G2-crf      y = argmin Σ_s n_s(y_s − f_s)² + λ Σ_(s,t)(y_s − y_t)² + ε‖y‖², f = w₀·z (E0)
                s_i = w₀·z_i + β·y′_seg(i), y′ = valor del segmento SIN la imagen i (Sherman–Morrison);
                β en train, λ en val. λ = 0 equivale a "media de las otras imágenes del segmento".

Etiquetas: solo pares de train. Features: transductivo (cualquier imagen puede ser contexto; nunca su
etiqueta). Imágenes fuera de las 3 ciudades o a > 30 m de un segmento: contexto = 0.

Uso:
    python exp042_pp_segment_graph_models.py
Salidas: outputs/exp042/
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import binomtest

from exp026_additive_context import load_images
from probe_utils import load_pairs, load_split
from segment_graph import dual_adjacency, normalized_adjacency, reach_matrices, segments_table, snap_to_segments
from utils import OUTPUTS, environment_info, log, set_seed
from visual_repr import BoVW, histograms

SEED = 42
OUT = OUTPUTS / "exp042"
GRAPHS = OUTPUTS / "street_graph_pilot" / "graphs"
CITIES = ["Atlanta", "Berlin", "Santiago"]
SNAP_MAX_M = 30.0
LR, WD, BATCH, EPOCHS, PATIENCE = 1e-3, 1e-4, 4096, 50, 5
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
RADII = [100.0, 200.0, 300.0]
N_BOOT = 2000


# ─── datos ────────────────────────────────────────────────────────────────────

def build_city_graphs(man: pd.DataFrame):
    """Segmentos de las 3 ciudades en un único índice global + snap de cada imagen."""
    import osmnx as ox
    segs, blocks, seg_of_img = [], [], np.full(len(man), -1)
    xy_img = np.full((len(man), 2), np.nan)
    snap_rows, offset = [], 0
    for city in CITIES:
        t = time.perf_counter()
        G = ox.load_graphml(GRAPHS / f"{city}_drive.graphml")
        Gp = ox.project_graph(ox.convert.to_undirected(G))
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        idx = man.index[man["city"] == city].to_numpy()
        import geopandas as gpd
        pts = gpd.GeoSeries(gpd.points_from_xy(man.loc[idx, "lon"], man.loc[idx, "lat"]), crs=4326).to_crs(Gp.graph["crs"])
        xy = np.c_[pts.x, pts.y]
        node_xy = {n: (d["x"], d["y"]) for n, d in Gp.nodes(data=True)}
        r = snap_to_segments(seg, A, xy, node_xy, max_dist=SNAP_MAX_M)
        seg_of_img[idx] = np.where(r.seg_idx >= 0, r.seg_idx + offset, -1)
        xy_img[idx] = xy
        snap_rows.append(dict(city=city, n_images=len(idx), segments=len(seg), dual_edges=int(A.nnz // 2),
                              pct_snapped=100 * (r.seg_idx >= 0).mean(),
                              snap_dist_median_m=float(np.median(r.dist_m[r.seg_idx >= 0])),
                              pct_ambiguous=100 * r.ambiguous[r.seg_idx >= 0].mean(),
                              distinct_segments_with_images=int(len(np.unique(r.seg_idx[r.seg_idx >= 0]))),
                              seconds=round(time.perf_counter() - t)))
        log(f"  {city}: {snap_rows[-1]}")
        segs.append(seg.assign(city=city))
        blocks.append(A)
        offset += len(seg)
    A = sp.block_diag(blocks, format="csr")
    return pd.concat(segs, ignore_index=True), A, seg_of_img, xy_img, pd.DataFrame(snap_rows)


def incidence(seg_of_img: np.ndarray, n_seg: int) -> sp.csr_matrix:
    """P[s, i] = 1 si la imagen i está en el segmento s."""
    i = np.flatnonzero(seg_of_img >= 0)
    return sp.csr_matrix((np.ones(len(i)), (seg_of_img[i], i)), shape=(n_seg, len(seg_of_img)))


def ego_context(Z, seg_of_img, A, K) -> tuple[np.ndarray, np.ndarray]:
    """Media de imágenes en segmentos a ≤ K saltos del segmento de i, SIN la propia i (LOO)."""
    n, d = Z.shape
    ctx, cnt = np.zeros((n, d), np.float32), np.zeros(n)
    img = np.flatnonzero(seg_of_img >= 0)
    src = np.unique(seg_of_img[img])
    pos = {s: r for r, s in enumerate(src)}
    P = incidence(seg_of_img, A.shape[0])
    R = reach_matrices(A, src, K)[K]
    RP = (R @ P).tocsr()                               # [fuente, imagen] = 1 si la imagen está en el ego-K
    SZ = RP @ Z.astype(np.float64)
    C = np.asarray(RP.sum(axis=1)).ravel()
    rows = np.array([pos[s] for s in seg_of_img[img]])
    tot = SZ[rows] - Z[img]
    c = C[rows] - 1
    ok = c > 0
    ctx[img[ok]] = (tot[ok] / c[ok, None]).astype(np.float32)
    cnt[img] = c
    return ctx, cnt


def lists_from_ego(seg_of_img, A, K) -> list[np.ndarray]:
    n = len(seg_of_img)
    img = np.flatnonzero(seg_of_img >= 0)
    src = np.unique(seg_of_img[img])
    pos = {s: r for r, s in enumerate(src)}
    RP = (reach_matrices(A, src, K)[K] @ incidence(seg_of_img, A.shape[0])).tocsr()
    out = [np.array([], int) for _ in range(n)]
    for i in img:
        r = pos[seg_of_img[i]]
        cols = RP.indices[RP.indptr[r]:RP.indptr[r + 1]]
        out[i] = cols[cols != i]
    return out


def mean_of_lists(Z, lists) -> np.ndarray:
    ctx = np.zeros_like(Z)
    for i, L in enumerate(lists):
        if len(L):
            ctx[i] = Z[L].mean(axis=0)
    return ctx


def radius_lists(man, xy_img, seg_of_img, r) -> list[np.ndarray]:
    out = [np.array([], int) for _ in range(len(man))]
    for city in CITIES:
        idx = man.index[(man["city"] == city).to_numpy() & (seg_of_img >= 0)].to_numpy()
        tree = cKDTree(xy_img[idx])
        for j, nb in enumerate(tree.query_ball_point(xy_img[idx], r)):
            out[idx[j]] = idx[[k for k in nb if k != j]]
    return out


def random_lists(man, seg_of_img, counts, rng) -> list[np.ndarray]:
    out = [np.array([], int) for _ in range(len(man))]
    for city in CITIES:
        idx = man.index[(man["city"] == city).to_numpy() & (seg_of_img >= 0)].to_numpy()
        for i in idx:
            c = int(counts[i])
            if c:
                pool = idx[idx != i]
                out[i] = rng.choice(pool, size=min(c, len(pool)), replace=False)
    return out


# ─── modelos y entrenador común ───────────────────────────────────────────────

class LinearScore(nn.Module):
    def __init__(self, X: np.ndarray):
        super().__init__()
        self.X = torch.from_numpy(X.astype(np.float32))
        self.w = nn.Linear(X.shape[1], 1, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.w(self.X[idx]).squeeze(-1)


class GCNScore(nn.Module):
    """s_i = w·z_i + m_i · (Â relu(Â X W1) W2)[seg(i)].  Â es una submatriz EXACTA para 2 capas."""

    def __init__(self, Z, seg_local, Xseg, Ahat, hidden=64, dropout=0.1):
        super().__init__()
        self.Z = torch.from_numpy(Z.astype(np.float32))
        self.seg = torch.from_numpy(seg_local)                     # −1 = sin segmento
        self.X = torch.from_numpy(Xseg.astype(np.float32))
        A = Ahat.tocoo()
        self.A = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), A.shape).coalesce()
        self.w = nn.Linear(Z.shape[1], 1, bias=False)
        self.W1 = nn.Linear(Xseg.shape[1], hidden)
        self.W2 = nn.Linear(hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self._H = None

    def context(self) -> torch.Tensor:
        H = torch.relu(torch.sparse.mm(self.A, self.W1(self.X)))
        return torch.sparse.mm(self.A, self.W2(self.drop(H))).squeeze(-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        c = self.context() if self.training or self._H is None else self._H
        s = self.seg[idx]
        extra = torch.where(s >= 0, c[s.clamp(min=0)], torch.zeros((), dtype=c.dtype))
        return self.w(self.Z[idx]).squeeze(-1) + extra

    def freeze_context(self) -> None:
        with torch.no_grad():
            self.eval()
            self._H = self.context()


def pair_arrays(pairs: pd.DataFrame, idx: list[int], row_of: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sub = pairs.loc[idx]
    sub = sub[sub["left"].isin(row_of) & sub["right"].isin(row_of)]
    return (sub["left"].map(row_of).to_numpy(), sub["right"].map(row_of).to_numpy(),
            sub["label"].to_numpy().astype(np.float32), sub)


def train_model(model: nn.Module, tr, va, seed: int = SEED) -> tuple[nn.Module, dict]:
    """Mismo protocolo que train_probe.py: AdamW, BCE, batch 4096, ≤ 50 épocas, paciencia 5 en val."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    L, R, Y = (torch.from_numpy(a) for a in tr[:3])
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(Y))
        for b in range(0, len(order), BATCH):
            j = order[b:b + BATCH]
            loss = crit(model(L[j]) - model(R[j]), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(crit(torch.from_numpy(predict(model, va)), torch.from_numpy(va[2])))
        hist.append(v)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    if isinstance(model, GCNScore):
        model.freeze_context()
    return model, {"epochs": len(hist), "best_val_logloss": best}


@torch.no_grad()
def predict(model: nn.Module, data) -> np.ndarray:
    model.eval()
    if isinstance(model, GCNScore):
        model._H = model.context()
    L, R = torch.from_numpy(data[0]), torch.from_numpy(data[1])
    return (model(L) - model(R)).numpy()


@torch.no_grad()
def image_scores(model: nn.Module, n: int) -> np.ndarray:
    model.eval()
    return model(torch.arange(n)).numpy()


# ─── G2: CRF gaussiano ────────────────────────────────────────────────────────

def crf_loo_context(f: np.ndarray, seg_of_img: np.ndarray, A: sp.csr_matrix, lam: float,
                    eps: float = 1e-3, batch: int = 128) -> np.ndarray:
    """y′_i = media posterior del segmento de i SIN la imagen i, para (N + λL + εI) y = N f̄.

    Sherman–Morrison: y′_s = (y_s − f_i g_s)/(1 − g_s), g_s = (M⁻¹)_ss. Exacto.
    """
    n_seg = A.shape[0]
    img = np.flatnonzero(seg_of_img >= 0)
    cnt = np.bincount(seg_of_img[img], minlength=n_seg).astype(float)
    b = np.bincount(seg_of_img[img], weights=f[img], minlength=n_seg)
    deg = np.asarray(A.sum(axis=1)).ravel()
    M = (sp.diags(cnt + eps) + lam * (sp.diags(deg) - A)).tocsc()
    lu = spla.splu(M)
    y = lu.solve(b)
    src = np.unique(seg_of_img[img])
    g = np.empty(len(src))
    for k in range(0, len(src), batch):
        cols = src[k:k + batch]
        E = np.zeros((n_seg, len(cols)))
        E[cols, np.arange(len(cols))] = 1.0
        g[k:k + batch] = lu.solve(E)[cols, np.arange(len(cols))]
    gs = dict(zip(src, g))
    out = np.zeros(len(f))
    for i in img:
        s = seg_of_img[i]
        out[i] = (y[s] - f[i] * gs[s]) / (1 - gs[s])
    return out


def fit_beta(base: np.ndarray, ctx: np.ndarray, data) -> float:
    L, R, Y = data[:3]

    def nll(beta):
        z = (base[L] + beta * ctx[L]) - (base[R] + beta * ctx[R])
        return float(np.mean(np.logaddexp(0, z) - Y * z))

    return float(minimize_scalar(nll, bounds=(-20, 20), method="bounded").x)


# ─── evaluación ───────────────────────────────────────────────────────────────

def logloss(z, y):
    return np.logaddexp(0, z) - y * z


def metrics(z, y) -> dict:
    return {"n": int(len(y)), "acc": float(((z > 0) == (y > 0.5)).mean()), "logloss": float(logloss(z, y).mean())}


def paired(z0, z1, y, groups, rng) -> dict:
    c0, c1 = (z0 > 0) == (y > 0.5), (z1 > 0) == (y > 0.5)
    d = c1.astype(float) - c0
    n = len(y)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    ug, inv = np.unique(groups, return_inverse=True)
    gsum = np.bincount(inv, weights=d)
    gcnt = np.bincount(inv)
    gb = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(ug), len(ug))
        gb.append(gsum[k].sum() / gcnt[k].sum())
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(n=n, acc_base=c0.mean(), acc=c1.mean(), delta_acc=d.mean(),
                ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                ci_low_grouped=np.percentile(gb, 2.5), ci_high_grouped=np.percentile(gb, 97.5),
                mcnemar_p=binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0,
                delta_logloss=float((logloss(z1, y) - logloss(z0, y)).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="image_disjoint")
    args = ap.parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    man, Z = load_images("dinov2_vit_b14")
    n = len(man)
    row_of = {lid: i for i, lid in enumerate(man["location_id"])}
    pairs, split = load_pairs(), load_split(args.protocol)
    tr = pair_arrays(pairs, split["train"], row_of)
    va = pair_arrays(pairs, split["val"], row_of)
    te = pair_arrays(pairs, split["test"], row_of)
    pilot = man["city"].isin(CITIES).to_numpy()
    touch = lambda d: pilot[d[0]] | pilot[d[1]]                               # noqa: E731
    log(f"{n:,} imágenes; pares train {len(tr[2]):,} val {len(va[2]):,} test {len(te[2]):,} "
        f"(test que toca piloto: {touch(te).sum():,})")

    seg, A, seg_of_img, xy_img, snapqa = build_city_graphs(man)
    snapqa.to_csv(OUT / "snap_qa.csv", index=False)

    # ── contextos G0 ──
    feats, ctx_info = {"E0": Z}, []
    for K in (0, 1, 2):
        c, cnt = ego_context(Z, seg_of_img, A, K)
        feats[f"G0-ego{K}"] = np.hstack([Z, c])
        m = (seg_of_img >= 0)
        ctx_info.append(dict(context=f"ego{K}", mean_images=float(cnt[m].mean()),
                             pct_pilot_images_with_context=100 * float((cnt[m] > 0).mean())))
        if K == 1:
            cnt1 = cnt
    feats["G0-rand"] = np.hstack([Z, mean_of_lists(Z, random_lists(man, seg_of_img, cnt1, rng))])
    radius_feats = {}
    for r in RADII:
        L = radius_lists(man, xy_img, seg_of_img, r)
        radius_feats[r] = np.hstack([Z, mean_of_lists(Z, L)])
        m = seg_of_img >= 0
        ctx_info.append(dict(context=f"radius{int(r)}", mean_images=float(np.mean([len(L[i]) for i in np.flatnonzero(m)])),
                             pct_pilot_images_with_context=100 * float(np.mean([len(L[i]) > 0 for i in np.flatnonzero(m)]))))
    # V3 sobre el barrio: vocabulario SOLO con imágenes de train
    train_imgs = np.unique(np.r_[tr[0], tr[1]])
    bv = BoVW(100, seed=0, n_init=2).fit(Z[rng.choice(train_imgs, size=min(30000, len(train_imgs)), replace=False)])
    words = bv.assign(Z)
    L1 = lists_from_ego(seg_of_img, A, 1)
    Hb = np.zeros((n, 100), np.float32)
    for i, Li in enumerate(L1):
        if len(Li):
            Hb[i] = np.bincount(words[Li], minlength=100) / len(Li)
    feats["G0-bovw"] = np.hstack([Z, Hb])
    pd.DataFrame(ctx_info).to_csv(OUT / "context_coverage.csv", index=False, float_format="%.3f")

    results, logits, train_info = {}, {}, {}

    def run_linear(name, X):
        tt = time.perf_counter()
        mdl, info = train_model(LinearScore(X), tr, va)
        logits[name] = {"val": predict(mdl, va), "test": predict(mdl, te)}
        train_info[name] = {**info, "params": X.shape[1], "seconds": round(time.perf_counter() - tt)}
        log(f"  {name}: {train_info[name]}  val_acc={metrics(logits[name]['val'], va[2])['acc']:.4f}")
        return mdl

    e0 = run_linear("E0", feats["E0"])
    for name in ("G0-ego0", "G0-ego1", "G0-ego2", "G0-rand", "G0-bovw"):
        run_linear(name, feats[name])
    best_r, best_v = None, -1
    for r in RADII:
        run_linear(f"G0-radius{int(r)}", radius_feats[r])
        v = metrics(logits[f"G0-radius{int(r)}"]["val"][touch(va)], va[2][touch(va)])["acc"]
        if v > best_v:
            best_r, best_v = r, v
    logits["G0-radius"] = logits[f"G0-radius{int(best_r)}"]
    train_info["G0-radius"] = {"selected_radius_m_on_val": best_r}

    # ── G1: GCN sobre el dual, restringida a ≤ 2 saltos de los segmentos con imágenes ──
    img_seg = np.unique(seg_of_img[seg_of_img >= 0])
    keep = np.unique(reach_matrices(A, img_seg, 2)[2].indices)
    loc = -np.ones(A.shape[0], int)
    loc[keep] = np.arange(len(keep))
    seg_local = np.where(seg_of_img >= 0, loc[np.clip(seg_of_img, 0, None)], -1)
    P = incidence(seg_of_img, A.shape[0])[keep]
    cnt_seg = np.asarray(P.sum(axis=1)).ravel()
    Xseg = np.hstack([(P @ Z) / np.maximum(cnt_seg, 1)[:, None], (cnt_seg > 0)[:, None].astype(float)])
    Ahat_full = normalized_adjacency(A)
    Ahat = Ahat_full[keep][:, keep]
    log(f"  GCN: {len(keep):,} segmentos (≤ 2 saltos de {len(img_seg):,} con imágenes)")
    for name, Am in (("G1-gcn", Ahat), ("G1-noedge", sp.identity(len(keep), format="csr"))):
        tt = time.perf_counter()
        mdl, info = train_model(GCNScore(Z, seg_local, Xseg, Am), tr, va)
        logits[name] = {"val": predict(mdl, va), "test": predict(mdl, te)}
        train_info[name] = {**info, "params": int(sum(p.numel() for p in mdl.parameters())),
                            "seconds": round(time.perf_counter() - tt)}
        log(f"  {name}: {train_info[name]}")

    # ── G2: CRF gaussiano sobre la base E0 ──
    base = image_scores(e0, n).astype(np.float64)
    f = base - base[train_imgs].mean()
    best = None
    crf_rows = []
    for lam in LAMBDAS:
        tt = time.perf_counter()
        ctx = crf_loo_context(f, seg_of_img, A, lam)
        beta = fit_beta(base, ctx, tr)
        zv = (base[va[0]] + beta * ctx[va[0]]) - (base[va[1]] + beta * ctx[va[1]])
        mv = metrics(zv[touch(va)], va[2][touch(va)])
        crf_rows.append(dict(lam=lam, beta=beta, val_touch_acc=mv["acc"], val_touch_logloss=mv["logloss"],
                             seconds=round(time.perf_counter() - tt)))
        log(f"  CRF λ={lam}: β={beta:.3f} val_touch={mv}")
        if best is None or mv["logloss"] < best[0]:
            best = (mv["logloss"], lam, beta, ctx)
    pd.DataFrame(crf_rows).to_csv(OUT / "crf_lambda_grid.csv", index=False, float_format="%.5f")
    _, lam, beta, ctx = best
    zt = (base[te[0]] + beta * ctx[te[0]]) - (base[te[1]] + beta * ctx[te[1]])
    logits["G2-crf"] = {"test": zt}
    train_info["G2-crf"] = {"selected_lambda_on_val": lam, "beta_on_train": beta}
    lam0 = [r for r in crf_rows if r["lam"] == 0.0][0]
    ctx0 = crf_loo_context(f, seg_of_img, A, 0.0)
    logits["G2-crf-lam0"] = {"test": (base[te[0]] + lam0["beta"] * ctx0[te[0]]) - (base[te[1]] + lam0["beta"] * ctx0[te[1]])}

    # ── evaluación ──
    T = touch(te)
    grp = np.where(pilot[te[0]], te[0], te[1])[T]
    acc_rows = []
    for name, d in logits.items():
        acc_rows.append(dict(model=name, subset="test_touch_pilot", **metrics(d["test"][T], te[2][T])))
        acc_rows.append(dict(model=name, subset="test_all", **metrics(d["test"], te[2])))
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT / "accuracy.csv", index=False, float_format="%.5f")
    comps = [(m, "E0") for m in logits if m != "E0" and not m.startswith("G0-radius1") and not m.startswith("G0-radius2")
             and not m.startswith("G0-radius3")] + [("G1-gcn", "G1-noedge"), ("G0-ego1", "G0-rand"),
                                                   ("G0-ego1", "G0-ego0"), ("G2-crf", "G2-crf-lam0")]
    prow = []
    for other, b in comps:
        r = paired(logits[b]["test"][T], logits[other]["test"][T], te[2][T], grp, rng)
        prow.append(dict(comparison=f"{other} - {b}", **r))
        log(f"  {other:>12} - {b:<10} Δacc={r['delta_acc']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
            f"agrupado [{r['ci_low_grouped']:+.4f},{r['ci_high_grouped']:+.4f}] p={r['mcnemar_p']:.2g}")
    par = pd.DataFrame(prow)
    par.to_csv(OUT / "paired_vs_baselines.csv", index=False, float_format="%.5f")
    summ = {"protocol": args.protocol, "cities": CITIES, "n_test_touch": int(T.sum()), "train_info": train_info,
            "crf_selected": {"lambda": lam, "beta": beta},
            "criteria": {"any_graph_model_beats_E0": bool((par[par.comparison.str.endswith(" - E0")]["ci_low"] > 0).any()),
                         "connectivity_helps_gcn": bool(par.set_index("comparison").loc["G1-gcn - G1-noedge", "ci_low"] > 0)},
            "seconds": round(time.perf_counter() - t0), "environment": environment_info()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    log(json.dumps(summ["criteria"]))
    log(f"listo en {summ['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
