"""
EXP-070 — Lima: ¿cuántos votos locales hacen falta y para qué? Simulación con las 10 ciudades retenidas de city_holdout
como "ciudad nueva". Pre-registro: docs/EXPERIMENT_REGISTRY.md, "Programa V3 (14/11/2026)".

  1. Cabeza global lineal-BT (train/val de city_holdout; nunca ve las ciudades retenidas).
  2. Imágenes del dominio retenido → mitades adaptación / evaluación (seed 42); solo pares test_strict con ambas
     imágenes en la misma mitad.
  3. Adaptación con n votos locales: w = w_global + δ, min  BCE + λ‖δ‖², λ por validación cruzada de 5 pliegues DENTRO
     de los n votos (realista: en Lima solo se tendrían esos votos). Aleatorio vs activo.
     Activo = los n pares más inciertos (|p_global − 0.5| mínimo) dentro de una submuestra aleatoria del 50 % del pool
     (la submuestra da variabilidad entre semillas; detalle de implementación no especificado en el pre-registro).
  4. Exactitud en la mitad de evaluación (fija). 5 semillas.
  5. N necesario para medir la exactitud con ±2 pts, corregido por el efecto de diseño de imágenes compartidas.

Uso: python src/exp070_lima_budget.py        Salida: outputs/exp070/
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, environment_info, log  # noqa: E402
from v3_common import (ENCODER, REG_ACC, SEED, fit_linear_bt, load_pairs, load_split, load_Z, pair_idx,  # noqa: E402
                       pigeonhole_ci, sigmoid)

OUT = OUTPUTS / "exp070"
NS = [0, 250, 500, 1000, 2000, 4000]
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def fit_delta(D, y, w0, lam):
    """Logística sobre diferencias D (n×d) con offset w0: min mean BCE(D(w0+δ)) + λ‖δ‖²."""
    n, d = D.shape
    off = D @ w0

    def f(delta):
        z = off + D @ delta
        p = sigmoid(z)
        loss = np.mean(np.logaddexp(0, z) - y * z) + lam * delta @ delta
        grad = D.T @ (p - y) / n + 2 * lam * delta
        return loss, grad
    r = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return w0 + r.x


def adapt(D, y, w0, rng):
    """Elige λ por CV de 5 pliegues dentro de los n votos y re-ajusta con todos."""
    folds = rng.integers(0, 5, len(y))
    best, best_lam = np.inf, LAMBDAS[-1]
    for lam in LAMBDAS:
        ls = []
        for k in range(5):
            tr, va = folds != k, folds == k
            if va.sum() == 0 or tr.sum() == 0:
                continue
            w = fit_delta(D[tr], y[tr], w0, lam)
            z = D[va] @ w
            ls.append(np.mean(np.logaddexp(0, z) - y[va] * z))
        if np.mean(ls) < best:
            best, best_lam = np.mean(ls), lam
    return fit_delta(D, y, w0, best_lam), best_lam


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, row = load_Z(ENCODER)
    Xd = X.astype(np.float64)
    df = load_pairs()
    sp = load_split("city_holdout")
    tr, va = df.loc[sp["train"]], df.loc[sp["val"]]
    li, ri, k = pair_idx(tr["left"], tr["right"], row)
    vli, vri, vk = pair_idx(va["left"], va["right"], row)
    g, info = fit_linear_bt(X, li, ri, tr["label"].to_numpy()[k], vli, vri, va["label"].to_numpy()[vk])
    w0 = np.linalg.lstsq(Xd, g, rcond=None)[0]
    T = df.loc[sp["test_strict"]]
    tli, tri, tk = pair_idx(T["left"], T["right"], row)
    T = T[tk].reset_index(drop=True)
    ty = T["label"].to_numpy().astype(float)
    acc_all = float((((Xd[tli] - Xd[tri]) @ w0 > 0) == (ty == 1)).mean())
    log(f"cabeza global: test_strict acc {acc_all:.4f} (registrada {REG_ACC['city_holdout']}); "
        f"corr(Xw0, g) {np.corrcoef(Xd @ w0, g)[0,1]:.6f}")
    if abs(acc_all - REG_ACC["city_holdout"]) > 0.002:
        raise RuntimeError("no se reproduce la cabeza global")

    # mitades de imágenes del dominio retenido
    imgs = np.unique(np.r_[tli, tri])
    half = dict(zip(imgs, np.random.default_rng(SEED).permutation(len(imgs)) % 2))
    hl, hr = np.vectorize(half.get)(tli), np.vectorize(half.get)(tri)
    A = np.where((hl == 0) & (hr == 0))[0]
    E = np.where((hl == 1) & (hr == 1))[0]
    log(f"dominio retenido: {len(imgs):,} imágenes · pares adaptación {len(A):,} · evaluación {len(E):,} "
        f"(se descartan {len(T)-len(A)-len(E):,} cruzados)")
    DA = Xd[tli[A]] - Xd[tri[A]]
    yA = ty[A]
    DE = Xd[tli[E]] - Xd[tri[E]]
    yE = ty[E]
    pA = sigmoid(DA @ w0)
    base_c = ((DE @ w0 > 0) == (yE == 1)).astype(float)
    acc0, lo0, hi0 = pigeonhole_ci(base_c, None, tli[E], tri[E], B=1000)
    log(f"n=0 (transferencia sin adaptar): {acc0:.4f} [{lo0:.4f}, {hi0:.4f}]")

    rows = []
    for n in NS[1:]:
        if n > len(A):
            continue
        for strat in ("aleatorio", "activo"):
            for s in SEEDS:
                rng = np.random.default_rng(1000 * n + s)
                if strat == "aleatorio":
                    sel = rng.choice(len(A), n, replace=False)
                else:
                    sub = rng.choice(len(A), len(A) // 2, replace=False)
                    sel = sub[np.argsort(np.abs(pA[sub] - 0.5))[:n]]
                w, lam = adapt(DA[sel], yA[sel], w0, rng)
                c = ((DE @ w > 0) == (yE == 1)).astype(float)
                rows.append({"n": n, "strategy": strat, "seed": s, "lambda": lam, "acc_eval": float(c.mean()),
                             "delta_vs_n0": float(c.mean() - acc0)})
                log(f"  n={n:>5} {strat:9s} seed {s}: acc {c.mean():.4f} (Δ {100*(c.mean()-acc0):+.2f} pts, λ={lam})")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "learning_curve_runs.csv", index=False)
    S = R.groupby(["n", "strategy"]).agg(acc_mean=("acc_eval", "mean"), acc_sd=("acc_eval", "std"),
                                         delta_mean=("delta_vs_n0", "mean"), delta_sd=("delta_vs_n0", "std"),
                                         lambda_mode=("lambda", lambda x: x.mode().iloc[0])).reset_index()
    S.to_csv(OUT / "learning_curve_summary.csv", index=False)
    # activo − aleatorio pareado por semilla
    P = R.pivot_table(index=["n", "seed"], columns="strategy", values="acc_eval").reset_index()
    P["act_minus_rand"] = P["activo"] - P["aleatorio"]
    AR = P.groupby("n")["act_minus_rand"].agg(["mean", "std", "min", "max"]).reset_index()
    AR.to_csv(OUT / "active_minus_random.csv", index=False)

    # H-070c: efecto de diseño y N para ±2 pts
    p = acc0
    se_pig = (hi0 - lo0) / 3.92
    se_bin = np.sqrt(p * (1 - p) / len(E))
    deff = float((se_pig / se_bin) ** 2)
    n_needed = {f"halfwidth_{int(h*100)}pts": int(np.ceil(deff * 1.96 ** 2 * p * (1 - p) / h ** 2))
                for h in (0.02, 0.03)}
    n_needed_naive = int(np.ceil(1.96 ** 2 * p * (1 - p) / 0.02 ** 2))
    vpi = float(2 * len(E) / len(np.unique(np.r_[tli[E], tri[E]])))

    res = {"acc_global_test_strict": acc_all, "acc_n0_eval_half": [acc0, lo0, hi0],
           "n_pairs_adapt_pool": int(len(A)), "n_pairs_eval": int(len(E)), "n_images_domain": int(len(imgs)),
           "votes_per_image_eval": vpi, "design_effect_images": deff,
           "n_votes_needed": n_needed, "n_votes_needed_without_deff": n_needed_naive,
           "summary": S.to_dict(orient="records"), "active_minus_random": AR.to_dict(orient="records"),
           "meta": {"encoder": ENCODER, "seeds": SEEDS, "ns": NS, "lambdas": LAMBDAS,
                    "runtime_s": round(time.time() - t0),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                    "caveat": "ciudades retenidas de Place Pulse como sustituto de Lima: mismas fotos de GSV y misma "
                              "población de votantes en línea; no reemplaza la validación local real"}}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"env": environment_info()}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    import matplotlib.pyplot as plt
    from figstyle import NEUTRAL, SERIES, setup
    setup()
    (OUT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for k, (st, s) in enumerate(S.groupby("strategy", sort=False)):
        ax.errorbar(s["n"], s["delta_mean"] * 100, yerr=s["delta_sd"] * 100, marker="o", ms=4, capsize=2,
                    color=SERIES[k], label=f"pares {st}s")
    ax.axhline(0, color=NEUTRAL, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(NS[1:], [f"{n:,}" for n in NS[1:]])
    ax.set_xlabel("Votos locales usados para adaptar la cabeza")
    ax.set_ylabel("Ganancia vs. no adaptar (pts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "lima_budget_es.png", dpi=200)
    log(S.round(4).to_string())
    log(AR.round(4).to_string())
    log(json.dumps({k: res[k] for k in ("acc_n0_eval_half", "design_effect_images", "n_votes_needed",
                                        "n_votes_needed_without_deff", "votes_per_image_eval")}, indent=1))
    log(f"listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
