"""
EXP-066 — Techo de ruido del voto con un modelo latente de Bradley–Terry (pre-registrado 2026-11-13 noche,
docs/EXPERIMENT_REGISTRY.md, ANTES de ejecutar).

Pregunta: ¿qué exactitud por pares tendría un predictor que conociera la percepción latente EXACTA de cada foto, si cada
par recibe un solo voto ruidoso? ¿Qué fracción de ese techo alcanza DINOv2 + lineal-BT (0.6785 fotos nuevas)?

Modelo generativo (el 97.8 % de los pares de Place Pulse 2.0 comparan fotos de ciudades DISTINTAS, así que la ciudad entra
explícitamente):
    λ_i = μ_c(i) + ε_i,   ε_i ~ N(0, τ²)   (μ_c fijos, estimados con todos los votos; 56 ciudades, ~15k votos/ciudad)
    y_ij ~ Bernoulli(F(λ_i − λ_j)),   F = logística (BT) o Φ (Thurstone V, robustez)

Estimación de τ (Bayes empírico por momentos simulados): S = exactitud de un BT-MAP (efectos de ciudad + ε con prior
N(0, 1), L-BFGS) ajustado con el 90 % de los votos al predecir el 10 % retenido. S_obs con los votos reales; S(τ) con
votos sintéticos sobre EXACTAMENTE el mismo grafo de pares (mismas imágenes, mismos pares, mismas particiones). τ̂ invierte
S(τ̂) = S_obs. El estimador del estadístico puede ser cualquiera (no tiene que ser óptimo): solo debe aplicarse igual a
datos reales y simulados.

Techo: C = media sobre los pares de test reales de E[F(|μ_cL − μ_cR + Δε|)], Δε ~ N(0, 2τ²) (Gauss–Hermite).

Chequeos (no usados para ajustar): (a) el oráculo BT en muestra (bt_global de EXP-055, todos los votos) simulado a τ̂
debe reproducir el 0.776–0.778 observado; (b) recuperación de τ conocido; (c) emparejamiento adaptativo: |Δλ̂| de pares
reales vs pares permutados, comparado con lo mismo en datos simulados (donde el emparejamiento es independiente de λ).

Uso:   python src/exp066_noise_ceiling.py            Salidas: outputs/exp066/{summary.json, grid.csv, report.md}
       python src/exp066_noise_ceiling.py --quick    (rejilla corta, para probar)
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")       # no competir con la extracción GPU (cuello de botella en CPU)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from utils import MANIFESTS, OUTPUTS, SPLITS, log

SEED = 66
OUT = OUTPUTS / "exp066"
HOLDOUT = 0.10
N_SPLITS_OBS = 5
N_SPLITS_SIM = 2
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(48)      # nodos para E[g(Z)], Z ~ N(0, 1)
GH_W = GH_W / GH_W.sum()


# ---------------------------------------------------------------------------------------------------------- datos
def load() -> dict:
    P = pd.read_csv(MANIFESTS / "pairs_safe.csv", usecols=["left", "right", "label", "city_left", "city_right"],
                    dtype={"left": str, "right": str})
    ids = pd.Index(pd.unique(pd.concat([P["left"], P["right"]])))
    city_of = pd.concat([pd.Series(P["city_left"].to_numpy(), index=P["left"]),
                         pd.Series(P["city_right"].to_numpy(), index=P["right"])])
    city_of = city_of[~city_of.index.duplicated()].reindex(ids)
    cities = pd.Index(sorted(city_of.unique()))
    return {"P": P, "ids": ids, "L": ids.get_indexer(P["left"]), "R": ids.get_indexer(P["right"]),
            "y": P["label"].to_numpy().astype(float), "city": cities.get_indexer(city_of.to_numpy()),
            "cities": cities}


# --------------------------------------------------------------------------------------------- BT con ciudad
def fit_bt_city(L, R, y, city, n, n_city, prior_var=1.0, city_prior_var=100.0, maxiter=400) -> np.ndarray:
    """MAP de λ_i = μ_c + ε_i. Devuelve λ (n,). μ con prior débil (identificabilidad), ε ~ N(0, prior_var)."""
    cL, cR = city[L], city[R]

    def f(th):
        mu, eps = th[:n_city], th[n_city:]
        d = mu[cL] - mu[cR] + eps[L] - eps[R]
        # log-verosimilitud estable: y log σ(d) + (1−y) log σ(−d) = −log(1 + e^{−s d}), s = 2y − 1
        s = 2 * y - 1
        ll = -np.logaddexp(0, -s * d).sum()
        r = y - expit(d)
        g_eps = np.bincount(L, r, n) - np.bincount(R, r, n) - eps / prior_var
        g_mu = np.bincount(cL, r, n_city) - np.bincount(cR, r, n_city) - mu / city_prior_var
        obj = ll - (eps ** 2).sum() / (2 * prior_var) - (mu ** 2).sum() / (2 * city_prior_var)
        return -obj, -np.r_[g_mu, g_eps]

    res = minimize(f, np.zeros(n_city + n), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x[:n_city][city] + res.x[n_city:]


def bt_global_plain(L, R, y, n, prior_var=1.0) -> np.ndarray:
    """Idéntico a exp038_bt_target.bt_global (el oráculo 0.776 de EXP-055), sobre índices."""
    def f(lam):
        d = lam[L] - lam[R]
        p = expit(d)
        ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) - np.sum(lam ** 2) / (2 * prior_var)
        r = y - p
        return -ll, -(np.bincount(L, r, n) - np.bincount(R, r, n) - lam / prior_var)
    return minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def acc(d: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.where(d == 0, 0.5, (d > 0) == (y == 1))))


def heldout_stat(D: dict, y: np.ndarray, splits: list[np.ndarray]) -> float:
    """Media sobre particiones de la exactitud del BT-MAP (90 %) en el 10 % retenido."""
    n, nc = len(D["ids"]), len(D["cities"])
    out = []
    for te in splits:
        tr = np.ones(len(y), bool)
        tr[te] = False
        lam = fit_bt_city(D["L"][tr], D["R"][tr], y[tr], D["city"], n, nc)
        out.append(acc(lam[D["L"][te]] - lam[D["R"][te]], y[te]))
    return float(np.mean(out))


# ------------------------------------------------------------------------------------------------- simulación
def draw_eps(tau: float, size, rng, dist: str = "gauss") -> np.ndarray:
    """ε con varianza τ²: normal o Laplace (colas pesadas; escala b = τ/√2)."""
    return rng.normal(0, tau, size) if dist == "gauss" else rng.laplace(0, tau / np.sqrt(2), size)


def simulate_votes(mu_img: np.ndarray, tau: float, D: dict, rng, link: str = "logit",
                   dist: str = "gauss") -> tuple[np.ndarray, np.ndarray]:
    lam = mu_img + draw_eps(tau, len(mu_img), rng, dist)
    d = lam[D["L"]] - lam[D["R"]]
    p = expit(d) if link == "logit" else norm.cdf(d)
    return (rng.random(len(p)) < p).astype(float), lam


def ceiling(m: np.ndarray, tau: float, link: str = "logit", dist: str = "gauss") -> float:
    """Media sobre pares de E[F(|m + Δε|)], Δε = ε_i − ε_j (Gauss–Hermite si normal; Monte Carlo si Laplace)."""
    if dist != "gauss":
        r = np.random.default_rng(0)
        x = np.abs(m[:, None] + draw_eps(tau, (len(m), 64), r, dist) - draw_eps(tau, (len(m), 64), r, dist))
        return float((expit(x) if link == "logit" else norm.cdf(x)).mean())
    x = np.abs(m[:, None] + np.sqrt(2) * tau * GH_X[None, :])
    F = expit(x) if link == "logit" else norm.cdf(x)
    return float((F * GH_W[None, :]).sum(axis=1).mean())


def curve(grid: pd.DataFrame) -> np.ndarray:
    """S(τ) suave: cúbica por mínimos cuadrados sobre TODAS las réplicas de la rejilla (el ruido de simulación de
    cada punto, ±0.002, hacía que la interpolación lineal de medias sesgara la inversión)."""
    return np.polyfit(grid["tau"].to_numpy(), grid["S"].to_numpy(), 3)


def invert(grid: pd.DataFrame, s_obs: float) -> float:
    """τ con S(τ) = s_obs sobre la curva suave; NaN si s_obs cae fuera del rango de la rejilla (no se extrapola)."""
    c = curve(grid)
    t = np.linspace(grid["tau"].min(), grid["tau"].max(), 2001)
    v = np.polyval(c, t)
    assert np.all(np.diff(v) > 0), "S(τ) ajustada no es monótona en la rejilla"
    if not (v[0] <= s_obs <= v[-1]):
        return float("nan")
    return float(np.interp(s_obs, v, t))


def match_predictor(lam_sim: np.ndarray, y_sim: np.ndarray, Lt, Rt, target: float, rng) -> float:
    """ρ = corr(s, λ) tal que un predictor s = ρ z(λ) + √(1−ρ²) ruido reproduce la exactitud observada."""
    z = (lam_sim - lam_sim.mean()) / lam_sim.std()
    noise = rng.normal(size=len(z))
    def a(rho):
        s = rho * z + np.sqrt(max(1 - rho ** 2, 0)) * noise
        return acc(s[Lt] - s[Rt], y_sim)
    lo, hi = 0.05, 0.999
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if a(mid) < target else (lo, mid)
    return (lo + hi) / 2


# ------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    D = load()
    n, nc, nv = len(D["ids"]), len(D["cities"]), len(D["y"])
    same_city = float(np.mean(D["city"][D["L"]] == D["city"][D["R"]]))
    log(f"{n:,} imágenes, {nc} ciudades, {nv:,} votos (sin empates); pares de la misma ciudad {same_city:.3%}")

    # particiones fijas de votos (las mismas para datos reales y simulados)
    splits = [rng.choice(nv, int(HOLDOUT * nv), replace=False) for _ in range(N_SPLITS_OBS)]

    # μ_c y λ̂ con todos los votos (efectos de ciudad; prior N(0,1) en ε)
    lam_full = fit_bt_city(D["L"], D["R"], D["y"], D["city"], n, nc)
    mu_c = pd.Series(lam_full).groupby(D["city"]).mean()
    # μ por imagen = media de λ̂ de su ciudad (ε tiene media ≈ 0 por ciudad con el prior)
    mu_img = mu_c.reindex(D["city"]).to_numpy()
    log(f"sd entre ciudades μ_c = {mu_c.std():.3f}; sd(λ̂) = {lam_full.std():.3f}")

    s_obs_each = []
    for te in splits:
        s_obs_each.append(heldout_stat(D, D["y"], [te]))
    s_obs = float(np.mean(s_obs_each))
    log(f"S_obs (exactitud retenida del BT-MAP, {N_SPLITS_OBS} particiones) = {s_obs:.4f} ± {np.std(s_obs_each):.4f}")

    taus = [0.6, 0.9, 1.2, 1.5] if args.quick else [0.25, 0.35, 0.45, 0.55, 0.62, 0.68, 0.74, 0.80, 0.86, 1.0, 1.2]
    # μ_c se estimó en escala logit: para el enlace probit se reescala (σ(1.702 x) ≈ Φ(x)), y la rejilla también
    PROBIT = 1.702
    mu_of = {"logit": mu_img, "probit": mu_img / PROBIT}
    taus_of = {"logit": taus, "probit": [t / PROBIT for t in taus]}
    reps = 1 if args.quick else 2
    variants = [("logit", "gauss")] if args.quick else [("logit", "gauss"), ("probit", "gauss"), ("logit", "laplace")]
    rows = []
    for link, dist in variants:
        for tau in taus_of[link]:
            for r in range(reps):
                ys, _ = simulate_votes(mu_of[link], tau, D, rng, link, dist)
                S = heldout_stat(D, ys, splits[:N_SPLITS_SIM])
                rows.append({"variant": f"{link}_{dist}", "link": link, "dist": dist, "tau": tau, "rep": r, "S": S})
                log(f"[{link}/{dist}] τ={tau:.2f} rep {r}: S={S:.4f}  ({(time.perf_counter() - t0) / 60:.1f} min)")
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)

    proto = {}
    for protocol, key in (("image_disjoint", "test"), ("city_holdout", "test_strict"), ("paper_protocol", "test")):
        te = np.array(json.loads((SPLITS / f"{protocol}.json").read_text())[key])
        proto[protocol] = te
    obs_acc = {"image_disjoint": 0.6785, "city_holdout": 0.6715, "paper_protocol": 0.6744}   # RQ1, metrics.json
    sota = pd.read_csv(OUTPUTS / "exp055" / "sota_protocol.csv")
    oracle_obs = {r.protocol: r.pairwise_accuracy for r in sota.itertuples() if r.model.startswith("Oráculo")}
    runs = OUTPUTS / "runs"
    for p, key in (("image_disjoint", "test"), ("city_holdout", "test_strict"), ("paper_protocol", "test")):
        mj = json.loads((runs / f"dinov2_vit_b14__{p}__linear__seed42" / "metrics.json").read_text())
        obs_acc[p] = float(mj["test"][key]["pairwise_accuracy"])

    summary = {"n_images": n, "n_cities": nc, "n_votes": nv, "same_city_pair_share": same_city,
               "sd_city_effects": float(mu_c.std()), "S_obs": s_obs, "S_obs_splits": s_obs_each,
               "dinov2_observed_accuracy": obs_acc, "oracle_in_sample_observed": oracle_obs, "links": {}}
    for link, dist in variants:
        vname = f"{link}_{dist}"
        main_v = vname == "logit_gauss"
        g = grid[grid["variant"] == vname]
        tau_hat = invert(g, s_obs)
        # incertidumbre: réplicas a τ̂ → S_sim → τ invertido (bootstrap paramétrico) + ruido entre particiones
        nb = 3 if args.quick else (12 if main_v else 6)
        tau_b, oracle_sim, pair_ratio_sim, rho_b = [], {p: [] for p in proto}, [], {p: [] for p in proto}
        for b in range(nb):
            ys, lam_s = simulate_votes(mu_of[link], tau_hat, D, rng, link, dist)
            Sb = heldout_stat(D, ys, [splits[b % N_SPLITS_OBS]])
            tau_b.append(invert(g, Sb))
            if b < (2 if args.quick else (4 if main_v else 2)):
                lam_o = bt_global_plain(D["L"], D["R"], ys, n)
                for p, te in proto.items():
                    oracle_sim[p].append(acc(lam_o[D["L"][te]] - lam_o[D["R"][te]], ys[te]))
                    rho_b[p].append(match_predictor(lam_s, ys[te], D["L"][te], D["R"][te], obs_acc[p], rng))
                if main_v:
                    lam_fit = fit_bt_city(D["L"], D["R"], ys, D["city"], n, nc)
                    d_real = np.abs(lam_fit[D["L"]] - lam_fit[D["R"]]).mean()
                    d_perm = np.abs(lam_fit[D["L"]] - lam_fit[rng.permutation(D["R"])]).mean()
                    pair_ratio_sim.append(d_real / d_perm)
            log(f"[{vname}] bootstrap {b}: τ_b = {tau_b[-1]:.3f}")
        q_lo, q_hi = np.nanpercentile(tau_b, [2.5, 97.5])
        lo, hi = max(2 * tau_hat - q_hi, 1e-6), 2 * tau_hat - q_lo          # bootstrap básico (corrige el sesgo)
        res = {"tau_hat": tau_hat, "tau_ci95": [float(lo), float(hi)], "tau_boot": tau_b,
               "tau_boot_bias": float(np.nanmean(tau_b) - tau_hat), "ci_method": "basic bootstrap",
               "S_curve_cubic": curve(g).tolist(), "protocols": {}}
        for p, te in proto.items():
            m = mu_of[link][D["L"][te]] - mu_of[link][D["R"][te]]
            C = ceiling(m, tau_hat, link, dist)
            C_lo, C_hi = ceiling(m, lo, link, dist), ceiling(m, hi, link, dist)   # C es monótono en τ
            a = obs_acc[p]
            res["protocols"][p] = {
                "n_test_pairs": int(len(te)), "ceiling": C, "ceiling_ci95": [float(C_lo), float(C_hi)],
                "ceiling_city_only": ceiling(m, 1e-6, link),
                "dinov2_accuracy": a, "efficiency_above_chance": (a - 0.5) / (C - 0.5),
                "efficiency_ci95": [float((a - 0.5) / (C_hi - 0.5)), float((a - 0.5) / (C_lo - 0.5))],
                "gap_pts": 100 * (C - a),
                "oracle_in_sample_sim_mean": float(np.mean(oracle_sim[p])) if oracle_sim[p] else None,
                "oracle_in_sample_observed": oracle_obs.get(p),
                "rho_model_latent": float(np.mean(rho_b[p])) if rho_b[p] else None,
                "r2_model_latent": float(np.mean(rho_b[p]) ** 2) if rho_b[p] else None}
            log(f"[{vname}] {p}: techo {C:.4f} [{C_lo:.4f}, {C_hi:.4f}]; DINOv2 {a:.4f} = "
                f"{res['protocols'][p]['efficiency_above_chance']:.1%} del rango; oráculo sim "
                f"{res['protocols'][p]['oracle_in_sample_sim_mean']} vs obs {oracle_obs.get(p)}; "
                f"ρ(modelo, λ) {res['protocols'][p]['rho_model_latent']}")
        if main_v:
            d_real = np.abs(lam_full[D["L"]] - lam_full[D["R"]]).mean()
            perm = [np.abs(lam_full[D["L"]] - lam_full[rng.permutation(D["R"])]).mean() for _ in range(20)]
            res["pairing_diagnostic"] = {"ratio_observed": float(d_real / np.mean(perm)),
                                         "ratio_simulated_independent": pair_ratio_sim,
                                         "note": "ratio < simulado ⇒ pares emparejados por similitud (adaptativo)"}
            log(f"emparejamiento: ratio |Δλ̂| real/permutado obs {d_real / np.mean(perm):.4f} vs sim "
                f"{np.round(pair_ratio_sim, 4).tolist()}")
        summary["links"][vname] = res

    # (b) recuperación de τ conocido (logit)
    if not args.quick:
        rec = []
        g = grid[grid["variant"] == "logit_gauss"]
        for tau_true in (0.45, 0.6, 0.95):
            for k in range(2):
                ys, _ = simulate_votes(mu_img, tau_true, D, rng, "logit")
                rec.append({"tau_true": tau_true, "tau_recovered": invert(g, heldout_stat(D, ys, [splits[k]]))})
                log(f"recuperación: τ={tau_true} → {rec[-1]['tau_recovered']:.3f}")
        summary["recovery"] = rec
    summary["minutes"] = (time.perf_counter() - t0) / 60
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"listo en {summary['minutes']:.1f} min → {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
