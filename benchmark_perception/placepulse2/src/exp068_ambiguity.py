"""
EXP-068 (flagship V3) — ¿El modelo sabe cuándo la gente duda?
Empates humanos, calibración, abstención selectiva y heterogeneidad de votantes. Pre-registro:
docs/EXPERIMENT_REGISTRY.md "Programa V3 (14/11/2026)" (SHA256 del snapshot 8c0506...).

Uso:  python src/exp068_ambiguity.py [--boot 1000]
Salida: outputs/exp068/{summary.json, by_city.csv, risk_coverage.csv, reliability.csv, voter_dispersion.csv,
        tie_margin.csv, run.log, config.json, figures/*.png}
CPU. ~10–20 min.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, environment_info, log  # noqa: E402
from v3_common import (ENCODER, REG_ACC, SEED, assign_rows, ece_equal_mass, fit_linear_bt, fit_temperature,  # noqa: E402
                       image_partition, load_pairs, load_split, load_ties, load_Z, nll, pair_idx, pigeonhole_ci,
                       pigeonhole_stat, risk_coverage, sigmoid, verify_partition, weighted_auc)

OUT = OUTPUTS / "exp068"
TARGET_ACC = 0.75


# ---------------------------------------------------------------- Davidson
def davidson_probs(d, nu):
    """d = β(s_i − s_j). P(i), P(j), P(empate) del modelo de Davidson (1970) dividido por √(π_i π_j)."""
    a, b = np.exp(d / 2), np.exp(-d / 2)
    den = a + b + nu
    return a / den, b / den, nu / den


def fit_davidson(d0, outcome):
    """outcome: 1 = gana i, 0 = gana j, 2 = empate. Ajusta β (escala de d0) y ν>0 por máxima verosimilitud."""
    from scipy.optimize import minimize

    def nll_fn(th):
        beta, lognu = th
        pi, pj, pt = davidson_probs(beta * d0, np.exp(lognu))
        p = np.where(outcome == 1, pi, np.where(outcome == 0, pj, pt))
        return -np.log(np.clip(p, 1e-12, 1)).mean()
    r = minimize(nll_fn, x0=[1.0, np.log(0.3)], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8})
    return float(r.x[0]), float(np.exp(r.x[1]))


def fit_const_tie(d0, outcome):
    """Baseline: P(empate) = t constante; decisivos ~ (1−t)·σ(β d0). β ajustado solo con decisivos (BT usual)."""
    from scipy.optimize import minimize_scalar
    dec = outcome != 2
    yd = (outcome[dec] == 1).astype(float)
    r = minimize_scalar(lambda b: nll(sigmoid(b * d0[dec]), yd), bounds=(0.1, 5), method="bounded")
    return float(r.x), float((outcome == 2).mean())


def loss3(p_i, p_j, p_t, outcome):
    p = np.where(outcome == 1, p_i, np.where(outcome == 0, p_j, p_t))
    return -np.log(np.clip(p, 1e-12, 1))


# ---------------------------------------------------------------- protocolo
def run_protocol(name, X, row, df, ties, split, tie_part, boot, res, figs):
    """Ajusta la lineal-BT, reproduce la exactitud registrada y corre H-068a..f para un protocolo."""
    test_key = "test" if name == "image_disjoint" else "test_strict"
    li, ri, _ = pair_idx(df.loc[split["train"], "left"], df.loc[split["train"], "right"], row)
    y = df.loc[split["train"], "label"].to_numpy()
    vli, vri, _ = pair_idx(df.loc[split["val"], "left"], df.loc[split["val"], "right"], row)
    vy = df.loc[split["val"], "label"].to_numpy()
    t0 = time.time()
    g, info = fit_linear_bt(X, li, ri, y, vli, vri, vy)
    T = df.loc[split[test_key]]
    tli, tri, keep = pair_idx(T["left"], T["right"], row)
    T = T[keep]
    ty = T["label"].to_numpy()
    logit = g[tli] - g[tri]
    acc = float(((logit > 0) == (ty == 1)).mean())
    log(f"[{name}] lineal-BT re-ajustada: test acc {acc:.4f} (registrada {REG_ACC[name]}) "
        f"ép. {info['epochs_best']}  {time.time()-t0:.0f}s")
    if abs(acc - REG_ACC[name]) > 0.002:
        raise RuntimeError(f"No se reproduce la exactitud registrada en {name}: {acc:.4f}")
    R = {"n_test_decisive": int(len(ty)), "acc_test_reproduced": acc, "acc_registered": REG_ACC[name],
         "epochs_best": info["epochs_best"]}

    # ---------------- H-068c calibración (temperatura en val)
    vlogit = g[vli] - g[vri]
    Tval = fit_temperature(vlogit, vy)
    p_raw, p_cal = sigmoid(logit), sigmoid(logit / Tval)
    R["calibration"] = {"temperature_val": Tval}
    for lab, p in (("raw", p_raw), ("temp", p_cal)):
        e, elo, ehi = pigeonhole_stat(lambda w, p=p: ece_equal_mass(p, ty, 15, w), tli, tri, B=max(200, boot // 2))
        R["calibration"][lab] = {"ece15": e, "ece_ci": [elo, ehi], "nll": nll(p, ty),
                                 "brier": float(np.mean((p - ty) ** 2))}
    d_nll = loss_bin(p_cal, ty) - loss_bin(p_raw, ty)
    est, lo, hi = pigeonhole_ci(d_nll, None, tli, tri, B=boot)
    R["calibration"]["delta_nll_temp_minus_raw"] = [est, lo, hi]
    # tabla de fiabilidad (10 bins de igual masa sobre la confianza)
    conf = np.maximum(p_cal, 1 - p_cal)
    corr = ((logit > 0) == (ty == 1)).astype(float)
    qs = np.quantile(conf, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(qs, conf, side="right") - 1, 0, 9)
    rel = pd.DataFrame({"bin": b, "conf": conf, "corr": corr}).groupby("bin").agg(
        conf_mean=("conf", "mean"), acc=("corr", "mean"), n=("corr", "size")).reset_index()
    rel.insert(0, "protocol", name)
    figs.setdefault("reliability", []).append(rel)

    # ---------------- H-068d abstención
    vconf = np.abs(sigmoid(vlogit / Tval) - 0.5)
    vcorr = ((vlogit > 0) == (vy == 1)).astype(float)
    order = np.argsort(-vconf)
    vacc_cum = np.cumsum(vcorr[order]) / np.arange(1, len(order) + 1)
    ok = np.where(vacc_cum >= TARGET_ACC)[0]
    min_n = 200  # no elegir τ con menos de 200 pares de val
    ok = ok[ok >= min_n - 1]
    tau = float(vconf[order][ok.max()]) if len(ok) else np.nan
    tconf = np.abs(p_cal - 0.5)
    sel = tconf >= tau
    acc_sel = pigeonhole_stat(lambda w: np.average(corr[sel], weights=None if w is None else w[sel])
                              if (w is None or w[sel].sum() > 0) else np.nan, tli, tri, B=boot)
    cov_sel = pigeonhole_stat(lambda w: np.average(sel.astype(float), weights=w), tli, tri, B=boot)
    cov, accc, aurc = risk_coverage(tconf, corr)
    aurc_rand = 1 - corr.mean()
    d_aurc = pigeonhole_stat(lambda w: risk_coverage(tconf, corr, w)[2] - (1 - np.average(corr, weights=w)),
                             tli, tri, B=max(200, boot // 2))
    R["abstention"] = {"target_acc_val": TARGET_ACC, "tau_val": tau,
                       "val_coverage_at_tau": float((vconf >= tau).mean()),
                       "test_acc_at_tau": list(acc_sel), "test_coverage_at_tau": list(cov_sel),
                       "aurc": aurc, "aurc_random": float(aurc_rand), "delta_aurc_vs_random": list(d_aurc)}
    for c in (1.0, 0.8, 0.5, 0.2):
        k = int(np.searchsorted(cov, c) if c < 1 else len(cov) - 1)
        R["abstention"][f"acc_at_cov_{int(c*100)}"] = float(accc[min(k, len(accc) - 1)])
    grid = np.linspace(0.05, 1, 20)
    figs.setdefault("risk_coverage", []).append(pd.DataFrame(
        {"protocol": name, "coverage": grid, "accuracy": [accc[min(int(np.searchsorted(cov, c)), len(accc) - 1)]
                                                          for c in grid]}))

    # ---------------- H-068a empates vs decisivos (margen)
    if name == "image_disjoint":
        tt = ties[tie_part == "test"]
    else:
        held = set(split.get("holdout_cities", [])) or set(HOLDOUT)
        tt = ties[ties["city_left"].isin(held) & ties["city_right"].isin(held)]
    xli, xri, k2 = pair_idx(tt["left"], tt["right"], row)
    tt = tt[k2]
    m_dec, m_tie = np.abs(logit), np.abs(g[xli] - g[xri])
    allm = np.r_[m_dec, m_tie]
    lab = np.r_[np.zeros(len(m_dec)), np.ones(len(m_tie))]
    ALI, ARI = np.r_[tli, xli], np.r_[tri, xri]
    auc = pigeonhole_stat(lambda w: weighted_auc(-allm, lab, w), ALI, ARI, B=max(200, boot // 2))
    R["ties"] = {"n_test_ties": int(len(m_tie)), "tie_rate_test": float(len(m_tie) / (len(m_tie) + len(m_dec))),
                 "auc_minus_margin_predicts_tie": list(auc),
                 "median_margin_dec": float(np.median(m_dec)), "median_margin_tie": float(np.median(m_tie))}
    # tasa de empate por quintil de margen (descriptivo)
    qe = np.quantile(allm, np.linspace(0, 1, 6))
    qb = np.clip(np.searchsorted(qe, allm, side="right") - 1, 0, 4)
    R["ties"]["tie_rate_by_margin_quintile"] = [float(lab[qb == k].mean()) for k in range(5)]
    figs.setdefault("tie_margin", []).append(pd.DataFrame(
        {"protocol": name, "margin_quintile": range(1, 6), "tie_rate": R["ties"]["tie_rate_by_margin_quintile"]}))

    # control del votante: AUROC estratificado dentro de votante + logística ajustada por propensión LOO
    vot = np.r_[T["voter_uniqueid"].to_numpy(), tt["voter_uniqueid"].to_numpy()]
    R["ties"]["voter_control"] = voter_control(allm, lab, vot, boot)

    # ---------------- H-068b Davidson (ajuste en train+val, evaluación en test)
    if name == "image_disjoint":
        fit_rows = []
        for part in ("train", "val"):
            D = df.loc[split[part]]
            a, b_, k = pair_idx(D["left"], D["right"], row)
            fit_rows.append((g[a] - g[b_], D["label"].to_numpy()[k].astype(int)))
            Tp = ties[tie_part == part]
            a, b_, k = pair_idx(Tp["left"], Tp["right"], row)
            fit_rows.append((g[a] - g[b_], np.full(k.sum(), 2)))
        d_fit = np.concatenate([r[0] for r in fit_rows])
        o_fit = np.concatenate([r[1] for r in fit_rows])
        beta_d, nu = fit_davidson(d_fit, o_fit)
        beta_c, trate = fit_const_tie(d_fit, o_fit)
        d_te = np.r_[logit, g[xli] - g[xri]]
        o_te = np.r_[ty.astype(int), np.full(len(m_tie), 2)]
        L_dav = loss3(*davidson_probs(beta_d * d_te, nu), o_te)
        pi_c = (1 - trate) * sigmoid(beta_c * d_te)
        L_con = loss3(pi_c, (1 - trate) - pi_c, np.full(len(d_te), trate), o_te)
        est, lo, hi = pigeonhole_ci(L_dav, L_con, ALI, ARI, B=boot)
        R["davidson"] = {"beta": beta_d, "nu": nu, "beta_const": beta_c, "tie_rate_train": trate,
                         "n_fit": int(len(o_fit)), "n_test": int(len(o_te)),
                         "logloss3_davidson": float(L_dav.mean()), "logloss3_const": float(L_con.mean()),
                         "delta_davidson_minus_const": [est, lo, hi]}

    # ---------------- H-068e dispersión por votante (con p calibrado)
    R["voter_dispersion"], vd = voter_dispersion(T["voter_uniqueid"].to_numpy(), p_cal, ty, boot)
    vd.insert(0, "protocol", name)
    figs.setdefault("voter_dispersion", []).append(vd)

    # ---------------- H-068f por ciudad (solo city_holdout)
    if name == "city_holdout":
        rows = []
        cities = HOLDOUT
        for c in cities:
            m = ((T["city_left"] == c) | (T["city_right"] == c)).to_numpy()
            if m.sum() < 50:
                continue
            e_, lo_, hi_ = pigeonhole_ci(corr[m], None, tli[m], tri[m], B=boot)
            rows.append({"city": c, "n_pairs": int(m.sum()), "acc": e_, "lo": lo_, "hi": hi_,
                         "ece15": ece_equal_mass(p_cal[m], ty[m], 10)})
        bc = pd.DataFrame(rows)
        se = (bc["hi"] - bc["lo"]) / 3.92
        wv = 1 / se ** 2
        mu_fe = (wv * bc["acc"]).sum() / wv.sum()
        Q = float((wv * (bc["acc"] - mu_fe) ** 2).sum())
        dfq = len(bc) - 1
        tau2 = max(0.0, (Q - dfq) / (wv.sum() - (wv ** 2).sum() / wv.sum()))
        w2 = 1 / (se ** 2 + tau2)
        mu = float((w2 * bc["acc"]).sum() / w2.sum())
        R["by_city"] = {"Q": Q, "df": dfq, "I2": float(max(0.0, (Q - dfq) / Q)) if Q > 0 else 0.0,
                        "tau": float(np.sqrt(tau2)), "mu_DL": mu, "se_mu": float(np.sqrt(1 / w2.sum())),
                        "note": "pares pueden involucrar 2 ciudades retenidas; las ciudades no son independientes"}
        bc.to_csv(OUT / "by_city.csv", index=False)
        figs["by_city"] = bc
    res[name] = R


def loss_bin(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def voter_control(margin, is_tie, voter, boot):
    """(1) AUROC de −margen dentro de cada votante (Mann–Whitney agregado sobre votantes con ≥1 empate y ≥1 decisivo);
    IC por bootstrap de votantes. (2) Logística: empate ~ z(margen) + logit(propensión LOO del votante); OR por DE."""
    from sklearn.linear_model import LogisticRegression
    d = pd.DataFrame({"m": margin, "t": is_tie, "v": voter})
    st = d.groupby("v")["t"].agg(["sum", "size"])
    elig = st[(st["sum"] >= 1) & (st["sum"] < st["size"])].index
    E = d[d["v"].isin(elig)]
    groups = {v: (g["m"].to_numpy(), g["t"].to_numpy()) for v, g in E.groupby("v")}

    def strat(vs):
        num = den = 0.0
        for v in vs:
            m, t = groups[v]
            mt, md = m[t == 1], m[t == 0]
            # P(margen_empate < margen_decisivo) + 0.5 empates
            cmp = (mt[:, None] < md[None, :]).sum() + 0.5 * (mt[:, None] == md[None, :]).sum()
            num += cmp
            den += len(mt) * len(md)
        return num / den
    vs = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    bs = [strat(vs[rng.integers(0, len(vs), len(vs))]) for _ in range(max(200, boot // 5))]
    out = {"n_voters_eligible": int(len(vs)), "n_votes_eligible": int(len(E)),
           "within_voter_auc": [float(strat(vs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    # logística ajustada (todos los votantes de test): propensión del votante a empatar, calculada con TODOS sus votos
    # del dataset (decisivos + empates, cualquier partición) dejando fuera el voto actual; suavizada (prior 12.4 %, peso 10).
    # Los empates nunca entran al entrenamiento del modelo, así que esto no filtra información al puntaje.
    tot = d["v"].map(VOTER_TIES).fillna(0)
    n = d["v"].map(VOTER_N).fillna(1)
    prop = (tot - d["t"] + 10 * 0.124) / (n - 1 + 10)
    Z = np.c_[(d["m"] - d["m"].mean()) / d["m"].std(), np.log(prop / (1 - prop))]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(Z, d["t"])
    ors = []
    allv = d["v"].unique()
    idx_by_v = d.groupby("v").indices
    for _ in range(200):
        pick = allv[rng.integers(0, len(allv), len(allv))]
        ii = np.concatenate([idx_by_v[v] for v in pick])
        ors.append(np.exp(LogisticRegression(C=1e6, max_iter=500).fit(Z[ii], d["t"].to_numpy()[ii]).coef_[0, 0]))
    out["adjusted_logit_OR_per_SD_margin"] = [float(np.exp(lr.coef_[0, 0])), float(np.percentile(ors, 2.5)),
                                              float(np.percentile(ors, 97.5))]
    out["adjusted_logit_coef_voter_propensity"] = float(lr.coef_[0, 1])
    return out


def voter_dispersion(voter, p_cal, y, boot, min_votes=10):
    """φ = Σ_v (k_v − E k_v)² / Σ_v Var k_v, con k_v = acuerdos del votante con el modelo y
    q = P(acuerdo) = max(p, 1−p) calibrado (Poisson-binomial). Nulo: bootstrap paramétrico simulando votos desde q."""
    q = np.maximum(p_cal, 1 - p_cal)
    agree = ((p_cal >= 0.5) == (y == 1)).astype(float)
    d = pd.DataFrame({"v": voter, "q": q, "a": agree})
    g = d.groupby("v").agg(n=("a", "size"), k=("a", "sum"), Ek=("q", "sum"),
                           Vk=("q", lambda s: float((s * (1 - s)).sum())))
    g = g[g["n"] >= min_votes]
    phi = float(((g["k"] - g["Ek"]) ** 2).sum() / g["Vk"].sum())
    rng = np.random.default_rng(SEED)
    sub = d[d["v"].isin(g.index)]
    codes = pd.factorize(sub["v"])[0]
    qs = sub["q"].to_numpy()
    Ek = np.bincount(codes, weights=qs)
    Vk = np.bincount(codes, weights=qs * (1 - qs))
    null = []
    for _ in range(max(200, boot // 2)):
        ksim = np.bincount(codes, weights=(rng.random(len(qs)) < qs).astype(float))
        null.append(((ksim - Ek) ** 2).sum() / Vk.sum())
    # IC de φ observado: bootstrap de votantes
    gi = g.reset_index(drop=True)
    phis = []
    for _ in range(max(200, boot // 2)):
        s = gi.iloc[rng.integers(0, len(gi), len(gi))]
        phis.append(((s["k"] - s["Ek"]) ** 2).sum() / s["Vk"].sum())
    out = {"n_voters": int(len(g)), "n_votes": int(g["n"].sum()), "phi": phi,
           "phi_ci_voter_boot": [float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5))],
           "null_phi_mean": float(np.mean(null)), "null_phi_95": float(np.percentile(null, 95)),
           "p_value_one_sided": float((np.sum(np.array(null) >= phi) + 1) / (len(null) + 1)),
           "agree_rate_quantiles_10_50_90": [float(x) for x in (g["k"] / g["n"]).quantile([.1, .5, .9])]}
    # EXPLORATORIO (no pre-registrado, añadido tras el smoke): ¿la sobre-dispersión es sesgo de lado (clic fijo)?
    lr_all = VOTER_LEFT.reindex(g.index)
    mid = (lr_all >= 0.2) & (lr_all <= 0.8)
    gm = g[mid.fillna(False).to_numpy()]
    out["exploratory_side_bias"] = {
        "def": "tasa de elección del lado izquierdo del votante en TODOS sus votos decisivos",
        "frac_voters_left_rate_outside_20_80": float((~mid.fillna(False)).mean()),
        "phi_excluding_side_biased": float(((gm["k"] - gm["Ek"]) ** 2).sum() / gm["Vk"].sum()),
        "n_voters_kept": int(len(gm))}
    tab = g.assign(rate=g["k"] / g["n"], left_rate=lr_all.to_numpy()).reset_index()[["n", "k", "Ek", "rate",
                                                                                      "left_rate"]]
    return out, tab


def test_retest(df, ties):
    """Consistencia intra-votante: mismo votante, mismo par no ordenado, ≥ 2 votos. Se comparan los 2 primeros votos."""
    a = pd.concat([df.assign(out=np.where(df["label"] == 1, df["left"], df["right"])),
                   ties.assign(out="TIE")], ignore_index=True)
    a["key"] = np.where(a["left"] < a["right"], a["left"] + "|" + a["right"], a["right"] + "|" + a["left"])
    a = a.sort_values("timestamp")
    g = a.groupby(["voter_uniqueid", "key"])["out"].apply(list)
    g = g[g.apply(len) >= 2]
    first2 = g.apply(lambda l: (l[0], l[1]))
    same = first2.apply(lambda t: t[0] == t[1])
    both_dec = first2.apply(lambda t: t[0] != "TIE" and t[1] != "TIE")
    return {"n_repeated_voter_pairs": int(len(g)), "identical_outcome_rate": float(same.mean()),
            "n_both_decisive": int(both_dec.sum()),
            "same_winner_rate_if_both_decisive": float(same[both_dec].mean()),
            "any_tie_rate": float((~both_dec).mean()),
            "caveat": "repetición inmediata posible (misma sesión): consistencia de memoria + juicio; no es acuerdo entre personas"}


HOLDOUT = ["Tokyo", "Bangkok", "Cape Town", "Gaborone", "Rio De Janeiro", "Santiago", "Moscow", "Barcelona",
           "Melbourne", "Houston"]


def make_figures(figs):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib.pyplot as plt
    from figstyle import INK2, NEUTRAL, SERIES, setup
    setup()
    fd = OUT / "figures"
    fd.mkdir(exist_ok=True)
    names = {"image_disjoint": "Fotos nuevas", "city_holdout": "Ciudades nuevas"}
    # riesgo–cobertura
    rc = pd.concat(figs["risk_coverage"])
    rc.to_csv(OUT / "risk_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, (p, s) in enumerate(rc.groupby("protocol", sort=False)):
        ax.plot(s["coverage"] * 100, s["accuracy"] * 100, color=SERIES[k], marker="o", ms=3, label=names[p])
    ax.axhline(50, color=NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("Cobertura (% de pares en que el modelo responde)")
    ax.set_ylabel("Exactitud en los pares respondidos (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fd / "risk_coverage_es.png", dpi=200)
    # fiabilidad
    rel = pd.concat(figs["reliability"])
    rel.to_csv(OUT / "reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([50, 100], [50, 100], color=NEUTRAL, lw=1, ls=":")
    for k, (p, s) in enumerate(rel.groupby("protocol", sort=False)):
        ax.plot(s["conf_mean"] * 100, s["acc"] * 100, color=SERIES[k], marker="o", ms=4, label=names[p])
    ax.set_xlabel("Confianza predicha (%)")
    ax.set_ylabel("Exactitud observada (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fd / "reliability_es.png", dpi=200)
    # empate por quintil de margen
    tm = pd.concat(figs["tie_margin"])
    tm.to_csv(OUT / "tie_margin.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    w = 0.38
    for k, (p, s) in enumerate(tm.groupby("protocol", sort=False)):
        ax.bar(s["margin_quintile"] + (k - 0.5) * w, s["tie_rate"] * 100, width=w, color=SERIES[k], label=names[p])
    ax.set_xlabel("Quintil de |s_i − s_j| (1 = el modelo duda más)")
    ax.set_ylabel("Pares declarados 'iguales' (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fd / "tie_rate_by_margin_es.png", dpi=200)
    # ciudades
    if "by_city" in figs:
        bc = figs["by_city"].sort_values("acc")
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        yy = np.arange(len(bc))
        ax.errorbar(bc["acc"] * 100, yy, xerr=[(bc["acc"] - bc["lo"]) * 100, (bc["hi"] - bc["acc"]) * 100],
                    fmt="o", color=SERIES[1], ecolor=INK2, capsize=2)
        ax.set_yticks(yy, [f"{c} (n={n:,})" for c, n in zip(bc["city"], bc["n_pairs"])])
        ax.axvline(50, color=NEUTRAL, lw=1, ls=":")
        ax.set_xlabel("Exactitud por pares (%) · IC 95 % bootstrap de imágenes")
        fig.tight_layout()
        fig.savefig(fd / "by_city_es.png", dpi=200)
    vd = pd.concat(figs["voter_dispersion"])
    vd.to_csv(OUT / "voter_dispersion.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, row = load_Z(ENCODER)
    df, ties = load_pairs(), load_ties()
    log(f"features {X.shape} · decisivos {len(df):,} · empates {len(ties):,}")
    split_d = load_split("image_disjoint")
    assign = image_partition(df)
    verify_partition(df, assign, split_d)
    tie_part = assign_rows(ties["left"], ties["right"], assign)
    global VOTER_TIES, VOTER_N, VOTER_LEFT
    VOTER_TIES = ties["voter_uniqueid"].value_counts()
    VOTER_N = (df["voter_uniqueid"].value_counts().add(VOTER_TIES, fill_value=0))
    VOTER_LEFT = df.groupby("voter_uniqueid")["label"].mean()
    log(f"empates por partición: {tie_part.value_counts(dropna=False).to_dict()}")
    res, figs = {}, {}
    run_protocol("image_disjoint", X, row, df, ties, split_d, tie_part, args.boot, res, figs)
    run_protocol("city_holdout", X, row, df, ties, load_split("city_holdout"), None, args.boot, res, figs)
    res["test_retest"] = test_retest(df, ties)
    make_figures(figs)
    res["meta"] = {"encoder": ENCODER, "seed": SEED, "boot": args.boot, "runtime_s": round(time.time() - t0),
                   "prereg": "docs/EXPERIMENT_REGISTRY.md · Programa V3 (14/11/2026) · snapshot SHA256 8c0506...",
                   "deviation": "control del votante: AUROC estratificado por votante + logística ajustada por propensión "
                                "LOO (statsmodels no instalado para logit condicional exacto)",
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"args": vars(args), "env": environment_info()}, indent=2,
                                                ensure_ascii=False), encoding="utf-8")
    log(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
    log(f"listo en {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
