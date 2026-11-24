"""
EXP-071 — Where does the model fail? Error analysis requested in the advisory of 15/11/2026.
Pre-registration: docs/EXPERIMENT_REGISTRY.md, "EXP-071".

CPU only, existing DINOv2-B features, image_disjoint partition, seed 42 (same baseline as EXP-068/069).
  1. Reproduce the linear-BT safe head (0.6785) or abort.
  2. Vote consensus for every test pair: Bradley-Terry fitted on the OTHER test votes (5 folds of pairs).
     Categories: hit with consensus / lucky hit (vote and model against consensus) / vote against consensus
     (model agrees with consensus) / model error (vote and consensus against the model) / no clear consensus.
  3. Cross-question confusion: human "wealthier" BT score of each image (all wealthy votes) and the model's
     own wealthy head; safe accuracy where wealth order agrees or disagrees with the safety vote.
  4. Map and neighbourhood: Gaussian-weighted mean of the model scores of nearby images (same city, any split),
     discrepancy d = s - n, error rate by |d|, and a one-parameter diagnostic s + beta*n (beta on validation).
  5. Figures and CSV files for manual review (random hits, most confident model errors).

Usage: python src/exp071_error_analysis.py [--boot 1000]      Output: outputs/exp071/
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MANIFESTS, OUTPUTS, ZIP_PATH, log  # noqa: E402
from v3_common import (ENCODER, REG_ACC, SEED, assign_rows, fit_linear_bt, image_partition, load_pairs,  # noqa: E402
                       load_split, load_Z, pair_idx, pigeonhole_ci, sigmoid, verify_partition)
from exp063_heads import bt_fit  # noqa: E402
from exp069_construct import load_votes  # noqa: E402

OUT = OUTPUTS / "exp071"
FIG = OUT / "figures"
CONS_THR = 0.40          # |lambda_i - lambda_j| >= 0.40  <=>  consensus probability >= 0.60
SIGMAS = (100.0, 250.0, 500.0)
EARTH_R = 6_371_000.0


def consensus_folds(li, ri, y, ali=None, ari=None, ay=None, k=5, seed=SEED):
    """Consensus margin for every test pair from a BT fit on the other test folds plus (optionally) all the other
    human votes ali/ari/ay (pairs that are not test pairs). The pair's own vote never enters. NaN if absent."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(li))
    cm = np.full(len(li), np.nan)
    for f in range(k):
        tr, te = fold != f, fold == f
        if ali is not None:
            rows, lam = bt_fit(np.r_[li[tr], ali], np.r_[ri[tr], ari], np.r_[y[tr], ay])
        else:
            rows, lam = bt_fit(li[tr], ri[tr], y[tr])
        pos = {r: i for i, r in enumerate(rows)}
        a = np.array([pos.get(r, -1) for r in li[te]])
        b = np.array([pos.get(r, -1) for r in ri[te]])
        ok = (a >= 0) & (b >= 0)
        v = np.full(te.sum(), np.nan)
        v[ok] = lam[a[ok]] - lam[b[ok]]
        cm[te] = v
    return cm


def coords_from_paths(paths: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = paths.str.replace("images/", "", regex=False).str.split("_")
    return parts.str[0].astype(float).to_numpy(), parts.str[1].astype(float).to_numpy()


def neighbourhood_scores(lat, lon, city, s, sigma):
    """Gaussian-weighted mean of s over other images of the same city within 3 sigma. NaN if no neighbour."""
    from sklearn.neighbors import BallTree
    n = np.full(len(s), np.nan)
    cnt = np.zeros(len(s), dtype=int)
    X = np.radians(np.c_[lat, lon])
    for c in np.unique(city):
        idx = np.where(city == c)[0]
        tree = BallTree(X[idx], metric="haversine")
        nb, dist = tree.query_radius(X[idx], r=3 * sigma / EARTH_R, return_distance=True)
        for j, (ii, dd) in enumerate(zip(nb, dist)):
            m = ii != j
            ii, dd = ii[m], dd[m] * EARTH_R
            if len(ii):
                w = np.exp(-0.5 * (dd / sigma) ** 2)
                n[idx[j]] = float(w @ s[idx[ii]] / w.sum())
                cnt[idx[j]] = len(ii)
    return n, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, row = load_Z(ENCODER)
    df = load_pairs()
    split = load_split("image_disjoint")
    assign = image_partition(df)
    verify_partition(df, assign, split)

    # ---------------------------------------------------------------- 1. baseline safe head
    tr, va, te = df.loc[split["train"]], df.loc[split["val"]], df.loc[split["test"]]
    li, ri, k = pair_idx(tr["left"], tr["right"], row)
    vli, vri, vk = pair_idx(va["left"], va["right"], row)
    tli, tri, tk = pair_idx(te["left"], te["right"], row)
    y_tr, y_va, y = tr["label"].to_numpy()[k], va["label"].to_numpy()[vk], te["label"].to_numpy()[tk]
    s, _ = fit_linear_bt(X, li, ri, y_tr, vli, vri, y_va)
    m = s[tli] - s[tri]
    correct = ((m > 0) == (y == 1)).astype(float)
    acc = correct.mean()
    log(f"safe test acc {acc:.4f} (registered {REG_ACC['image_disjoint']})")
    if abs(acc - REG_ACC["image_disjoint"]) > 0.002:
        raise RuntimeError("baseline not reproduced")
    res: dict = {"n_test_pairs": int(len(y)), "acc_baseline": float(acc)}
    te = te[tk].reset_index(drop=True)

    # ---------------------------------------------------------------- 2. consensus categories
    # all other human safety votes (train-train, val, and the mixed pairs dropped by image_disjoint)
    test_set = set(split["test"])
    other = df.loc[[i for i in df.index if i not in test_set]]
    ali, ari, ak = pair_idx(other["left"], other["right"], row)
    ay = other["label"].to_numpy()[ak].astype(float)
    cm_test_only = consensus_folds(tli, tri, y.astype(float))
    cm = consensus_folds(tli, tri, y.astype(float), ali, ari, ay)
    res["consensus_sign_agreement_testonly_vs_all"] = float(
        np.nanmean(np.sign(cm_test_only) == np.sign(cm)))
    test_imgs = np.unique(np.r_[tli, tri])
    allv = np.bincount(np.r_[tli, tri, ali, ari], minlength=X.shape[0])
    res["safety_votes_per_test_image_median"] = float(np.median(allv[test_imgs]))
    res["test_votes_per_image_median"] = float(np.median(np.bincount(np.r_[tli, tri])[test_imgs]))
    clear = np.isfinite(cm) & (np.abs(cm) >= CONS_THR)
    cons_dir = cm > 0                      # consensus says left
    vote_dir = y == 1
    model_dir = m > 0
    cat = np.full(len(y), "no_clear_consensus", dtype=object)
    cat[clear & (model_dir == vote_dir) & (vote_dir == cons_dir)] = "hit_with_consensus"
    cat[clear & (model_dir == vote_dir) & (vote_dir != cons_dir)] = "lucky_hit"
    cat[clear & (model_dir != vote_dir) & (model_dir == cons_dir)] = "vote_against_consensus"
    cat[clear & (model_dir != vote_dir) & (vote_dir == cons_dir)] = "model_error"
    order = ["hit_with_consensus", "lucky_hit", "vote_against_consensus", "model_error", "no_clear_consensus"]
    cats = {c: {"n": int((cat == c).sum()), "share": float((cat == c).mean())} for c in order}
    res["categories"] = cats
    res["share_clear_consensus"] = float(clear.mean())
    # accuracy against the consensus (instead of the single vote) on clear pairs
    agree_cons = (model_dir == cons_dir).astype(float)
    res["acc_vs_consensus_clear"] = pigeonhole_ci(agree_cons[clear], None, tli[clear], tri[clear], B=args.boot)
    res["acc_vs_vote_clear"] = float(correct[clear].mean())
    res["vote_agrees_consensus_clear"] = float((vote_dir == cons_dir)[clear].mean())
    errs = correct == 0
    res["share_errors_that_are_vote_against_consensus"] = float((cat == "vote_against_consensus").sum() / errs.sum())
    res["share_errors_that_are_model_errors"] = float((cat == "model_error").sum() / errs.sum())
    res["share_errors_no_clear_consensus"] = float(((cat == "no_clear_consensus") & errs).sum() / errs.sum())
    conf = sigmoid(np.abs(m))
    res["mean_conf"] = {c: float(conf[cat == c].mean()) for c in order}
    log(f"categories: { {c: round(v['share'], 3) for c, v in cats.items()} }")

    # ---------------------------------------------------------------- 3. cross-question confusion
    v = load_votes(row)
    vw = v[v["attr"] == "wealthy"]
    wli, wri = vw["left"].map(row).to_numpy(), vw["right"].map(row).to_numpy()
    wrows, wlam = bt_fit(wli, wri, vw["label"].to_numpy().astype(float))
    hw = np.full(X.shape[0], np.nan)
    hw[wrows] = wlam
    hw_m = hw[tli] - hw[tri]
    known = np.isfinite(hw_m) & (np.abs(hw_m) >= CONS_THR)
    wealth_agrees = (hw_m > 0) == vote_dir
    # model's own wealthy head (same recipe as EXP-069)
    v["part"] = assign_rows(v["left"], v["right"], assign)
    wtr, wva = vw.join(v["part"]).query("part == 'train'"), vw.join(v["part"]).query("part == 'val'")
    a1, b1, k1 = pair_idx(wtr["left"], wtr["right"], row)
    a2, b2, k2 = pair_idx(wva["left"], wva["right"], row)
    gw, _ = fit_linear_bt(X, a1, b1, wtr["label"].to_numpy()[k1], a2, b2, wva["label"].to_numpy()[k2])
    mw = gw[tli] - gw[tri]
    rows_cq = []
    for name, mask in [("human wealth order agrees with safety vote", known & wealth_agrees),
                       ("human wealth order opposes safety vote", known & ~wealth_agrees),
                       ("no clear human wealth order", ~known)]:
        e, lo, hi = pigeonhole_ci(correct[mask], None, tli[mask], tri[mask], B=args.boot)
        cons_ok = clear & mask
        rows_cq.append({"subset": name, "n_pairs": int(mask.sum()), "share": float(mask.mean()),
                        "acc_safe_head": e, "lo": lo, "hi": hi,
                        "acc_wealthy_head": float((((mw > 0) == vote_dir)[mask]).mean()),
                        "vote_agrees_consensus": float(((vote_dir == cons_dir)[cons_ok]).mean())
                        if cons_ok.sum() else np.nan,
                        "model_follows_wealth": float((((m > 0) == (hw_m > 0))[mask]).mean())})
    # same split but against the safety CONSENSUS (clear pairs), so that vote noise is not read as wealth confusion
    for name, mask in [("CONSENSUS: wealth order agrees with safety consensus", clear & known & ((hw_m > 0) == cons_dir)),
                       ("CONSENSUS: wealth order opposes safety consensus", clear & known & ((hw_m > 0) != cons_dir))]:
        e, lo, hi = pigeonhole_ci(agree_cons[mask], None, tli[mask], tri[mask], B=args.boot)
        rows_cq.append({"subset": name, "n_pairs": int(mask.sum()), "share": float(mask.mean()),
                        "acc_safe_head": e, "lo": lo, "hi": hi,
                        "acc_wealthy_head": float((((mw > 0) == cons_dir)[mask]).mean()),
                        "vote_agrees_consensus": float(((vote_dir == cons_dir)[mask]).mean()),
                        "model_follows_wealth": float((((m > 0) == (hw_m > 0))[mask]).mean())})
    cq = pd.DataFrame(rows_cq)
    cq.to_csv(OUT / "cross_question.csv", index=False)
    me = cat == "model_error"
    res["model_errors_choose_wealthier_human"] = float((((m > 0) == (hw_m > 0))[me & known]).mean())
    res["hits_choose_wealthier_human"] = float((((m > 0) == (hw_m > 0))[(cat == "hit_with_consensus") & known]).mean())
    log(cq.round(3).to_string())

    # ---------------------------------------------------------------- 4. map and neighbourhood
    man = pd.read_csv(MANIFESTS / "images_manifest.csv", dtype=str).set_index("location_id")
    inv_row = np.empty(len(row), dtype=object)
    for lid, i in row.items():
        inv_row[i] = lid
    paths = man.reindex(inv_row)["image_path"]
    lat, lon = coords_from_paths(paths.fillna("images/nan_nan_x_x"))
    city = man.reindex(inv_row)["city"].fillna("?").to_numpy()
    sp = []
    nb = {}
    for sg in SIGMAS:
        n, cnt = neighbourhood_scores(lat, lon, city, s, sg)
        nb[sg] = (n, cnt)
        has = np.isfinite(n[tli]) & np.isfinite(n[tri])
        # does the neighbourhood pick the voted image where the model fails?
        nm = n[tli] - n[tri]
        n_ok = ((nm > 0) == vote_dir).astype(float)
        # discrepancy of the pair: largest |s - n| of its two images
        d = np.fmax(np.abs(s[tli] - n[tli]), np.abs(s[tri] - n[tri]))
        q = pd.qcut(d[has], 5, labels=False)
        err_by_q = [float(1 - correct[has][q == j].mean()) for j in range(5)]
        # same, within quintiles of the model margin (a discordant image is often an extreme one)
        mq = pd.qcut(np.abs(m[has]), 5, labels=False)
        strat = []
        for j in range(5):
            mm = mq == j
            dd = d[has][mm]
            hi_ = dd >= np.median(dd)
            strat.append(float((1 - correct[has][mm][hi_].mean()) - (1 - correct[has][mm][~hi_].mean())))
        # in model errors: does the neighbourhood pick the voted image more often when the discrepancy is high?
        me_has = me[has]
        dq_hi = d[has] >= np.median(d[has])
        neigh_err_hi = float(n_ok[has][me_has & dq_hi].mean())
        neigh_err_lo = float(n_ok[has][me_has & ~dq_hi].mean())
        # one-parameter diagnostic: s + beta*n, beta on validation pairs
        vhas = np.isfinite(n[vli]) & np.isfinite(n[vri])
        best_b, best_a = 0.0, -1
        for bta in np.linspace(-1, 1, 41):
            mv = (s[vli] - s[vri]) + bta * (n[vli] - n[vri])
            a_ = ((mv[vhas] > 0) == (y_va[vhas] == 1)).mean()
            if a_ > best_a:
                best_b, best_a = float(bta), a_
        mc = m + best_b * nm
        c2 = ((mc > 0) == vote_dir).astype(float)
        dlt = pigeonhole_ci(c2[has], correct[has], tli[has], tri[has], B=args.boot)
        sp.append({"sigma_m": sg, "test_pairs_with_neighbours": int(has.sum()), "share": float(has.mean()),
                   "median_neighbours": float(np.median(cnt[np.unique(np.r_[tli, tri])])),
                   "acc_model_on_subset": float(correct[has].mean()),
                   "acc_neighbourhood_only": float(n_ok[has].mean()),
                   "neigh_right_where_model_error": float(n_ok[has & me].mean()) if (has & me).sum() else np.nan,
                   "neigh_right_where_hit": float(n_ok[has & (cat == 'hit_with_consensus')].mean()),
                   "err_rate_q1_low_discrepancy": err_by_q[0], "err_rate_q5_high_discrepancy": err_by_q[4],
                   "err_by_quintile": err_by_q, "err_diff_high_minus_low_disc_within_margin_q": strat,
                   "neigh_right_in_model_errors_high_disc": neigh_err_hi,
                   "neigh_right_in_model_errors_low_disc": neigh_err_lo, "beta_val": best_b,
                   "delta_acc_pts": 100 * dlt[0], "delta_lo": 100 * dlt[1], "delta_hi": 100 * dlt[2]})
        log(f"sigma {sg:.0f} m: {sp[-1]}")
    spd = pd.DataFrame(sp)
    spd.drop(columns=["err_by_quintile", "err_diff_high_minus_low_disc_within_margin_q"]).to_csv(OUT / "neighbourhood.csv", index=False)
    res["neighbourhood"] = sp

    # ---------------------------------------------------------------- 5. review files and figures
    ids_l, ids_r = te["left"].to_numpy(), te["right"].to_numpy()
    rev = pd.DataFrame({"left": ids_l, "right": ids_r, "vote_left": y, "model_margin": m, "model_conf": conf,
                        "consensus_margin": cm, "category": cat,
                        "city_left": te["city_left"], "city_right": te["city_right"],
                        "lat_left": lat[tli], "lon_left": lon[tli], "lat_right": lat[tri], "lon_right": lon[tri],
                        "neigh250_left": nb[250.0][0][tli], "neigh250_right": nb[250.0][0][tri],
                        "s_left": s[tli], "s_right": s[tri], "human_wealth_margin": hw_m,
                        "image_left": te["image_left"], "image_right": te["image_right"]})
    rng = np.random.default_rng(SEED)
    errs_df = rev[rev["category"] == "model_error"].sort_values("model_conf", ascending=False)
    errs_df.to_csv(OUT / "model_errors_for_review.csv", index=False)
    hits_df = rev[rev["category"] == "hit_with_consensus"]
    hits_df.iloc[rng.choice(len(hits_df), 200, replace=False)].to_csv(OUT / "random_hits_for_review.csv", index=False)
    by_city = rev.assign(err=(rev["category"] == "model_error")).groupby("city_left")["err"].agg(["sum", "count"])
    res["city_most_model_errors"] = str(by_city["sum"].idxmax())
    make_figures(rev, cats, order, lat, lon, city, s, res["city_most_model_errors"], rng, nb[250.0][0])

    res["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log(f"done in {res['minutes']} min -> {OUT}")


def make_figures(rev, cats, order, lat, lon, city, s, cname, rng, n250):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    labels = {"hit_with_consensus": "Hit, agrees\nwith consensus", "lucky_hit": "Hit against\nconsensus",
              "vote_against_consensus": "Vote against\nconsensus", "model_error": "Model error",
              "no_clear_consensus": "No clear\nconsensus"}
    cols = {"hit_with_consensus": "#3b7d4f", "lucky_hit": "#9cc5a6", "vote_against_consensus": "#d9a441",
            "model_error": "#b8423a", "no_clear_consensus": "#a7a7a7"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    sh = [100 * cats[c]["share"] for c in order]
    ax.bar(range(5), sh, color=[cols[c] for c in order])
    for i, v in enumerate(sh):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(5), [labels[c] for c in order], fontsize=8)
    ax.set_ylabel("Share of test pairs (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "exp071_categories.pdf")
    plt.close(fig)

    # map of one city: (a) model score of every image; (b) discrepancy with the Gaussian neighbourhood (250 m),
    # with the images of model errors marked
    msk = city == cname
    sub = rev[(rev["city_left"] == cname) | (rev["city_right"] == cname)]
    q = sub[sub["category"] == "model_error"]
    pts = pd.concat([q.loc[q["city_left"] == cname, ["lon_left", "lat_left"]].set_axis(["x", "y"], axis=1),
                     q.loc[q["city_right"] == cname, ["lon_right", "lat_right"]].set_axis(["x", "y"], axis=1)])
    d = s - n250
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    lim = np.nanpercentile(np.abs(d[msk]), 95)
    for ax, val, cmap, vmin, vmax, lab in [
            (axes[0], s, "RdYlGn", np.percentile(s, 5), np.percentile(s, 95), "model safety score $s$"),
            (axes[1], d, "PuOr", -lim, lim, "$s - n$ (image minus Gaussian neighbourhood, 250 m)")]:
        sc = ax.scatter(lon[msk], lat[msk], c=val[msk], s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.scatter(pts["x"], pts["y"], s=14, marker="x", c="black", linewidths=0.6,
                   label=f"image in a model error ({len(pts)})")
        ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lat[msk]))))
        ax.set_xlabel("longitude")
        ax.legend(loc="lower left", fontsize=7, frameon=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(lab, fontsize=8)
    axes[0].set_ylabel("latitude")
    axes[0].set_title(f"(a) {cname}: predicted perceived safety", fontsize=9)
    axes[1].set_title("(b) how much each image departs from its surroundings", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "exp071_map.pdf")
    plt.close(fig)

    # gallery: 2 random hits, 2 most confident model errors (pairs from different cities avoided: same city first)
    errs = rev[rev["category"] == "model_error"].sort_values("model_conf", ascending=False)
    errs_same = errs[errs["city_left"] == errs["city_right"]].head(2)
    hits = rev[(rev["category"] == "hit_with_consensus") & (rev["city_left"] == rev["city_right"])]
    hits = hits.iloc[rng.choice(len(hits), 2, replace=False)]
    sel = [("Random hit", r) for _, r in hits.iterrows()] + [("Confident model error", r) for _, r in errs_same.iterrows()]
    fig, axes = plt.subplots(len(sel), 2, figsize=(7.0, 2.35 * len(sel)))
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i, (tag, r) in enumerate(sel):
            for j, side in enumerate(["left", "right"]):
                ax = axes[i, j]
                im = Image.open(io.BytesIO(zf.read(r[f"image_{side}"]))).convert("RGB")
                ax.imshow(im)
                ax.set_xticks([]); ax.set_yticks([])
                voted = (r["vote_left"] == 1) == (side == "left")
                model_pick = (r["model_margin"] > 0) == (side == "left")
                for sp_ in ax.spines.values():
                    sp_.set_color("#2e8b57" if voted else "#bbbbbb")
                    sp_.set_linewidth(4 if voted else 1)
                ax.set_title(f"{tag} | {r[f'city_{side}']} | s={r[f's_{side}']:+.2f}"
                             f"{' (model pick)' if model_pick else ''}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "exp071_gallery.pdf", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
