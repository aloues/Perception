"""
EXP-059 — ¿El contexto denso ayuda cuando la etiqueta es del LUGAR? Percepción por barrio en Santiago con Mapillary.

EXP-042/048: el contexto no mejora la predicción del voto de UNA foto (el votante solo ve esa foto). Aquí la unidad es
la celda (barrio): puntaje humano BT de la celda con rivales fijos (EXP-038), a partir de los votos de las fotos de
Place Pulse de esa celda. Probe DINOv2 entrenado en `city_holdout` (Santiago fuera de train), aplicado a las fotos de
PP y a las 65,457 imágenes Mapillary (EXP-048; sin etiqueta).

Estimadores por celda: A = media PP (las fotos juzgadas) · B = media Mapillary · C = media(z(A), z(B)) (principal) ·
C_pool = media de todas las imágenes · B_seg = media de medias por segmento (Mapillary) · D = C con el Mapillary de
OTRA celda al azar (control). Métrica: Spearman con el BT humano de la celda, bootstrap de celdas (2,000), Δ pareado.
Criterio pre-registrado: ρ(C) − ρ(A) > 0 con IC que excluye 0 y ρ(C) − ρ(D) > 0. Lados 1,000 m (principal) y 500 m.

Uso:  python exp059_neighborhood_mapillary.py        Salidas: outputs/exp059/
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp026_additive_context import load_images
from exp028_neighborhood_scale import rho
from exp038_bt_target import bt_global, cell_map
from probe_utils import load_pairs, load_split, train_scores
from utils import OUTPUTS, log, set_seed

SEED = 42
CITY = "Santiago"
OUT = OUTPUTS / "exp059"
M48 = OUTPUTS / "exp048"
MIN_VOTES = 20
MIN_MAP = 5
N_BOOT = 2000
N_SPLIT = 100


def z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].to_numpy()
    pairs = load_pairs()
    ch = load_split("city_holdout")
    st = pairs.iloc[ch["test_strict"]]
    assert CITY in set(st["city_left"]) | set(st["city_right"]), "Santiago debe estar retenida en city_holdout"
    s, info = train_scores(Z, list(ids), pairs, ch, "test_strict")
    w = info["weights"]
    lam_img = bt_global(pairs)

    sm = man["city"].to_numpy() == CITY
    lat0 = man.loc[sm, "lat"].mean()
    to_xy = lambda lon, lat: np.c_[np.asarray(lon) * 111320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110540]
    pp_xy = to_xy(man.loc[sm, "lon"], man.loc[sm, "lat"])
    pp_ids, pp_s = ids[sm], s[sm]

    E = np.load(M48 / "mapillary_dinov2.npz")
    S = pd.read_csv(M48 / "mapillary_selection.csv", dtype={"id": str}).set_index("id")
    mid = E["ids"].astype(str)
    S = S.loc[mid]
    m_s = E["cls"] @ w
    m_xy = to_xy(S["lon"], S["lat"])
    m_seg = S["seg_idx"].to_numpy()

    # votos de las fotos de Santiago (test_touch: el probe no los vio), rival con λ global fijo
    tt = pairs.iloc[ch["test_touch"]]
    pp_set = set(pp_ids)
    recs = []
    for side, other, win in (("left", "right", tt["label"].to_numpy()), ("right", "left", 1 - tt["label"].to_numpy())):
        keep = tt[side].isin(pp_set).to_numpy()
        recs.append(pd.DataFrame({"img": tt[side].to_numpy()[keep], "opp": lam_img.reindex(tt[other].to_numpy()[keep]).to_numpy(),
                                  "y": win[keep].astype(float)}))
    V = pd.concat(recs, ignore_index=True)

    rows, delta_rows = [], []
    for side_m in (1000, 500):
        cell_pp = pd.Series([f"{a}|{b}" for a, b in np.floor(pp_xy / side_m).astype(int)], index=pp_ids)
        cell_m = np.array([f"{a}|{b}" for a, b in np.floor(m_xy / side_m).astype(int)])
        v = V.assign(cell=V["img"].map(cell_pp))
        cnt = v.groupby("cell").size()
        nm = pd.Series(cell_m).value_counts()
        units = pd.Index([c for c in cnt[cnt >= MIN_VOTES].index if nm.get(c, 0) >= MIN_MAP])
        v = v[v["cell"].isin(units)].reset_index(drop=True)
        ui = units.get_indexer(v["cell"])
        hum, _ = cell_map(ui, v["opp"].to_numpy(), v["y"].to_numpy(), len(units))
        # fiabilidad split-half del puntaje humano de la celda
        rel = []
        for _ in range(N_SPLIT):
            h = rng.random(len(v)) < 0.5
            a = cell_map(ui[h], v["opp"].to_numpy()[h], v["y"].to_numpy()[h], len(units))[0]
            b = cell_map(ui[~h], v["opp"].to_numpy()[~h], v["y"].to_numpy()[~h], len(units))[0]
            r = rho(a, b)
            rel.append(2 * r / (1 + r))
        reliab = float(np.mean(rel))

        A = pd.Series(pp_s, index=cell_pp.to_numpy()).groupby(level=0).mean().reindex(units).to_numpy()
        msr = pd.Series(m_s, index=cell_m)
        B = msr.groupby(level=0).mean().reindex(units).to_numpy()
        Bseg = (pd.DataFrame({"c": cell_m, "g": m_seg, "s": m_s}).groupby(["c", "g"])["s"].mean()
                .groupby(level=0).mean().reindex(units).to_numpy())
        allc = np.r_[cell_pp.to_numpy(), cell_m]
        Cpool = pd.Series(np.r_[pp_s, m_s], index=allc).groupby(level=0).mean().reindex(units).to_numpy()
        C = (z(A) + z(B)) / 2
        perm = rng.permutation(len(units))
        D = (z(A) + z(B[perm])) / 2
        est = {"A_pp": A, "B_mapillary": B, "B_seg_balanced": Bseg, "C_pp_plus_mapillary": C,
               "C_pool": Cpool, "D_mapillary_random_cell": D}
        n_pp = cell_pp.value_counts().reindex(units).to_numpy()
        boot_ix = [rng.integers(0, len(units), len(units)) for _ in range(N_BOOT)]
        for name, x in est.items():
            r = rho(x, hum)
            bs = [rho(x[i], hum[i]) for i in boot_ix]
            rows.append(dict(side_m=side_m, estimator=name, n_cells=len(units), rho=r,
                             ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                             human_reliability=reliab, rho_disattenuated=r / np.sqrt(reliab),
                             median_pp_imgs=float(np.median(n_pp)), median_map_imgs=float(nm.reindex(units).median())))
            log(f"{side_m} m {name:24s} ρ={r:.3f} [{rows[-1]['ci_low']:.3f}, {rows[-1]['ci_high']:.3f}] (n={len(units)})")
        for a, b in (("C_pp_plus_mapillary", "A_pp"), ("B_mapillary", "A_pp"), ("C_pp_plus_mapillary", "D_mapillary_random_cell"),
                     ("C_pool", "A_pp"), ("B_seg_balanced", "B_mapillary")):
            d = [rho(est[a][i], hum[i]) - rho(est[b][i], hum[i]) for i in boot_ix]
            delta_rows.append(dict(side_m=side_m, comparison=f"{a} - {b}", delta=rho(est[a], hum) - rho(est[b], hum),
                                   ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
            log(f"   Δ {a} − {b}: {delta_rows[-1]['delta']:+.3f} [{delta_rows[-1]['ci_low']:+.3f}, {delta_rows[-1]['ci_high']:+.3f}]")
        log(f"{side_m} m: fiabilidad humana de la celda {reliab:.3f} (techo ρ ≈ {np.sqrt(max(reliab, 0)):.3f})")
    pd.DataFrame(rows).to_csv(OUT / "cell_rho.csv", index=False, float_format="%.4f")
    pd.DataFrame(delta_rows).to_csv(OUT / "cell_delta.csv", index=False, float_format="%.4f")
    (OUT / "run_info.json").write_text(json.dumps({"seed": SEED, "city": CITY, "min_votes_cell": MIN_VOTES,
                                                   "min_mapillary_cell": MIN_MAP, "probe": "DINOv2 CLS, city_holdout",
                                                   "probe_acc_test_strict": info["metrics"]["pairwise_accuracy"]},
                                                  indent=2), encoding="utf-8")




def loco_split(pairs: pd.DataFrame, city: str, seed: int = SEED) -> dict:
    """Leave-one-city-out: train/val = pares que NO tocan la ciudad (95/5 al azar); test_touch = pares que la tocan."""
    touch = ((pairs["city_left"] == city) | (pairs["city_right"] == city)).to_numpy()
    both = ((pairs["city_left"] == city) & (pairs["city_right"] == city)).to_numpy()
    rest = np.flatnonzero(~touch)
    r = np.random.default_rng(seed).permutation(rest)
    nv = len(r) // 20
    return {"train": r[nv:].tolist(), "val": r[:nv].tolist(), "test_strict": np.flatnonzero(both).tolist(),
            "test_touch": np.flatnonzero(touch).tolist()}


def cross_photo(city: str = CITY, split_mode: str = "city_holdout") -> None:
    """EXP-059b / EXP-062: objetivo con los votos de OTRAS fotos de la celda (mitad B), predictor con la mitad A + Mapillary."""
    set_seed(SEED)
    rng = np.random.default_rng(SEED + 1)
    man, Z = load_images("dinov2_vit_b14")
    ids = man["location_id"].to_numpy()
    pairs = load_pairs()
    ch = load_split("city_holdout") if split_mode == "city_holdout" else loco_split(pairs, city)
    if split_mode == "city_holdout":
        st = pairs.iloc[ch["test_strict"]]
        assert city in set(st["city_left"]) | set(st["city_right"]), f"{city} no está retenida en city_holdout: usar --split loco"
    m48 = M48 if city == "Santiago" else OUTPUTS / f"exp062_{city.lower()}"
    s, info = train_scores(Z, list(ids), pairs, ch, "test_strict")
    lam_img = bt_global(pairs)
    sm = man["city"].to_numpy() == city
    lat0 = man.loc[sm, "lat"].mean()
    to_xy = lambda lon, lat: np.c_[np.asarray(lon) * 111320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110540]
    pp_ids, pp_s = ids[sm], s[sm]
    pp_xy = to_xy(man.loc[sm, "lon"], man.loc[sm, "lat"])
    E = np.load(m48 / "mapillary_dinov2.npz")
    S = pd.read_csv(m48 / "mapillary_selection.csv", dtype={"id": str}).set_index("id").loc[E["ids"].astype(str)]
    m_s = E["cls"] @ info["weights"]
    m_xy = to_xy(S["lon"], S["lat"])
    tt = pairs.iloc[ch["test_touch"]]
    pp_set = set(pp_ids)
    recs = []
    for side, other, win in (("left", "right", tt["label"].to_numpy()), ("right", "left", 1 - tt["label"].to_numpy())):
        keep = tt[side].isin(pp_set).to_numpy()
        recs.append(pd.DataFrame({"img": tt[side].to_numpy()[keep], "opp": lam_img.reindex(tt[other].to_numpy()[keep]).to_numpy(),
                                  "y": win[keep].astype(float)}))
    V = pd.concat(recs, ignore_index=True)
    score_of = pd.Series(pp_s, index=pp_ids)
    rows, N_PART, NB = [], 20, 1000
    for side_m in (1000, 500):
        cell_pp = pd.Series([f"{a}|{b}" for a, b in np.floor(pp_xy / side_m).astype(int)], index=pp_ids)
        cell_m = np.array([f"{a}|{b}" for a, b in np.floor(m_xy / side_m).astype(int)])
        nm = pd.Series(cell_m).value_counts()
        Bm_all = pd.Series(m_s, index=cell_m).groupby(level=0).mean()
        npc = cell_pp.value_counts()
        cand = pd.Index([c for c in npc[npc >= 2].index if nm.get(c, 0) >= MIN_MAP])
        parts = []
        for _ in range(N_PART):
            half = {}
            for c, g in cell_pp[cell_pp.isin(cand)].groupby(cell_pp[cell_pp.isin(cand)]):
                imgs = rng.permutation(g.index.to_numpy())
                k = len(imgs) // 2
                half.update({i: "A" for i in imgs[:k]}); half.update({i: "B" for i in imgs[k:]})
            vb = V[V["img"].map(half) == "B"].assign(cell=lambda x: x["img"].map(cell_pp))
            cnt = vb.groupby("cell").size()
            units = pd.Index(sorted(c for c in cand if cnt.get(c, 0) >= 10))
            vb = vb[vb["cell"].isin(units)]
            hum, _ = cell_map(units.get_indexer(vb["cell"]), vb["opp"].to_numpy(), vb["y"].to_numpy(), len(units))
            a_imgs = [i for i, h in half.items() if h == "A"]
            Aq = score_of.loc[a_imgs].groupby(cell_pp.loc[a_imgs].to_numpy()).mean().reindex(units).to_numpy()
            Bq = Bm_all.reindex(units).to_numpy()
            parts.append(pd.DataFrame({"cell": units, "hum": hum, "A": Aq, "B": Bq, "Brand": Bq[rng.permutation(len(units))]}))
        allcells = pd.Index(sorted(set().union(*[set(p["cell"]) for p in parts])))
        def stat(cells_sel):
            out = {k: [] for k in ("A", "B", "C", "D")}
            for p in parts:
                q = p.set_index("cell").reindex(cells_sel).dropna()
                out["A"].append(rho(q["A"], q["hum"])); out["B"].append(rho(q["B"], q["hum"]))
                out["C"].append(rho((z(q["A"]) + z(q["B"])) / 2, q["hum"])); out["D"].append(rho((z(q["A"]) + z(q["Brand"])) / 2, q["hum"]))
            return {k: float(np.mean(v)) for k, v in out.items()}
        base = stat(allcells)
        boots = [stat(allcells[rng.integers(0, len(allcells), len(allcells))]) for _ in range(NB)]
        for k in ("A", "B", "C", "D"):
            b = [x[k] for x in boots]
            rows.append(dict(side_m=side_m, quantity=f"rho_{k}", value=base[k], ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5)))
        for a, b2 in (("C", "A"), ("C", "D"), ("B", "A")):
            d = [x[a] - x[b2] for x in boots]
            rows.append(dict(side_m=side_m, quantity=f"delta_{a}-{b2}", value=base[a] - base[b2], ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
        log(f"{side_m} m (≈{len(allcells)} celdas, {N_PART} particiones): " + ", ".join(f"{r['quantity']}={r['value']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" for r in rows if r["side_m"] == side_m))
    tag = "" if (city == "Santiago" and split_mode == "city_holdout") else f"_{city.lower()}_{split_mode}"
    pd.DataFrame(rows).to_csv(OUT / f"cross_photo{tag}.csv", index=False, float_format="%.4f")
    log(f"probe {split_mode}: exactitud en pares que tocan {city} = {info['metrics']['pairwise_accuracy']:.4f}" if split_mode == "loco" else "")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--city", default=CITY)
    ap.add_argument("--split", choices=["city_holdout", "loco"], default="city_holdout")
    a = ap.parse_args()
    cross_photo(a.city, a.split) if a.cross else main()
