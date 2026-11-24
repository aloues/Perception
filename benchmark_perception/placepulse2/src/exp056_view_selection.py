"""
EXP-056 — Selección sistemática de vistas por ángulo RELATIVO a la calle (asesoría 05/11). Diseño en
docs/EXPERIMENT_REGISTRY.md (pre-registrado antes de ejecutar).

Pregunta del asesor: con 0/90/180/270 absolutos, en una calle norte–sur dos vistas miran a lo largo de la calle y
repiten pista y autos. ¿Es mejor una X (diagonales), solo las fachadas (izquierda/derecha) o las 4 vistas más
distintas entre sí? Aquí se mide SIN etiquetas humanas: fidelidad, redundancia, re-identificación del lugar y
percepción PREDICHA (probe DINOv2 de Place Pulse). Cuál predice mejor el juicio humano en Lima lo dirá la encuesta.

Uso:  python exp056_view_selection.py        Salidas: outputs/exp056/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp041_lima_segment_graph import load_graph
from exp051_perception_crime_gap import pp_probe
from exp052_connectivity_effect_size import DINO
from lima_views import PLIMA_EXPORTS
from segment_graph import segments_table
from utils import OUTPUTS, environment_info, log, set_seed
from visual_repr import l2n

SEED = 42
OUT = OUTPUTS / "exp056"
SNAP = OUTPUTS / "exp041_plima" / "spot_to_segment.csv"
HEADINGS = np.arange(0, 360, 30)
N_BOOT = 1000
N_RAND = 10


def load_district(folder: str) -> tuple[list[str], np.ndarray]:
    """Spots con las 12 vistas → (spot_ids, tensor [n_spots, 12, d] ordenado por heading)."""
    z = np.load(PLIMA_EXPORTS / folder / "dinov2.npz")["cls"]
    paths = json.loads((PLIMA_EXPORTS / folder / "dinov2_index.json").read_text(encoding="utf-8"))["paths"]
    P = pd.DataFrame({"spot": [p.split("/")[1] for p in paths],
                      "h": [int(p.split("heading_")[1].split(".")[0]) for p in paths], "row": range(len(paths))})
    P = P[P.groupby("spot")["h"].transform("nunique") == 12]
    spots = sorted(P["spot"].unique())
    T = np.zeros((len(spots), 12, z.shape[1]), np.float32)
    si = {s: i for i, s in enumerate(spots)}
    T[P["spot"].map(si).to_numpy(), (P["h"] // 30).to_numpy()] = z[P["row"].to_numpy()]
    return spots, l2n(T)


def street_bearing(seg: pd.DataFrame, seg_idx: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Rumbo (grados desde el norte, sentido horario) de la tangente local del segmento, en [0, 180)."""
    out = np.full(len(seg_idx), np.nan)
    for n, (k, (x, y)) in enumerate(zip(seg_idx, xy)):
        if k < 0:
            continue
        from shapely.geometry import Point
        g = seg["geometry"].iloc[int(k)]
        d = g.project(Point(x, y))
        a, b = g.interpolate(max(d - 3, 0)), g.interpolate(min(d + 3, g.length))
        out[n] = np.degrees(np.arctan2(b.x - a.x, b.y - a.y)) % 180
    return out


def nearest_idx(angle_deg: np.ndarray) -> np.ndarray:
    return np.round((np.asarray(angle_deg) % 360) / 30).astype(int) % 12


def maxdiv(T: np.ndarray, k: int = 4) -> np.ndarray:
    """Punto más lejano en coseno: empieza por la vista menos parecida a la media, añade la más distinta al grupo."""
    sel = np.zeros((len(T), k), int)
    for i, V in enumerate(T):
        S = V @ V.T
        first = int(np.argmin(V @ l2n(V.mean(0))))
        chosen = [first]
        while len(chosen) < k:
            cand = [j for j in range(12) if j not in chosen]
            chosen.append(cand[int(np.argmin(S[cand][:, chosen].max(1)))])
        sel[i] = chosen
    return sel


def strategies(T: np.ndarray, bearing: np.ndarray, rng) -> dict[str, np.ndarray]:
    n = len(T)
    b = np.nan_to_num(bearing, nan=0.0)[:, None]
    S = {"ALL12": np.tile(np.arange(12), (n, 1)),
         "ABS4": np.tile(nearest_idx([0, 90, 180, 270]), (n, 1)),
         "REL_CROSS": nearest_idx(b + np.array([0, 90, 180, 270])),
         "REL_X": nearest_idx(b + np.array([45, 135, 225, 315])),
         "REL_PERP2": nearest_idx(b + np.array([90, 270])),
         "REL_FACADE4_posthoc": nearest_idx(b + np.array([60, 120, 240, 300])),   # POST-HOC: 4 sin el eje de la calle
         "REL_ALONG2": nearest_idx(b + np.array([0, 180])),
         "MAXDIV4": maxdiv(T)}
    for r in range(N_RAND):
        S[f"RAND4_{r}"] = np.stack([rng.choice(12, 4, replace=False) for _ in range(n)])
    return S


def metrics(T: np.ndarray, sel: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    """Métricas por spot para una selección [n, k] de índices de vista."""
    n = len(T)
    ridx = np.arange(n)[:, None]
    Vsel = T[ridx, sel]                                         # n, k, d
    rep = l2n(Vsel.mean(1))
    full = l2n(T.mean(1))
    k = sel.shape[1]
    G = np.einsum("nkd,njd->nkj", Vsel, Vsel)
    redund = (G.sum((1, 2)) - k) / (k * (k - 1))
    out = {"fidelity": (rep * full).sum(1), "redundancy": redund,
           "score": T.mean(1) @ w if k == 12 else Vsel.mean(1) @ w}
    if k < 12:                                                  # re-ID con las vistas NO elegidas
        mask = np.ones((n, 12), bool)
        mask[ridx, sel] = False
        rest = l2n((T * mask[..., None]).sum(1) / mask.sum(1, keepdims=True))
        sim = rep @ rest.T
        out["reid_top1"] = (sim.argmax(1) == np.arange(n)).astype(float)
        out["reid_rank"] = (sim > sim[np.arange(n), np.arange(n)][:, None]).sum(1).astype(float)
    return out


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    d = a - b
    bs = [d[i].mean() for i in (rng.integers(0, len(d), len(d)) for _ in range(N_BOOT))]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    w, _, acc = pp_probe("dinov2_vit_b14")
    log(f"probe DINOv2 PP: exactitud test image_disjoint {acc:.4f}")
    snap = pd.read_csv(SNAP, dtype={"spot_id": str})
    rows, diffs, per_spot = [], [], []
    import geopandas as gpd
    from exp041_lima_segment_graph import lima_spots
    S_all, *_ = lima_spots(plima=True)                          # coordenadas del spot (lat/lon) y filtro M28
    for d, folder in DINO.items():
        spots, T = load_district(folder)
        Gp, _ = load_graph(d)
        seg = segments_table(Gp)
        sn = snap[snap["district"] == d].copy()
        sn["key"] = sn["spot_id"].astype(str)
        sn = sn.set_index("key")
        Sd = S_all[S_all["district"] == d].set_index("spot_id")
        keep = [s for s in spots if s in Sd.index and s in sn.index]
        idx = [spots.index(s) for s in keep]
        T = T[idx]
        pts = gpd.GeoSeries(gpd.points_from_xy(Sd.loc[keep, "lon"], Sd.loc[keep, "lat"]), crs=4326).to_crs(Gp.graph["crs"])
        xy = np.c_[pts.x, pts.y]
        seg_idx = sn.loc[keep, "seg_idx"].fillna(-1).astype(int).to_numpy()
        bearing = street_bearing(seg, seg_idx, xy)
        has_b = np.isfinite(bearing)
        T, bearing, keep = T[has_b], bearing[has_b], [s for s, h in zip(keep, has_b) if h]
        log(f"{d}: {len(keep):,} spots con 12 vistas y rumbo de calle (de {len(spots):,})")
        S = strategies(T, bearing, rng)
        M = {name: metrics(T, sel, w) for name, sel in S.items()}
        sc12 = M["ALL12"]["score"]
        for name, m in M.items():
            if name.startswith("RAND4_") and name != "RAND4_0":
                continue
            label = "RAND4" if name == "RAND4_0" else name
            r = dict(district=d, strategy=label, n_spots=len(T), views=S[name].shape[1],
                     fidelity=float(m["fidelity"].mean()), redundancy=float(m["redundancy"].mean()),
                     score_spearman_vs12=float(spearmanr(m["score"], sc12).statistic),
                     score_sd_between_spots=float(m["score"].std()))
            if "reid_top1" in m:
                r["reid_top1"] = float(m["reid_top1"].mean())
                r["reid_median_rank"] = float(np.median(m["reid_rank"]))
            if label == "RAND4":                                 # media y dispersión entre los 10 sorteos
                R = [M[f"RAND4_{j}"] for j in range(N_RAND)]
                for key in ("fidelity", "redundancy", "reid_top1"):
                    vals = [x[key].mean() for x in R]
                    r[key], r[f"{key}_sd_draws"] = float(np.mean(vals)), float(np.std(vals))
                sc = np.stack([x["score"] for x in R])
                r["score_within_spot_sd_over_between"] = float(sc.std(0).mean() / sc.mean(0).std())
            rows.append(r)
        for name in ("REL_CROSS", "REL_X", "REL_PERP2", "REL_FACADE4_posthoc", "REL_ALONG2", "MAXDIV4", "RAND4_0"):
            for key in ("fidelity", "redundancy", "reid_top1"):
                a = np.mean([M[f"RAND4_{j}"][key] for j in range(N_RAND)], 0) if name == "RAND4_0" else M[name][key]
                mu, lo, hi = boot_diff(a, M["ABS4"][key], rng)
                diffs.append(dict(district=d, strategy="RAND4" if name == "RAND4_0" else name, metric=key,
                                  delta_vs_ABS4=mu, ci_low=lo, ci_high=hi))
        per_spot.append(pd.DataFrame({"district": d, "spot_id": keep, "street_bearing": bearing,
                                      **{f"{n}_views": ["|".join(map(str, HEADINGS[s])) for s in S[n]]
                                         for n in ("ABS4", "REL_CROSS", "REL_X", "REL_PERP2", "MAXDIV4")}}))
    pd.DataFrame(rows).to_csv(OUT / "view_selection_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(diffs).to_csv(OUT / "delta_vs_abs4.csv", index=False, float_format="%.4f")
    pd.concat(per_spot).to_csv(OUT / "per_spot_selection.csv", index=False, float_format="%.1f")
    (OUT / "run_info.json").write_text(json.dumps({"seed": SEED, "n_boot": N_BOOT, "n_rand": N_RAND,
                                                   "probe_acc_pp": acc, "minutes": (time.perf_counter() - t0) / 60,
                                                   "env": environment_info()}, indent=2, default=str), encoding="utf-8")
    log(f"listo en {(time.perf_counter() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
