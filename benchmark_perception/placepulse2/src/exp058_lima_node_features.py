"""
EXP-058 — Tabla de atributos de nodo del grafo dual de Lima (asesoría 05/11, A9: "cantidad de robos, luego conexiones,
luego los embeddings… en el nodo"). Diseño en docs/GRAPH_DESIGN_2026-11-05.md §2.

Por segmento (nodo) de cada distrito: nº de spots, embedding DINOv2 medio de las 4 vistas REL_X (EXP-056), grado dual,
longitud, tipo de vía, puente/túnel y nº de robos 2016 (B, LV, SI; Lince sin registro). Mide además la confusión que
impide usar el crimen como etiqueta: Spearman entre nº de spots y nº de robos por segmento, y robos en segmentos con y
sin fotos. Sin etiquetas de percepción: es infraestructura + diagnóstico.

Uso:  python exp058_lima_node_features.py        Salidas: outputs/exp058/ (nodes_<distrito>.csv, node_embeddings.npz)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from exp041_lima_segment_graph import load_graph
from exp051_perception_crime_gap import CRIME_DISTRICT, snap_crimes
from exp052_connectivity_effect_size import DINO
from exp056_view_selection import load_district
from lima_views import CRIMES
from segment_graph import dual_adjacency, segments_table
from utils import OUTPUTS, log, set_seed
from visual_repr import l2n

SEED = 42
OUT = OUTPUTS / "exp058"
SNAP = OUTPUTS / "exp041_plima" / "spot_to_segment.csv"
SEL = OUTPUTS / "exp056" / "per_spot_selection.csv"


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    snap = pd.read_csv(SNAP, dtype={"spot_id": str})
    sel = pd.read_csv(SEL, dtype={"spot_id": str})
    C = pd.read_csv(CRIMES, sep=";")
    emb, diag = {}, []
    for d, folder in DINO.items():
        Gp, _ = load_graph(d)
        seg = segments_table(Gp)
        A = dual_adjacency(seg)
        spots, T = load_district(folder)
        row = {s: i for i, s in enumerate(spots)}
        sd = sel[sel["district"] == d]
        sn = snap[(snap["district"] == d) & (snap["seg_idx"] >= 0)].set_index("spot_id")["seg_idx"].astype(int)
        Z = np.zeros((len(seg), T.shape[2]), np.float32)
        n = np.zeros(len(seg), int)
        for s, views in zip(sd["spot_id"], sd["REL_X_views"]):
            if s not in row or s not in sn.index:
                continue
            idx = [int(h) // 30 for h in str(views).split("|")]
            Z[sn[s]] += T[row[s], idx].mean(0)
            n[sn[s]] += 1
        Z[n > 0] = l2n(Z[n > 0] / n[n > 0, None])
        N = pd.DataFrame({"seg_idx": np.arange(len(seg)), "segment": seg["seg"], "n_spots": n,
                          "has_images": n > 0, "dual_degree": np.asarray(A.sum(1)).ravel().astype(int),
                          "length_m": seg["length_m"].round(1), "highway": seg["highway"],
                          "bridge": seg["bridge"], "tunnel": seg["tunnel"]})
        if d in CRIME_DISTRICT:
            Cy = C[(C["DISTRITO"] == CRIME_DISTRICT[d]) & (C["YEAR"] == 2016)]
            cnt, frac = snap_crimes(seg, Gp, Cy[["longitude", "latitude"]].to_numpy())
            N["robos_2016"] = cnt
            w = N["has_images"]
            r = spearmanr(N.loc[w, "n_spots"], N.loc[w, "robos_2016"]).statistic
            mw = mannwhitneyu(N.loc[w, "robos_2016"], N.loc[~w, "robos_2016"])
            diag.append(dict(district=d, segments=len(N), with_images=int(w.sum()),
                             crimes_snapped=float(frac),
                             spearman_nspots_robos=float(r),
                             robos_mean_with_images=float(N.loc[w, "robos_2016"].mean()),
                             robos_mean_without_images=float(N.loc[~w, "robos_2016"].mean()),
                             share_with_images_and_zero_robos=float((N.loc[w, "robos_2016"] == 0).mean()),
                             share_robos_in_segments_with_images=float(N.loc[w, "robos_2016"].sum() / max(N["robos_2016"].sum(), 1)),
                             mannwhitney_p=float(mw.pvalue)))
        else:
            N["robos_2016"] = np.nan
            diag.append(dict(district=d, segments=len(N), with_images=int(N["has_images"].sum())))
        N.to_csv(OUT / f"nodes_{d}.csv", index=False)
        emb[d] = Z
        log(f"{d}: {diag[-1]}")
    np.savez_compressed(OUT / "node_embeddings.npz", **emb)
    pd.DataFrame(diag).to_csv(OUT / "crime_sampling_diagnostic.csv", index=False, float_format="%.4f")
    (OUT / "run_info.json").write_text(json.dumps({"seed": SEED, "views": "REL_X (EXP-056)", "encoder": "DINOv2 ViT-B/14 CLS"},
                                                  indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
