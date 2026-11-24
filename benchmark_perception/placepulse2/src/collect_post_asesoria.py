"""
Fuente ÚNICA de cifras post-asesoría 12/11 para EC1, PPT, póster y guía (coherencia R02). Lee SOLO outputs reales.

Salidas (deliverables/numbers/):
  post_asesoria_numbers.json   todo lo medido, con fuente por bloque y estado (measured / in_progress / missing)
  tab_exp063_heads.tex         tabla LaTeX (booktabs) de cabezas, image_disjoint (+ city_holdout si existe)
  tab_exp064_encoders.tex      tabla LaTeX de encoders con la misma cabeza lineal
  tab_exp063_place.tex         tabla LaTeX de Track D (lugar), si existe
  tab_exp061_finetune.tex      tabla LaTeX de EXP-061 (fine-tuning vs congelado), si existe
Uso: python src/collect_post_asesoria.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, REPO_ROOT, RUNS  # noqa: E402

DST = REPO_ROOT / "deliverables" / "numbers"
NAMES = {"linear_bt": "Linear BT (baseline)", "mlp_bt": "MLP BT", "ridge": "Ridge $\\to\\lambda$",
         "knn": "KNN (cosine) $\\to\\lambda$", "svr_rbf": "SVR-RBF$^\\dagger$ $\\to\\lambda$",
         "rf": "Random forest $\\to\\lambda$", "xgb": "XGBoost $\\to\\lambda$", "mlp_reg": "MLP $\\to\\lambda$",
         "logreg_clf": "Logistic (quartiles)", "linsvm_clf": "Linear SVM (quartiles)",
         "svm_rbf_clf": "RBF-SVM$^\\dagger$ (quartiles)", "knn_clf": "KNN (quartiles)",
         "rf_clf": "Random forest (quartiles)", "xgb_clf": "XGBoost (quartiles)", "mlp_clf": "MLP (quartiles)"}
ENC_NAMES = {"dinov2_vit_b14": "DINOv2 ViT-B/14 (CLS)", "dinov3_vitb16_cls": "DINOv3 ViT-B/16 (CLS)",
             "dinov3_vitb16_meanpatch": "DINOv3 ViT-B/16 (mean patch)",
             "dinov2_vitl14_cls": "DINOv2 ViT-L/14 (CLS)", "dinov2_vitl14_meanpatch": "DINOv2 ViT-L/14 (mean patch)",
             "ijepa_vith14_meanpatch": "I-JEPA ViT-H/14 IN-1k (mean patch)",
             "ijepa_vith14_in22k_meanpatch": "I-JEPA ViT-H/14 IN-22k (mean patch)",
             "vjepa2_vitl16_meanpatch": "V-JEPA 2 ViT-L/16 (mean patch)", "imagenet_vit_b16": "ImageNet ViT-B/16",
             "ours_simclr_vit_b16": "SimCLR ViT-B/16 (inherited)",
             "lejepa_resnet18_pilot": "LeJEPA ResNet-18 pilot (in-domain)",
             "resnet18_random_init": "ResNet-18 random init (control)"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pts(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.2f}"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    out: dict = {"generated_from": "outputs/ (benchmark_perception/placepulse2)", "blocks": {}}

    # ---------------- EXP-063
    sh = OUTPUTS / "exp063" / "summary_heads.csv"
    if sh.exists():
        df = pd.read_csv(sh)
        df = df[df.encoder == "dinov2_vit_b14"]
        out["blocks"]["exp063"] = {"status": "measured", "source": "outputs/exp063/summary_heads.csv",
                                   "rows": json.loads(df.to_json(orient="records"))}
        # seeds: media ± sd por cabeza (image_disjoint, z)
        sd = df[(df.protocol == "image_disjoint") & (df.rep == "z")].groupby("head")["test_acc"].agg(["mean", "std", "count"])
        out["blocks"]["exp063"]["seed_summary_image_disjoint"] = json.loads(sd.reset_index().to_json(orient="records"))
        lines = [r"\begin{tabular}{llcccc}", r"\toprule",
                 r"Track & Head & \multicolumn{2}{c}{\texttt{image\_disjoint}} & \multicolumn{2}{c}{\texttt{city\_holdout}} \\",
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
                 r" & & Acc. [95\% CI] & $\Delta$ (pts) [95\% CI] & Acc. [95\% CI] & $\Delta$ (pts) [95\% CI] \\", r"\midrule"]
        base = df[(df.rep == "z") & (df.seed == 42)]
        heads = [h for h in NAMES if h in set(base["head"])]
        for h in heads:
            cells = []
            for prot in ("image_disjoint", "city_holdout"):
                r = base[(base.protocol == prot) & (base["head"] == h)]
                if r.empty:
                    cells += ["--", "--"]
                    continue
                r = r.iloc[0]
                cells.append(f"{r.test_acc:.3f} [{r.test_ci_img_lo:.3f}, {r.test_ci_img_hi:.3f}]")
                cells.append("ref." if h == "linear_bt" else
                             f"{pts(r.delta_vs_linear)} [{pts(r.delta_img_lo)}, {pts(r.delta_img_hi)}]")
            tr = base[base["head"] == h].track.iloc[0]
            lines.append(f"{tr} & {NAMES[h]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp063_heads.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- EXP-063 Track D
    pl = OUTPUTS / "exp063" / "place_cells_city_holdout.csv"
    if pl.exists():
        p = pd.read_csv(pl)
        out["blocks"]["exp063_place"] = {"status": "measured", "source": str(pl.relative_to(OUTPUTS.parent)),
                                         "rows": json.loads(p.to_json(orient="records"))}
        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 r"Head & Within-city $\rho$ [95\% CI] & $\Delta\rho$ vs linear [95\% CI] \\", r"\midrule"]
        for r in p.itertuples():
            d = "ref." if r.head == "linear_bt" else f"{r.delta_vs_linear:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
            lines.append(f"{NAMES.get(r.head, r.head)} & {r.rho_intra_city:.3f} [{r.rho_lo:.3f}, {r.rho_hi:.3f}] & {d} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp063_place.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- EXP-064 (+ EXP-065 probes) — misma cabeza lineal
    pv = OUTPUTS / "exp064" / "paired_vs_dinov2.csv"
    if pv.exists():
        e = pd.read_csv(pv)
        out["blocks"]["exp064"] = {"status": "measured", "source": "outputs/exp064/paired_vs_dinov2.csv",
                                   "rows": json.loads(e.to_json(orient="records"))}
        prots = [p for p in ("image_disjoint", "city_holdout", "paper_protocol") if p in set(e.protocol)]
        lines = [r"\begin{tabular}{l" + "cc" * len(prots) + "}", r"\toprule",
                 "Encoder (frozen) & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\texttt{{{p.replace('_', chr(92) + '_')}}}}}"
                                                    for p in prots) + r" \\",
                 " & " + " & ".join(["Acc. & $\\Delta$ vs DINOv2 (pts) [95\\% CI]"] * len(prots)) + r" \\", r"\midrule"]
        for enc in [x for x in ENC_NAMES if x in set(e.encoder)]:
            cells = []
            for p in prots:
                r = e[(e.protocol == p) & (e.encoder == enc)]
                if r.empty:
                    cells += ["--", "--"]
                    continue
                r = r.iloc[0]
                cells.append(f"{r.acc:.3f}")
                cells.append("ref." if enc == "dinov2_vit_b14" else
                             f"{pts(r.delta_vs_dinov2)} [{pts(r.delta_img_lo)}, {pts(r.delta_img_hi)}]")
            lines.append(ENC_NAMES[enc] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp064_encoders.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- EXP-061 — fine-tuning parcial (K = 2) vs congelado, mismos pares
    pf = OUTPUTS / "exp061" / "paired_vs_frozen.csv"
    if pf.exists():
        f61 = pd.read_csv(pf)
        f61 = f61.sort_values("protocol", key=lambda c: c.map({"image_disjoint": 0, "city_holdout": 1, "paper_protocol": 2}))
        out["blocks"]["exp061"] = {"status": "measured", "source": "outputs/exp061/paired_vs_frozen.csv",
                                   "rows": json.loads(f61.to_json(orient="records"))}
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Protocol (test pairs) & Frozen + linear BT & Fine-tuned $K{=}2$ [95\% CI] & $\Delta$ (pts) [95\% CI] & Log-loss (FT / frozen) \\",
                 r"\midrule"]
        for r in f61.itertuples():
            prot = r.protocol.replace("_", "\_")
            lines.append(r"\texttt{" + prot + "}" + f" ({r.n:,}) & {r.acc_frozen:.4f} & "
                         f"{r.acc_finetune:.4f} [{r.acc_ft_img_lo:.3f}, {r.acc_ft_img_hi:.3f}] & "
                         f"{pts(r.delta)} [{pts(r.delta_img_lo)}, {pts(r.delta_img_hi)}] & "
                         f"{r.logloss_finetune:.3f} / {r.logloss_frozen:.3f} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp061_finetune.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- EXP-066 — techo de ruido del voto (modelo latente BT)
    s66 = OUTPUTS / "exp066" / "summary.json"
    if s66.exists():
        j = json.loads(s66.read_text(encoding="utf-8"))
        out["blocks"]["exp066"] = {"status": "measured", "source": "outputs/exp066/summary.json", **j}
        main_v = j["links"]["logit_gauss"]
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Protocol (test pairs) & DINOv2 + linear & Ceiling $C$ [95\% CI] & Share of range & "
                 r"Oracle (obs.\ / sim.) \\",
                 r"\midrule"]
        for prot in ("image_disjoint", "city_holdout", "paper_protocol"):
            r = main_v["protocols"][prot]
            lines.append(r"\texttt{" + prot.replace("_", r"\_") + "}" + f" ({r['n_test_pairs']:,}) & "
                         f"{r['dinov2_accuracy']:.3f} & {r['ceiling']:.3f} [{r['ceiling_ci95'][0]:.3f}, "
                         f"{r['ceiling_ci95'][1]:.3f}] & {100 * r['efficiency_above_chance']:.0f}" + r"\% & "
                         f"{r['oracle_in_sample_observed']:.3f} / {r['oracle_in_sample_sim_mean']:.3f} " + r"\\")
        lines.append(r"\midrule")
        lab = {"logit_gauss": "logistic link, Gaussian $\\varepsilon$",
               "probit_gauss": "probit link (Thurstone V), Gaussian $\\varepsilon$",
               "logit_laplace": "logistic link, Laplace $\\varepsilon$"}
        for v, res in j["links"].items():
            r = res["protocols"]["image_disjoint"]
            lines.append(r"\multicolumn{5}{l}{\footnotesize " + lab.get(v, v) +
                         f": $\\hat\\tau={res['tau_hat']:.2f}$ [{res['tau_ci95'][0]:.2f}, {res['tau_ci95'][1]:.2f}], "
                         f"ceiling on new images {r['ceiling']:.3f}, DINOv2 reaches "
                         f"{100 * r['efficiency_above_chance']:.0f}" + r"\% of the range} \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp066_ceiling.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- EXP-067 — contrastes pre-registrados (datos, JEPA vigente, escala)
    c67 = OUTPUTS / "exp067" / "contrasts.csv"
    if c67.exists():
        c = pd.read_csv(c67)
        out["blocks"]["exp067"] = {"status": "measured", "source": "outputs/exp067/contrasts.csv",
                                   "rows": json.loads(c.to_json(orient="records"))}

    # ---------------- EXP-065
    e65 = {"status": "smoke_test"}
    for name in ("sanity.json", "pilot/train_meta.json", "pilot/history.json"):
        f = OUTPUTS / "exp065" / name
        if f.exists():
            e65[name] = json.loads(f.read_text(encoding="utf-8"))
    for enc in ("lejepa_resnet18_pilot", "resnet18_random_init"):
        for prot, key in (("image_disjoint", "test"), ("city_holdout", "test_strict")):
            m = RUNS / f"{enc}__{prot}__linear__seed42" / "metrics.json"
            if m.exists():
                t = json.loads(m.read_text())["test"][key]
                e65.setdefault("probes", {})[f"{enc}/{prot}"] = {k: t[k] for k in ("pairwise_accuracy", "ci95_low",
                                                                                     "ci95_high", "n")}
    if "pilot/train_meta.json" in e65:
        e65["status"] = "pilot_trained" + ("_and_probed" if "probes" in e65 else "")
    out["blocks"]["exp065"] = e65

    # ---------------- ejemplo trabajado y baselines
    we = OUTPUTS / "exp063" / "worked_example.json"
    if we.exists():
        out["blocks"]["worked_example"] = json.loads(we.read_text(encoding="utf-8"))
    base = {}
    for enc in ("dinov2_vit_b14", "imagenet_vit_b16", "ours_simclr_vit_b16"):
        for prot, key in (("image_disjoint", "test"), ("city_holdout", "test_strict"), ("paper_protocol", "test")):
            m = RUNS / f"{enc}__{prot}__linear__seed42" / "metrics.json"
            if m.exists():
                t = json.loads(m.read_text())["test"][key]
                base[f"{enc}/{prot}"] = {k: t[k] for k in ("pairwise_accuracy", "ci95_low", "ci95_high", "n")}
    out["blocks"]["rq1_baselines"] = {"status": "measured", "source": "outputs/runs/*/metrics.json", "rows": base}
    (DST / "post_asesoria_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("bloques:", {k: v.get("status") if isinstance(v, dict) else "" for k, v in out["blocks"].items()})


if __name__ == "__main__":
    main()
