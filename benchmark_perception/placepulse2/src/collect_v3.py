"""
Fuente ÚNICA de cifras del programa V3 (EXP-067 completo, EXP-068, EXP-069, EXP-070) para EC1, PPT, póster y guía.
Lee SOLO outputs reales. Complementa collect_post_asesoria.py (que sigue generando EXP-061/063/064/066).

Salidas (deliverables/numbers/):
  v3_numbers.json            cifras con ruta de origen
  tab_exp068_reliability.tex tabla LaTeX (empates, calibración, abstención, votantes)
  tab_exp069_construct.tex   tabla LaTeX (validez de constructo)
  tab_exp070_budget.tex      tabla LaTeX (votos locales)
Uso: python src/collect_v3.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, REPO_ROOT  # noqa: E402

DST = REPO_ROOT / "deliverables" / "numbers"
FIG = REPO_ROOT / "deliverables" / "figures" / "v3"
EN = {"wealthy": "wealthier", "beautiful": "more beautiful", "lively": "livelier", "boring": "more boring",
      "depressing": "more depressing"}


def rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return f"{x:.3f}"


def pts(x):
    return f"{100 * x:+.2f}"


def ci_pts(lo, hi):
    return f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    out = {"generated_from": "benchmark_perception/placepulse2/outputs/exp06{7,8,9}, exp070", "blocks": {}}

    # EXP-067 (contrastes)
    c = OUTPUTS / "exp067" / "contrasts.csv"
    if c.exists():
        out["blocks"]["exp067"] = {"source": str(c.relative_to(REPO_ROOT)),
                                   "rows": pd.read_csv(c).to_dict(orient="records")}
    bh = OUTPUTS / "exp067" / "best_head.csv"
    if bh.exists():
        out["blocks"]["exp067_best_head"] = {"source": str(bh.relative_to(REPO_ROOT)),
                                             "rows": pd.read_csv(bh).to_dict(orient="records")}

    # EXP-068
    r = rd(OUTPUTS / "exp068" / "summary.json")
    if r:
        out["blocks"]["exp068"] = {"source": "benchmark_perception/placepulse2/outputs/exp068/summary.json", **r}
        rows = []
        for p, lab in (("image_disjoint", "New images"), ("city_holdout", "New cities")):
            R = r[p]
            t, ab, ca, vd = R["ties"], R["abstention"], R["calibration"], R["voter_dispersion"]
            rows.append((lab, R["n_test_decisive"], t["n_test_ties"],
                         f"{f3(t['auc_minus_margin_predicts_tie'][0])} [{f3(t['auc_minus_margin_predicts_tie'][1])}, {f3(t['auc_minus_margin_predicts_tie'][2])}]",
                         f"{t['voter_control']['adjusted_logit_OR_per_SD_margin'][0]:.2f} [{t['voter_control']['adjusted_logit_OR_per_SD_margin'][1]:.2f}, {t['voter_control']['adjusted_logit_OR_per_SD_margin'][2]:.2f}]",
                         f"{ca['raw']['ece15']:.3f}", f"{ca['temperature_val']:.2f}",
                         f"{f3(ab['test_acc_at_tau'][0])} [{f3(ab['test_acc_at_tau'][1])}, {f3(ab['test_acc_at_tau'][2])}]",
                         f"{100 * ab['test_coverage_at_tau'][0]:.1f}\\,\\%",
                         f"{vd['phi']:.2f} [{vd['phi_ci_voter_boot'][0]:.2f}, {vd['phi_ci_voter_boot'][1]:.2f}]",
                         f"{vd['null_phi_95']:.2f}", vd["n_voters"]))
        a, b = rows
        labels = ["Decisive test votes", "Tied test votes (unused in training)",
                  r"AUROC of $-|s_i-s_j|$ for a tie",
                  r"Tie odds ratio per SD of margin, voter-adj.",
                  "Expected calibration error (15 equal-mass bins)", "Temperature fitted on validation",
                  r"Accuracy at the validation threshold", "Coverage at that threshold",
                  r"Voter dispersion $\varphi$", r"Voters with $\ge 10$ test votes",
                  r"$\varphi$ under one observer (95th pct.)"]
        lines = [r"\begin{tabular}{lcc}", r"\toprule", r" & New images & New cities \\", r"\midrule"]
        for k, lab in enumerate(labels):
            idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10][k]
            va, vb = a[idx], b[idx]
            if isinstance(va, int):
                va, vb = f"{va:,}", f"{vb:,}"
            lines.append(f"{lab} & {va} & {vb} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp068_reliability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        d = r["image_disjoint"]["davidson"]
        out["blocks"]["exp068_davidson_text"] = (f"Davidson: beta {d['beta']:.3f}, nu {d['nu']:.3f}; log-loss3 "
                                                 f"{d['logloss3_davidson']:.4f} vs {d['logloss3_const']:.4f}")
        for f in (OUTPUTS / "exp068" / "figures").glob("*.png"):
            shutil.copy2(f, FIG / f"exp068_{f.name}")

    # EXP-069
    r = rd(OUTPUTS / "exp069" / "summary.json")
    if r:
        out["blocks"]["exp069"] = {"source": "benchmark_perception/placepulse2/outputs/exp069/summary.json", **r}
        hc = {x["attr"]: x for x in r["human_control"]["table"]}
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Head trained on & Own acc. & Acc. on \emph{safer} & $\Delta$ vs safe (pts) [95\% CI] & $r$ model & $\rho$ votes \\",
                 r"\midrule",
                 f"\\emph{{safer}} (reference) & {f3(r['safe_acc_test'])} & {f3(r['safe_acc_test'])} & ref. & 1 & 1 \\\\"]
        for x in r["cross_attribute"]:
            sign = " ($-$)" if x["sign"] < 0 else ""
            lines.append(f"\\emph{{{EN[x['attr']]}}}{sign} & {f3(x['own_attr_test_acc'])} & {f3(x['acc_on_safe_test'])} & "
                         f"{pts(x['delta_vs_safe_head'])} {ci_pts(x['dlo'], x['dhi'])} & "
                         f"{x['pearson_score_test_images_vs_safe']:+.2f} & {hc[x['attr']]['rho_disattenuated']:+.2f} \\\\")
        cb, res = r["combination"], r["residual"]["score_space_residual"]
        lines += [r"\midrule",
                  f"Combination of the five (fit on safe train) & -- & {f3(cb['acc'])} & {pts(cb['delta_vs_safe_head'][0])} {ci_pts(cb['delta_vs_safe_head'][1], cb['delta_vs_safe_head'][2])} & -- & -- \\\\",
                  f"Safe score minus its projection on the five & -- & {f3(res['acc'])} [{f3(res['ci'][0])}, {f3(res['ci'][1])}] & -- & -- & -- \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp069_construct.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for f in (OUTPUTS / "exp069" / "figures").glob("*.png"):
            shutil.copy2(f, FIG / f"exp069_{f.name}")

    # EXP-070
    r = rd(OUTPUTS / "exp070" / "summary.json")
    if r:
        out["blocks"]["exp070"] = {"source": "benchmark_perception/placepulse2/outputs/exp070/summary.json", **r}
        S = pd.DataFrame(r["summary"])
        AR = {int(x["n"]): x for x in r["active_minus_random"]}
        lines = [r"\begin{tabular}{rccc}", r"\toprule",
                 r"Local votes $n$ & Random: $\Delta$ (pts), mean $\pm$ SD & Active: $\Delta$ (pts), mean $\pm$ SD & Active $-$ random (pts) \\",
                 r"\midrule"]
        for n in sorted(S["n"].unique()):
            a = S[(S.n == n) & (S.strategy == "aleatorio")].iloc[0]
            b = S[(S.n == n) & (S.strategy == "activo")].iloc[0]
            lines.append(f"{int(n):,} & {100*a.delta_mean:+.2f} $\\pm$ {100*a.delta_sd:.2f} & "
                         f"{100*b.delta_mean:+.2f} $\\pm$ {100*b.delta_sd:.2f} & {100*AR[int(n)]['mean']:+.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (DST / "tab_exp070_budget.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for f in (OUTPUTS / "exp070" / "figures").glob("*.png"):
            shutil.copy2(f, FIG / f"exp070_{f.name}")

    (DST / "v3_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    print("bloques:", list(out["blocks"]))


if __name__ == "__main__":
    main()
