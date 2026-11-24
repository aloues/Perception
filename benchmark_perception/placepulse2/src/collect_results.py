"""
Agrega todos los runs del probe en una tabla unica (CSV + Markdown).

Distingue explicitamente (seccion 21 del brief) entre:
  FULL  = corrida sobre las 110,987 imagenes  -> "Resultado experimental nuevo"
  SMOKE = corrida sobre el subconjunto de 20k -> "Resultado preliminar - smoke test"

La marca se deriva del run_id (prefijo `smoke__`), no se escribe a mano.
"""

from __future__ import annotations

import json

import pandas as pd

from utils import OUTPUTS, RUNS, log

ENCODER_LABEL = {
    "ours_simclr_vit_b16": "OURS (SimCLR ViT-B/16)",
    "imagenet_vit_b16": "ImageNet ViT-B/16",
    "dinov2_vit_b14": "DINOv2 ViT-B/14",
}
PROTOCOL_ORDER = ["paper_protocol", "image_disjoint", "city_holdout"]
ENCODER_ORDER = list(ENCODER_LABEL)


def main() -> None:
    rows = []
    for d in sorted(RUNS.glob("*")):
        f = d / "metrics.json"
        if not f.exists():
            log(f"  (sin metrics.json, run incompleto) {d.name}")
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        if d.name.startswith("smoke__"):
            regime = "SMOKE"
        elif d.name.startswith("ctrl__"):
            regime = "CTRL"
        else:
            regime = "FULL"
        cfg = {}
        cfgf = d / "config.yaml"
        if cfgf.exists():
            for line in cfgf.read_text(encoding="utf-8").splitlines():
                if ": " in line and not line.startswith("environment"):
                    k, v = line.split(": ", 1)
                    cfg[k] = v
        for split_key, v in m["test"].items():
            rows.append({
                "regime": regime,
                "encoder": m["encoder"],
                "encoder_label": ENCODER_LABEL.get(m["encoder"], m["encoder"]),
                "protocol": m["protocol"],
                "test_split": split_key,
                "head": m["head"],
                "seed": cfg.get("seed", ""),
                "embed_dim": cfg.get("embed_dim", ""),
                "n_train_pairs": cfg.get("n_train", ""),
                "n_val_pairs": cfg.get("n_val", ""),
                "n_test_pairs": v["n"],
                "pairwise_accuracy": round(v["pairwise_accuracy"], 4),
                "ci95_low": round(v["ci95_low"], 4),
                "ci95_high": round(v["ci95_high"], 4),
                "balanced_accuracy": round(v["balanced_accuracy"], 4),
                "log_loss": round(v["log_loss"], 4),
                "brier_score": round(v["brier_score"], 4),
                "label_balance": round(v["label_balance"], 4),
                "macro_acc_by_city": (round(v["by_city"]["macro_average"], 4)
                                      if "by_city" in v else ""),
                "n_cities": (v["by_city"]["n_cities"] if "by_city" in v else ""),
                "val_pairwise_accuracy": round(m["val"]["pairwise_accuracy"], 4),
                "status_label": {"SMOKE": "Resultado preliminar - smoke test",
                                 "CTRL": "Resultado experimental nuevo - control",
                                 "FULL": "Resultado experimental nuevo"}[regime],
                "run_id": m["run_id"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        log("No hay runs con metrics.json todavia.")
        return
    df["_e"] = df["encoder"].apply(lambda e: ENCODER_ORDER.index(e) if e in ENCODER_ORDER else 9)
    df["_p"] = df["protocol"].apply(lambda p: PROTOCOL_ORDER.index(p) if p in PROTOCOL_ORDER else 9)
    df = df.sort_values(["regime", "_p", "_e", "test_split", "seed"]).drop(columns=["_e", "_p"])

    # ── variabilidad entre seeds (seccion 9 del brief) ───────────────────
    seedvar = []
    for (reg, enc, proto, split), g in df.groupby(["regime", "encoder", "protocol", "test_split"]):
        if g["seed"].nunique() < 2:
            continue
        a = g["pairwise_accuracy"].astype(float)
        seedvar.append({
            "regime": reg, "encoder": enc, "protocol": proto, "test_split": split,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": ",".join(sorted(g["seed"].astype(str))),
            "mean_pairwise_accuracy": round(a.mean(), 4),
            "std_pairwise_accuracy": round(a.std(ddof=1), 5),
            "min": round(a.min(), 4), "max": round(a.max(), 4),
            "range_pts": round((a.max() - a.min()) * 100, 3),
        })
    if seedvar:
        sv = pd.DataFrame(seedvar)
        sv.to_csv(OUTPUTS / "seed_variability.csv", index=False)
        log(f"CSV -> {OUTPUTS / 'seed_variability.csv'}  ({len(sv)} filas)")

    out_csv = OUTPUTS / "benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    log(f"CSV -> {out_csv}  ({len(df)} filas)")

    # ── Markdown legible ─────────────────────────────────────────────────
    lines = ["# Resultados del benchmark Place Pulse 2.0 — atributo `safe`", ""]
    lines += ["Probe de ranking lineal sobre embeddings **congelados**. "
              "Metrica principal: pairwise accuracy (azar = 0.50). "
              "IC 95% por bootstrap percentil (2000 resamples).", ""]
    main = df[df["seed"].astype(str) == "42"]
    for regime, tag in (("FULL", "Resultado experimental nuevo — dataset completo "
                                 "(110,987 imagenes / 440,758 pares)"),
                        ("CTRL", "Control de tamano de train — paper_protocol entrenado "
                                 "con el MISMO numero de pares que image_disjoint"),
                        ("SMOKE", "Resultado preliminar — SMOKE TEST "
                                  "(subconjunto reproducible de 20,000 imagenes)")):
        sub = main[main["regime"] == regime]
        if sub.empty:
            continue
        lines += [f"## {regime} — {tag}", ""]
        hdr = ("| Encoder | Protocolo | Test split | n pares | Pairwise acc | IC 95% | "
               "Balanced acc | Macro acc/ciudad |")
        lines += [hdr, "|" + "---|" * 8]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['encoder_label']} | `{r['protocol']}` | `{r['test_split']}` | "
                f"{r['n_test_pairs']:,} | **{r['pairwise_accuracy']:.4f}** | "
                f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
                f"{r['balanced_accuracy']:.4f} | {r['macro_acc_by_city'] or '—'} |"
            )
        lines.append("")

    if seedvar:
        lines += ["## Variabilidad entre seeds (mismo protocolo, misma particion)", ""]
        lines += ["| Encoder | Protocolo | Split | Seeds | Media | Desv. tipica | Rango (pts) |",
                  "|" + "---|" * 7]
        for r in seedvar:
            lines.append(f"| {ENCODER_LABEL.get(r['encoder'], r['encoder'])} | `{r['protocol']}` | "
                         f"`{r['test_split']}` | {r['seeds']} | {r['mean_pairwise_accuracy']:.4f} | "
                         f"{r['std_pairwise_accuracy']:.5f} | {r['range_pts']:.2f} |")
        lines.append("")

    out_md = OUTPUTS / "benchmark_results.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log(f"MD  -> {out_md}")


if __name__ == "__main__":
    main()
