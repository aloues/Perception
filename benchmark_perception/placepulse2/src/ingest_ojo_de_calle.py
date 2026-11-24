"""
EXP-023 — convierte las sesiones guardadas por la herramienta web "Ojo de Calle" al formato
que ya espera `annotation_metrics.py` (columnas: image_id, concept, value).

Cómo se usa:
1. Exporta la colección 'annotations' de la herramienta web a <esta carpeta>/raw_sessions
   (no hay API pública; paginar si hay más de 100 sesiones).
2. Cada sesion queda como un .json en raw_sessions/annotations/<doc_id>.json.
3. Corre este script para convertirlas a outputs/concept_validation/human_annotations/annotator_web.csv
   (todas las sesiones se juntan en un solo "anotador" agregado por ahora: ver LIMITACION abajo).

LIMITACION IMPORTANTE
----------------------
Esta herramienta pide una respuesta de presencia (si/no/no-se-ve-claro) por concepto e imagen a
CUALQUIER visitante, sin turnos fijos ni entrenamiento formal. No es el protocolo formal de 2-3
anotadores fijos descrito en ANNOTATION_GUIDE.md (ese sigue siendo necesario para los conceptos
subjetivos: facade_condition, visual_openness, sightline_obstruction, commercial_activity).
Sirve como PRIMERA SENAL, no como reemplazo. Cuando haya >= 2 sesiones de personas DISTINTAS que
respondieron la MISMA imagen+concepto, se puede medir acuerdo real (kappa); con una sola respuesta
por celda no hay con qué comparar y esa celda no entra en `human_agreement.csv`.

Salida:
    outputs/concept_validation/human_annotations/annotator_web.csv   (una fila por persona x imagen x concepto)
    outputs/concept_validation/human_annotations/web_sessions_summary.json

Uso:
    python ingest_ojo_de_calle.py --raw-dir <carpeta con los .json descargados>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import OUTPUTS, log

CV = OUTPUTS / "concept_validation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="carpeta con los .json de sesiones (uno por doc_id de la coleccion 'annotations')")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    files = sorted(raw.rglob("*.json"))
    log(f"{len(files)} sesiones encontradas en {raw}")

    rows, sessions = [], []
    for f in files:
        if f.name.startswith("_"):
            continue  # documentos de verificacion, no son datos reales
        d = json.loads(f.read_text(encoding="utf-8"))
        data = d.get("data", d)
        annot = data.get("annotator_id", f.stem)
        label = data.get("annotator_label") or annot
        answers = data.get("answers", [])
        for a in answers:
            rows.append(dict(annotator_id=annot, annotator_label=label,
                             image_id=a["image_id"], concept=a["concept"],
                             value=a["value"]))
        sessions.append(dict(doc_id=f.stem, annotator_id=annot, n_answers=len(answers),
                             started_at=data.get("started_at"), finished_at=data.get("finished_at")))

    if not rows:
        log("sin respuestas todavia — nada que convertir")
        return

    df = pd.DataFrame(rows)
    # value: 1 = si, 0 = no, None/NaN = "no estoy seguro" (queda vacio, como pide annotation_metrics.py)
    (CV / "human_annotations").mkdir(parents=True, exist_ok=True)
    out = CV / "human_annotations" / "annotator_web.csv"
    df.to_csv(out, index=False)
    log(f"-> {out}  ({len(df)} juicios, {df.annotator_id.nunique()} personas distintas)")

    dup = df.groupby(["image_id", "concept"]).annotator_id.nunique()
    log(f"celdas (imagen, concepto) con >= 2 personas distintas: {(dup >= 2).sum()} / {len(dup)}")

    (CV / "human_annotations" / "web_sessions_summary.json").write_text(
        json.dumps(dict(n_sessions=len(sessions), n_answers=len(df),
                        n_annotators=int(df.annotator_id.nunique()),
                        n_cells_with_2plus_annotators=int((dup >= 2).sum()),
                        sessions=sessions), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
