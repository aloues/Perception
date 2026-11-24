"""
CODEBOOK V1 — THEORY VERIFIED (nivel 1 de 3: respaldo teorico/bibliografico).

Tres niveles de validez, que NO se mezclan:
  1. respaldo teorico/bibliografico   -> justifica que valga la pena MEDIR el concepto
  2. validacion de medicion           -> anotacion humana: se puede medir de forma fiable?
  3. validacion perceptual            -> Place Pulse / encuesta: se asocia a perceived safety?
Este archivo cubre SOLO el nivel 1, y aun asi de forma PROVISIONAL.

PROVENANCE (honesto): ni los PDF ni la matriz bibliografica del Drive estan en esta
maquina. Las afirmaciones "que apoya / que NO apoya" proceden del conocimiento del
autor de la sesion sobre la literatura canonica y NO estan verificadas contra texto
completo (fulltext_verified=False en TODAS las filas). Los trabajos recientes que la
usuaria identifico (Zhang 2022, Solymosi 2024, Mendoza 2026, Hu 2023, Wei 2026, Ma 2024)
se registran como NEEDS_FULLTEXT_VERIFICATION sin atribuirles contenido.
`evidence_status` es por tanto PROVISIONAL hasta verificar con el Drive.

Salidas: outputs/codebook/CODEBOOK_V1_THEORY_VERIFIED.md, codebook_v1.json,
         CODEBOOK_EVIDENCE_MATRIX.csv
"""
import csv
import json
from utils import OUTPUTS

OUT = OUTPUTS / "codebook"
V0 = json.loads((OUT / "codebook_v0.json").read_text(encoding="utf-8"))
REFS = V0["references"]

# nivel de detectabilidad observado en el PoC (72 imgs; acuerdo entre herramientas != correccion)
POC = {
    "vehicles": "acuerdo alto (rho .78 YOLO-World~GDINO)",
    "lighting_infrastructure": "acuerdo alto (rho .71)",
    "pedestrians": "acuerdo moderado (rho .64)",
    "tree_canopy": "acuerdo moderado/bajo (rho .41)",
    "street_signage": "acuerdo moderado (rho .47)",
    "street_furniture": "acuerdo bajo (rho .23)",
    "graffiti": "sin acuerdo; GDINO disparaba sobre logo Google",
    "litter": "sin acuerdo",
    "commercial_activity": "sin acuerdo",
    "security_bars": "solo YOLO-World (frase compuesta se pierde en GDINO)",
}
NOTPOC = "no evaluado como objeto en el PoC (concepto de escena/area)"

C = []


def add(cid, name, family, definition, evid, refs, claims, obs, tool, out, lit, risks,
        decision, reason, notes=""):
    C.append(dict(concept_id=cid, name=name, family=family, operational_definition=definition,
                  evidence_status=evid, evidence_status_note="PROVISIONAL (sin texto completo)",
                  references=refs, what_literature_claims=claims, visual_observability=obs,
                  candidate_tool=tool, expected_output=out, literature_expectation=lit,
                  empirical_direction="UNKNOWN", causal_claim="NO", risks=risks,
                  decision=decision, decision_reason=reason,
                  poc_detectability=POC.get(cid, NOTPOC),
                  measurement_validation="PENDIENTE (requiere anotacion humana)",
                  perceptual_validation="PENDIENTE (requiere Place Pulse / encuesta local)",
                  notes=notes))


add("greenery", "vegetacion / verde visible", "naturaleza",
    "Fraccion del encuadre ocupada por vegetacion visible a nivel de calle (arboles, setos, cesped, jardineras).",
    "A", ["kuo_sullivan_2001", "michael_hull_1994", "li_2015_gvi"],
    "Vegetacion cuidada se asocia a menos crimen reportado (Kuo & Sullivan); vegetacion densa que reduce visibilidad puede asociarse a mas miedo (Michael & Hull); GVI es un metodo de medicion sobre street view (Li et al.).",
    "alta", "segmentacion semantica / SAM 2 guiado por texto", "fraccion_area [0,1]",
    "context-dependent", "Signo contradictorio segun cuidado y densidad; estacionalidad.",
    "CORE", "Correlato mas estudiado; medible en street view.",
    "Antes era `vegetation`. Posible redundancia con tree_canopy: exigir correlacion.")
add("tree_canopy", "arbolado (copa alta)", "naturaleza",
    "Arboles de porte alto con copa visible (conteo y fraccion de area de copa), separados de vegetacion baja.",
    "B", ["li_2015_gvi", "michael_hull_1994"],
    "Sustenta separar copa alta de seto denso por su efecto distinto sobre visibilidad; no se conoce (sin verificar) evidencia perceptual directa especifica de copa.",
    "alta", "YOLO-World / Grounding DINO + SAM 2 para area", "conteo + fraccion_area",
    "context-dependent", "Solapa con greenery.", "CORE",
    "Permite testear si la ambiguedad de greenery se debe al porte.")
add("visual_openness", "apertura visual", "prospect-refuge",
    "Campo visual abierto: fraccion de cielo visible y profundidad del eje de calle.",
    "A", ["appleton_1975", "fisher_nasar_1992", "ewing_handy_2009"],
    "Prospect-refuge (Appleton); Fisher & Nasar relacionan prospecto/refugio/escape con miedo al crimen; Ewing & Handy operacionalizan 'enclosure'.",
    "media", "segmentacion semantica (cielo/edificio)", "fraccion_area [0,1]",
    "generally positive", "Encuadre de street view fija parte del cielo; controlar por heading/camara.",
    "CORE", "Teoria explicita y operacionalizacion medible.")
add("sightline_obstruction", "obstruccion de la linea de vision", "prospect-refuge",
    "Elementos grandes que bloquean la vision a distancia media (muros ciegos, vehiculos grandes, vegetacion densa a altura de ojos, contenedores).",
    "A", ["fisher_nasar_1992", "newman_1972"],
    "Escondites potenciales/ausencia de escape se asocian a miedo (Fisher & Nasar); vigilancia natural (Newman).",
    "media", "deteccion open-vocabulary + heuristica de posicion/area", "score [0,1]",
    "generally negative", "Definicion operativa fragil y subjetiva.", "CORE",
    "Requiere 3 anotadores o adjudicacion por subjetividad.")
add("graffiti", "graffiti / pintadas", "desorden fisico",
    "Pintadas no autorizadas visibles en muros/mobiliario. Excluye murales autorizados (se anotan aparte).",
    "A", ["wilson_kelling_1982", "ordonez_berg_2014"],
    "Broken windows: signos de desorden se asocian a percepcion de descuido; NO establece que el graffiti cause crimen.",
    "media", "YOLO-World/GDINO (varios noun phrases) + CLIP contrastivo", "presencia + conteo + area",
    "generally negative", "Mural vs tag ambiguo; PoC: detectores no concuerdan; logo Google genera falsos positivos.",
    "CORE", "Relevante teoricamente aunque hoy no sabemos medirlo bien.",
    "La relevancia teorica NO implica que el detector actual lo mida correctamente.")
add("litter", "basura visible", "desorden fisico",
    "Basura dispersa o acumulada en via/vereda (bolsas, escombros, carton). Excluye contenedores normales.",
    "B", ["wilson_kelling_1982"],
    "Desorden fisico como senal de descuido (broken windows). Evidencia perceptual directa especifica sin verificar.",
    "media", "YOLO-World/GDINO + CLIP contrastivo", "presencia + conteo",
    "generally negative", "Objetos pequenos, resolucion 224 insuficiente; ambiguo con contenedores.", "CORE",
    "Nucleo de desorden fisico.")
add("facade_condition", "estado de fachadas", "desorden fisico",
    "Deterioro visible de fachadas/edificios (pintura desconchada, danos, abandono) en escala ordinal.",
    "A", ["wilson_kelling_1982", "naik_2014_streetscore", "salesses_2013"],
    "Streetscore/Place Pulse muestran que rasgos visuales de calidad de edificacion se asocian a puntajes de percepcion; Broken windows lo respalda teoricamente.",
    "media", "CLIP ordinal (prompts) o clasificador entrenado", "ordinal 1-5",
    "generally negative", "Muy subjetivo; requiere 3 anotadores.", "CORE", "Fuerte soporte perceptual y teorico.")
add("vacant_lot", "lote vacio / abandonado", "desorden fisico",
    "Terreno sin construir, abandonado o baldio visible.",
    "B", ["wilson_kelling_1982", "newman_1972"],
    "Espacios abandonados sin control natural. Evidencia especifica sin verificar.",
    "baja", "CLIP contrastivo + segmentacion", "presencia / area",
    "generally negative", "Confundible con estacionamientos, obras, parques.", "CORE",
    "Mantener con validacion estricta; candidato a EXPLORATORY si la anotacion es baja.")
add("fortification_transparency", "fortificacion / transparencia de fachada", "fortificacion",
    "Grado en que la frontera calle-edificio es opaca/fortificada vs transparente: muros ciegos, rejas/cercos, "
    "aberturas con barrotes, puertas/ventanas visibles. Subfeatures a definir: wall_fraction, fence_fraction, "
    "barred_openings, visible_doors_windows, facade_transparency.",
    "B", ["newman_1972", "jacobs_1961", "ewing_handy_2009"],
    "Territorialidad/vigilancia natural (Newman, Jacobs) y transparencia como cualidad de diseno (Ewing & Handy); direccion del efecto de fortificacion es ambigua.",
    "media", "segmentacion + deteccion de ventanas/puertas/rejas", "score compuesto + subfeatures",
    "ambiguous", "Fortificacion puede indicar inseguridad previa o proteccion; construccion compuesta.", "CORE",
    "Absorbe `blank_wall`. Solo se documenta la definicion; subfeatures no implementados.")
add("blank_wall", "muro ciego (aislado)", "fortificacion",
    "Muro sin aberturas visibles.", "D", ["ewing_handy_2009", "jacobs_1961"],
    "Ver fortification_transparency.", "media", "-", "-", "context-dependent",
    "Redundante como concepto aislado.", "MERGE", "Integrado en fortification_transparency como `wall_fraction`.")
add("sidewalk", "vereda / acera", "infraestructura peatonal",
    "Presencia, continuidad y estado de la vereda.", "B", ["ewing_handy_2009", "jacobs_1961"],
    "Infraestructura peatonal como cualidad de caminabilidad (Ewing & Handy) y actividad de calle (Jacobs).",
    "alta", "segmentacion semantica", "fraccion_area + ordinal de estado",
    "generally positive", "Vereda presente no implica peatones ni cuidado.", "CORE", "Medible y con base de diseno urbano.")
add("lighting_infrastructure", "infraestructura de alumbrado", "infraestructura",
    "Presencia de postes/luminarias visibles. NO mide iluminacion efectiva: las imagenes son diurnas.",
    "B", ["welsh_farrington_2008"],
    "Revision sistematica: mejor alumbrado se asocia a menos crimen; no se infiere efecto de la mera presencia de postes.",
    "alta", "YOLO-World / GDINO", "conteo",
    "context-dependent", "Renombrado desde `street_lighting` para no sugerir luz efectiva; postes tambien son de energia/telefonia.",
    "CORE", "Objeto discreto con alto acuerdo entre detectores.")
add("pedestrians", "peatones", "actividad humana",
    "Personas visibles en el espacio publico (conteo).", "B", ["jacobs_1961", "naik_2014_streetscore"],
    "Ojos en la calle (Jacobs); presencia de personas como rasgo visual analizado en percepcion.",
    "alta", "YOLO-World / GDINO", "conteo",
    "context-dependent", "Una foto es un instante; 0 peatones depende de la hora.", "CORE", "Objeto estandar.")
add("vehicles", "vehiculos", "actividad humana",
    "Vehiculos visibles (conteo y area ocupada).", "C", ["naik_2014_streetscore", "ordonez_berg_2014"],
    "Sin respaldo teorico fuerte verificado; se usa como covariable de contexto/control.",
    "alta", "YOLO-World", "conteo + fraccion_area", "ambiguous",
    "Puede ser proxy de trafico y no de seguridad.", "CORE",
    "Deteccion mas fiable del PoC; util como control de la tuberia.")
add("commercial_activity", "actividad comercial", "actividad humana",
    "Evidencia visible de frente comercial activo (letrero, vitrina, entrada, puesto ambulante).",
    "B", ["jacobs_1961", "ewing_handy_2009"],
    "Usos mixtos y actividad como vigilancia natural (Jacobs).", "media",
    "GDINO/YOLO-World con noun phrases + CLIP", "presencia + conteo", "generally positive",
    "No asumir actividad por tipo de edificio; PoC sin acuerdo.", "EXPLORATORY", "Relevante pero deteccion no fiable aun.")
add("street_signage", "senaletica vial", "infraestructura",
    "Senales de transito/calle visibles.", "C", ["ewing_handy_2009"],
    "Soporte debil (legibilidad/complejidad urbana).", "alta", "YOLO-World", "conteo", "ambiguous",
    "Poca relacion teorica directa con seguridad.", "EXPLORATORY", "Mantener solo como exploratorio.")
add("street_furniture", "mobiliario urbano", "infraestructura",
    "Bancas, papeleras, paraderos y similares.", "C", ["ewing_handy_2009"],
    "Soporte debil; cualidad de diseno.", "media", "GDINO/YOLO-World", "conteo", "context-dependent",
    "Objetos heterogeneos; PoC acuerdo bajo.", "EXPLORATORY", "Mantener solo como exploratorio.")
add("security_bars", "rejas en aberturas", "fortificacion",
    "Rejas o barrotes en ventanas/puertas visibles.", "C", ["newman_1972"],
    "Territorialidad/fortificacion; direccion ambigua.", "media",
    "YOLO-World + noun phrases (window bars, metal grille, security grille, metal gate)", "presencia + conteo",
    "ambiguous", "Frase compuesta 'window with metal bars' se pierde en Grounding DINO.", "EXPLORATORY",
    "Se solapa con fortification_transparency (`barred_openings`); revisar fusion tras anotar.")

# ── evidencia por (concepto, referencia) ───────────────────────────────────────
# Afirmaciones prudentes; NINGUNA verificada en texto completo.
NV = "NOT verified against fulltext"
ROWS = []
def row(cid, ref, year, target, etype, supports, notsup, context, notes=""):
    ROWS.append(dict(concept_id=cid, reference=ref, year=year, target_type=target, evidence_type=etype,
                     what_the_paper_supports=supports, what_it_does_NOT_support=notsup, context=context,
                     fulltext_verified="False", notes=(notes + " " + NV).strip()))

for c in C:
    for k in c["references"]:
        full = REFS[k]
        yr = full.split("(")[1].split(")")[0][:4] if "(" in full else ""
        et = ("teoria" if k in ("wilson_kelling_1982", "newman_1972", "jacobs_1961", "appleton_1975")
              else "revision sistematica" if k == "welsh_farrington_2008"
              else "estudio perceptual/ML" if k in ("naik_2014_streetscore", "salesses_2013", "ordonez_berg_2014", "zhang_2018")
              else "metodo de medicion" if k == "li_2015_gvi" else "estudio empirico")
        tt = ("perceived_safety" if k in ("naik_2014_streetscore", "salesses_2013", "ordonez_berg_2014", "zhang_2018")
              else "fear_of_crime" if k == "fisher_nasar_1992"
              else "observed_crime" if k in ("kuo_sullivan_2001", "michael_hull_1994", "welsh_farrington_2008")
              else "theory/method")
        row(c["concept_id"], full, yr, tt, et, c["what_literature_claims"],
            "No demuestra causalidad sobre percepcion de seguridad en street view de Lima; no valida el detector.",
            "contexto original del estudio (ciudad/escala) por verificar",
            "citada ya en CODEBOOK_V0" + ("; si el estudio usa crimen observado NO es evidencia perceptual" if tt == "observed_crime" else ""))

for k, yr, tt, note in [
    ("koh_2020_cbm", "2020", "method", "Concept Bottleneck Models: predecir via conceptos intermedios interpretables."),
    ("dubey_2016", "2016", "perceived_safety", "Place Pulse 2.0: comparaciones pareadas a escala global."),
    ("radford_2021_clip", "2021", "method", "CLIP: scoring contrastivo texto-imagen."),
    ("liu_2024_gdino", "2024", "method", "Grounding DINO: deteccion open-set guiada por texto."),
    ("cheng_2024_yoloworld", "2024", "method", "YOLO-World: deteccion open-vocabulary en tiempo real."),
]:
    row("GENERAL", REFS.get(k, k), yr, tt, "metodo/dataset", note,
        "No valida por si mismo ningun concepto del codebook.", "herramienta/dataset de la tuberia", "")

for ref, yr in [("Chen et al. 2020 — SimCLR", "2020"), ("Oquab et al. — DINOv2", "2023/2024"),
                ("Jiang et al. 2023 — From CLIP to DINO", "2023"),
                ("Zhang et al. 2022 — Urban-Safety Perception framework", "2022"),
                ("Solymosi et al. 2024 — fear of crime / computational systematic social observation", "2024"),
                ("Mendoza et al. 2026 — perceived street safety + semantic features", "2026"),
                ("Hu et al. 2023 — UPDExplainer", "2023"),
                ("Wei et al. 2026", "2026"), ("Ma et al. 2024", "2024")]:
    row("GENERAL", ref, yr, "por determinar", "por determinar", "NEEDS_FULLTEXT_VERIFICATION (no asignado a ningun concepto)",
        "NEEDS_FULLTEXT_VERIFICATION", "Identificado por la usuaria; no disponible localmente",
        "Contenido NO atribuido en esta sesion.")

with open(OUT / "CODEBOOK_EVIDENCE_MATRIX.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0])); w.writeheader(); w.writerows(ROWS)

meta = dict(version="v1-theory-verified (PROVISIONAL)", base="codebook_v0",
            validity_levels=["1 teorico/bibliografico", "2 medicion (anotacion humana)",
                             "3 perceptual (Place Pulse/encuesta)"],
            evidence_status_scale={"A": "respaldo perceptual directo", "B": "respaldo teorico/empirico indirecto",
                                   "C": "exploratorio", "D": "redundante / fusionar"},
            provenance_limitation="Sin PDF/matriz del Drive en esta maquina; fulltext_verified=False en todo.",
            concepts=C)
(OUT / "codebook_v1.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

# ── markdown ───────────────────────────────────────────────────────────────────
L = ["# CODEBOOK V1 — THEORY VERIFIED (PROVISIONAL)", "",
     "Evoluciona desde `codebook_v0` (17 candidatos con citas no verificadas). **No esta validado.**", "",
     "## Tres niveles de validez (no se mezclan)",
     "1. **Respaldo teorico/bibliografico** — justifica que valga la pena medir el concepto (este documento, provisional).",
     "2. **Validacion de medicion** — la anotacion humana valida si puede medirse (PENDIENTE: `outputs/concept_validation/`).",
     "3. **Validacion perceptual** — Place Pulse / encuesta local valida si se asocia a perceived safety (PENDIENTE).", "",
     "Un concepto solo pasa a *validated measurable concept* si cumple los tres. La relevancia teorica no implica que "
     "el detector actual lo mida bien (ej.: graffiti).", "",
     "> Provenance: los PDF y la matriz del Drive no estan en esta maquina. `evidence_status` es PROVISIONAL y "
     "ninguna fila esta verificada contra texto completo; los trabajos recientes (Zhang 2022, Solymosi 2024, Mendoza 2026, "
     "Hu 2023, Wei 2026, Ma 2024) figuran como NEEDS_FULLTEXT_VERIFICATION. Ver `CODEBOOK_EVIDENCE_MATRIX.csv`.", "",
     "> **Ningun concepto se afirma como causa de inseguridad.** `empirical_direction = UNKNOWN` en todos hasta ejecutar el experimento; "
     "`literature_expectation` es solo lo que la literatura hace esperar.", "",
     "Escala de evidencia: A directo perceptual · B indirecto teorico/empirico · C exploratorio · D redundante/fusionar.", "",
     "## Resumen", "", "| # | Concepto | Familia | Evid. | Decision | Expectativa literatura | Detectabilidad PoC |", "|---|---|---|---|---|---|---|"]
for i, c in enumerate(C, 1):
    L.append(f"| {i} | `{c['concept_id']}` | {c['family']} | {c['evidence_status']} | **{c['decision']}** | "
             f"{c['literature_expectation']} | {c['poc_detectability']} |")
L += ["", "Cambios respecto a V0: `vegetation`→`greenery`; `street_lighting`→`lighting_infrastructure` (imagenes diurnas, "
      "no se mide luz efectiva); `blank_wall` fusionado en `fortification_transparency`; `polarity`→`literature_expectation` "
      "+ `empirical_direction=UNKNOWN`.", "", "## Fichas", ""]
for c in C:
    L += [f"### `{c['concept_id']}` — {c['name']}  ·  {c['decision']}  ·  evidencia {c['evidence_status']} (provisional)", "",
          f"- **Familia**: {c['family']}",
          f"- **Definicion operacional**: {c['operational_definition']}",
          f"- **Que afirma la literatura (sin verificar en texto completo)**: {c['what_literature_claims']}",
          f"- **Referencias**: " + "; ".join(REFS[k] for k in c["references"]),
          f"- **Observabilidad visual**: {c['visual_observability']} · **Herramienta candidata**: {c['candidate_tool']} · **Salida**: {c['expected_output']}",
          f"- **literature_expectation**: {c['literature_expectation']} · **empirical_direction**: UNKNOWN · **causal_claim**: NO",
          f"- **PoC**: {c['poc_detectability']}",
          f"- **Validacion de medicion**: {c['measurement_validation']} · **Validacion perceptual**: {c['perceptual_validation']}",
          f"- **Riesgos**: {c['risks']}",
          f"- **Decision**: {c['decision']} — {c['decision_reason']}"]
    if c["notes"]:
        L.append(f"- **Notas**: {c['notes']}")
    L.append("")
(OUT / "CODEBOOK_V1_THEORY_VERIFIED.md").write_text("\n".join(L), encoding="utf-8")
print(len(C), "conceptos;", len(ROWS), "filas de evidencia")
