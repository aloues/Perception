"""Genera ANNOTATION_GUIDE.md, annotation_schema.json y concept_definitions/*.md
(nivel 2: validacion de medicion). Los ejemplos positivos/negativos/ambiguos con IDs reales
se completan tras una ronda piloto: NO se inventan aqui."""
import json
from utils import OUTPUTS

OUT = OUTPUTS / "concept_validation"
V1 = json.loads((OUTPUTS / "codebook" / "codebook_v1.json").read_text(encoding="utf-8"))["concepts"]

# tipo de anotacion por concepto: presence (+count | ordinal | area_bin)
SPEC = {
 "greenery": ("area_bin", "Fraccion del encuadre con vegetacion.", "Cesped seco/arbustos y arboles visibles.", "Macetas aisladas diminutas; pasto artificial; arboles pintados/murales.", "Arbol en sombra oscura no distinguible de edificio."),
 "tree_canopy": ("count", "Arboles con copa visible (conteo).", "Arboles de calle o de jardin con tronco y copa visibles.", "Arbustos bajos, setos, palmeras muy pequenas <5% del alto.", "Arboles muy lejanos o cortados por el borde."),
 "visual_openness": ("ordinal_1_5", "1=calle muy cerrada, 5=muy abierta (cielo y profundidad).", "Calificar segun cielo visible y profundidad del eje.", "No calificar por belleza ni ancho de vereda.", "Curvas donde la profundidad no es evaluable."),
 "sightline_obstruction": ("ordinal_1_5", "1=vision libre a media distancia, 5=bloqueada.", "Muros, vehiculos grandes, follaje denso a altura de ojos que bloquean la vista.", "Objetos pequenos que no bloquean.", "Vehiculo grande justo al costado de la camara."),
 "graffiti": ("presence+area_bin", "Pintadas/tags no autorizados en muros o mobiliario.", "Tags, pintadas con spray, pintadas politicas.", "Murales autorizados/artisticos (anotar `mural`); publicidad; letreros pintados de comercios.", "Si no se distingue mural de pintada: marcar `ambiguous`."),
 "litter": ("presence+count", "Basura dispersa/acumulada visible en via o vereda.", "Bolsas fuera de contenedor, escombros, acumulaciones, carton, botellas visibles.", "Contenedores/tachos normales y cerrados.", "Bolsas para recojo programado; bultos indistinguibles."),
 "facade_condition": ("ordinal_1_5", "1=muy deteriorada/abandonada, 5=muy bien mantenida (fachada dominante).", "Pintura, revoque, ventanas rotas, humedad, abandono.", "No evaluar estilo arquitectonico ni antiguedad como tal.", "Fachadas ocultas por vehiculos o vegetacion."),
 "vacant_lot": ("presence", "Terreno baldio/abandonado sin construir.", "Solares vacios, terrenos con maleza o escombros.", "Parques, estacionamientos formales, obras activas con cerco.", "Obra abandonada vs activa."),
 "fortification_transparency": ("ordinal_1_5+subfeatures", "1=muy fortificada/opaca, 5=muy transparente. Subfeatures: wall_fraction (area_bin), fence_fraction (area_bin), barred_openings (presence), visible_doors_windows (count), facade_transparency (ordinal).", "Muros altos ciegos, cercos, portones, rejas, ventanas enrejadas.", "Muros bajos decorativos; barandas.", "Rejas decorativas vs de seguridad."),
 "sidewalk": ("presence+area_bin", "Vereda visible y transitable.", "Acera pavimentada continua o discontinua.", "Berma de tierra sin pavimento (anotar `unpaved`).", "Calles peatonales sin vereda definida."),
 "lighting_infrastructure": ("count", "Postes/luminarias visibles. NO se infiere iluminacion efectiva.", "Postes de alumbrado, brazos con luminaria, farolas.", "Postes sin luminaria evidente (electricos/telefonia) -> `ambiguous`.", "Lamparas lejanas o ocultas por arboles."),
 "pedestrians": ("count", "Personas visibles en espacio publico.", "Personas caminando, paradas o sentadas en via/vereda.", "Personas dentro de vehiculos; imagenes en carteles.", "Personas muy pequenas o cortadas."),
 "vehicles": ("count", "Vehiculos motorizados visibles.", "Autos, buses, camiones, motos.", "Bicicletas (no cuentan); vehiculos en carteles.", "Vehiculos parcialmente visibles."),
 "commercial_activity": ("presence+ordinal", "Evidencia visible de frente comercial ACTIVO (letrero, vitrina, entrada, puesto).", "Tiendas con letrero/vitrina, puestos ambulantes, mercados.", "NO asumir actividad por el tipo de edificio.", "Comercio cerrado/abandonado -> `ambiguous`."),
 "street_signage": ("count", "Senales de transito/calle.", "Senales viales, nombres de calle.", "Letreros comerciales.", "Senales de espaldas."),
 "street_furniture": ("count", "Bancas, papeleras, paraderos, bolardos.", "Mobiliario publico fijo.", "Objetos privados.", "Objetos de uso dudoso."),
 "security_bars": ("presence+count", "Rejas/barrotes en ventanas o puertas.", "Ventanas enrejadas, rejillas metalicas, portones metalicos.", "Rejas decorativas claramente ornamentales.", "Portones cerrados vs rejas."),
}

schema = {
  "version": "1", "unit": "imagen",
  "common_fields": {"image_id": "str", "annotator_id": "str", "quality_flag": ["ok", "unusable", "occluded", "night", "blurry"], "notes": "str"},
  "per_concept": {c["concept_id"]: {"type": SPEC[c["concept_id"]][0], "allow": ["value", "ambiguous", "not_assessable"]}
                  for c in V1 if c["concept_id"] in SPEC},
  "area_bins": ["0", "(0,10%]", "(10,30%]", "(30,60%]", ">60%"],
  "bbox_subset": "100 imagenes MAIN para IoU/Dice en conceptos de objeto (formato COCO xywh)",
  "annotators": {"standard": 2, "subjective_triple": ["facade_condition", "visual_openness", "sightline_obstruction", "commercial_activity"],
                 "adjudication": "desacuerdos resueltos por 3er anotador; ambiguos se reportan aparte, no se fuerzan"},
}
(OUT / "annotation_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

for c in V1:
    cid = c["concept_id"]
    if cid not in SPEC:
        continue
    t, d, inc, exc, amb = SPEC[cid]
    (OUT / "concept_definitions" / f"{cid}.md").write_text(
        f"# {cid} — {c['name']}\n\n- **Tipo de salida**: {t}\n- **Definicion**: {d}\n- **Incluir**: {inc}\n"
        f"- **Excluir**: {exc}\n- **Casos ambiguos**: {amb}\n- **Decision codebook**: {c['decision']}\n\n"
        "## Ejemplos con image_id\n\nPENDIENTE: completar tras la ronda piloto (positivos / negativos / ambiguos reales). "
        "No se inventan ejemplos.\n", encoding="utf-8")

L = ["# Guia de anotacion de conceptos (validacion de MEDICION)", "",
     "Objetivo: generar ground truth humano para saber si los conceptos del `CODEBOOK_V1` pueden medirse de forma "
     "consistente por personas y por extractores automaticos. NO anota percepcion de seguridad.", "",
     "## Reglas generales",
     "- Anotar solo lo VISIBLE en la imagen (diurna). No inferir funcion, causas ni seguridad.",
     "- Si no se puede decidir: `ambiguous` (se reporta como categoria propia) o `not_assessable` (no visible). No forzar.",
     "- Ignorar la marca de agua 'Google' de la esquina inferior izquierda.",
     "- Las **preanotaciones automaticas** (YOLO-World/CLIP) son sugerencias: el humano acepta/corrige/rechaza; "
     "nunca son ground truth. Para evitar sesgo de anclaje, una parte de las imagenes se anota SIN preanotacion.",
     "- Anotadores independientes; no discutir antes de terminar.", "",
     "## Diseno",
     "- 300 imagenes MAIN (30 ciudades, 10 c/u) + 150 STRESS (oscuras, borrosas, alto contraste). Ver `annotation_sample.csv`.",
     "- 2 anotadores por imagen; 100 imagenes MAIN con 3er anotador para conceptos subjetivos "
     "(facade_condition, visual_openness, sightline_obstruction, commercial_activity); desacuerdos adjudicados.",
     "- Lima: BLOQUEADO (no hay imagenes locales).", "",
     "## Metricas (criterios PROPUESTOS, no umbrales universales)",
     "| Tipo | Acuerdo humano | Extractor vs humano |", "|---|---|---|",
     "| binario | Cohen/Fleiss kappa o Krippendorff alpha | precision, recall, F1, AP |",
     "| ordinal | kappa ponderado + matriz de confusion | kappa ponderado, Spearman |",
     "| conteo | ICC | MAE, correlacion |",
     "| area | ICC sobre fracciones | MAE; IoU/Dice si hay mascara |", "",
     "**Criterio experimental propuesto** (a fijar antes de ver resultados del extractor): un concepto se considera "
     "*medible de forma fiable* si el acuerdo humano es al menos 'sustancial' (kappa/alpha >= 0.6, ICC >= 0.75) y el extractor "
     "alcanza en el set MAIN un F1/kappa/correlacion no inferior a ese nivel de acuerdo humano menos un margen "
     "predefinido (propuesta: 0.10). Son propuestas justificables, no estandares; se documentan y se pueden revisar antes de anotar.", "",
     "Un concepto pasa a *validated measurable concept* solo si: (1) respaldo teorico, (2) humanos consistentes, (3) extractor aceptable.", "",
     "## Fichas por concepto", "Ver `concept_definitions/<concepto>.md` (definicion, incluir, excluir, ambiguos, unidad de salida).", ""]
for cid, (t, d, inc, exc, amb) in SPEC.items():
    L.append(f"- **{cid}** [{t}] — {d} *Incluir:* {inc} *Excluir:* {exc} *Ambiguo:* {amb}")
L += ["", "Ejemplos: por ejemplo `graffiti` incluye tags visibles y separa mural/artwork; `lighting_infrastructure` cuenta "
      "poste/luminaria y NO infiere 'calle bien iluminada'; `commercial_activity` exige evidencia de frente comercial "
      "activo, no asumir por tipo de edificio."]
(OUT / "ANNOTATION_GUIDE.md").write_text("\n".join(L), encoding="utf-8")
print("ok")
