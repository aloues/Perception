"""
FASE C — Codebook urbano v0: de embeddings opacos a conceptos interpretables.

Que es y que NO es
------------------
Es un diccionario de CONCEPTOS CANDIDATOS: elementos visuales del entorno
urbano que la literatura ha relacionado con el juicio humano de seguridad y que
podrian medirse de forma reproducible sobre una imagen de street view.

NO afirma que ninguno de estos elementos CAUSE inseguridad. El codebook es el
instrumento que permite PREGUNTARLO empiricamente: primero se mide el concepto,
despues se contrasta su asociacion con el voto humano `safe` de Place Pulse.

Cada entrada declara:
    concept        identificador estable
    name_es/en     nombre legible
    definition     definicion OPERATIVA (que cuenta y que no)
    grounding      por que esta en el codebook, con referencia bibliografica
    observable     como podria observarse en una imagen
    tool           herramienta candidata
    output         tipo de salida (binaria, conteo, fraccion de area, score)
    polarity       signo ESPERADO por la literatura, o 'ambiguo'
                   -> es una hipotesis a testear, nunca un supuesto
    risks          limitaciones conocidas de medirlo asi

Limitacion de provenance (importante)
-------------------------------------
Los PDF del Drive no estan disponibles en esta maquina (el inventario de Drive
solo contiene checkpoints y artefactos del grafo de Lima, ningun PDF). Las
referencias son por tanto a la literatura canonica del area, y deben
verificarse contra las copias del Drive antes de citarlas en el documento final
de tesis. Cada entrada lleva `citation_verified: false` para no perder de vista
esa deuda.

Salidas:
    outputs/codebook/codebook_v0.json
    outputs/codebook/CODEBOOK_V0.md

Uso:
    python build_codebook.py
"""

from __future__ import annotations

import json
from datetime import date

from utils import OUTPUTS, log

OUT = OUTPUTS / "codebook"

REFERENCES = {
    "wilson_kelling_1982": "Wilson, J. Q., & Kelling, G. L. (1982). Broken Windows: "
                           "The police and neighborhood safety. The Atlantic Monthly.",
    "newman_1972": "Newman, O. (1972). Defensible Space: Crime Prevention Through "
                   "Urban Design. Macmillan.",
    "jacobs_1961": "Jacobs, J. (1961). The Death and Life of Great American Cities. "
                   "Random House.",
    "appleton_1975": "Appleton, J. (1975). The Experience of Landscape. Wiley.",
    "fisher_nasar_1992": "Fisher, B. S., & Nasar, J. L. (1992). Fear of crime in "
                         "relation to three exterior site features: Prospect, refuge "
                         "and escape. Environment and Behavior, 24(1), 35-65.",
    "ewing_handy_2009": "Ewing, R., & Handy, S. (2009). Measuring the unmeasurable: "
                        "Urban design qualities related to walkability. Journal of "
                        "Urban Design, 14(1), 65-84.",
    "welsh_farrington_2008": "Welsh, B. C., & Farrington, D. P. (2008). Effects of "
                             "improved street lighting on crime. Campbell Systematic "
                             "Reviews, 4(1).",
    "kuo_sullivan_2001": "Kuo, F. E., & Sullivan, W. C. (2001). Environment and crime "
                         "in the inner city: Does vegetation reduce crime? "
                         "Environment and Behavior, 33(3), 343-367.",
    "michael_hull_1994": "Michael, S. E., & Hull, R. B. (1994). Effects of vegetation "
                         "on crime in urban parks. Univ. of Illinois / Virginia Tech.",
    "li_2015_gvi": "Li, X., Zhang, C., Li, W., Ricard, R., Meng, Q., & Zhang, W. "
                   "(2015). Assessing street-level urban greenery using Google Street "
                   "View and a modified green view index. Urban Forestry & Urban "
                   "Greening, 14(3), 675-685.",
    "salesses_2013": "Salesses, P., Schechtner, K., & Hidalgo, C. A. (2013). The "
                     "collaborative image of the city: Mapping the inequality of urban "
                     "perception. PLoS ONE, 8(7), e68400.",
    "naik_2014_streetscore": "Naik, N., Philipoom, J., Raskar, R., & Hidalgo, C. "
                             "(2014). Streetscore: Predicting the perceived safety of "
                             "one million streetscapes. CVPR Workshops.",
    "dubey_2016": "Dubey, A., Naik, N., Parikh, D., Raskar, R., & Hidalgo, C. A. "
                  "(2016). Deep learning the city: Quantifying urban perception at a "
                  "global scale. ECCV.",
    "ordonez_berg_2014": "Ordonez, V., & Berg, T. L. (2014). Learning high-level "
                         "judgments of urban perception. ECCV.",
    "zhang_2018": "Zhang, F., Zhou, B., Liu, L., Liu, Y., Fung, H. H., Lin, H., & "
                  "Ratti, C. (2018). Measuring human perceptions of a large-scale "
                  "urban region using machine learning. Landscape and Urban Planning, "
                  "180, 148-160.",
    "koh_2020_cbm": "Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., "
                    "Kim, B., & Liang, P. (2020). Concept Bottleneck Models. ICML.",
    "radford_2021_clip": "Radford, A., et al. (2021). Learning transferable visual "
                         "models from natural language supervision. ICML.",
    "liu_2024_gdino": "Liu, S., et al. (2024). Grounding DINO: Marrying DINO with "
                      "grounded pre-training for open-set object detection. ECCV.",
    "cheng_2024_yoloworld": "Cheng, T., et al. (2024). YOLO-World: Real-time "
                            "open-vocabulary object detection. CVPR.",
    "ravi_2024_sam2": "Ravi, N., et al. (2024). SAM 2: Segment anything in images and "
                      "videos. arXiv:2408.00714.",
    "darcet_2024_registers": "Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. "
                             "(2024). Vision Transformers Need Registers. ICLR.",
}

CONCEPTS = [
    dict(concept="vegetation", name_es="vegetación", name_en="vegetation",
         family="naturaleza",
         definition="Cobertura vegetal visible a nivel de calle: arbolado, setos, "
                    "jardineras y césped. Se mide como fraccion del encuadre, no "
                    "como presencia binaria, porque la literatura discute la "
                    "CANTIDAD, no la mera existencia.",
         grounding="La vegetacion urbana es uno de los correlatos mas estudiados de "
                   "la percepcion del entorno. Kuo & Sullivan encuentran menos "
                   "crimen reportado con mas vegetacion cuidada; Michael & Hull "
                   "advierten lo contrario cuando la vegetacion densa reduce la "
                   "visibilidad. El Green View Index de Li et al. es el "
                   "procedimiento estandar para medirla sobre street view.",
         refs=["kuo_sullivan_2001", "michael_hull_1994", "li_2015_gvi"],
         observable="fraccion de pixeles de clase vegetacion/arbol en el encuadre",
         tool="segmentacion semantica (o SAM 2 guiado por texto)",
         output="fraccion_area [0,1]", polarity="ambiguo",
         risks="Signo contradictorio en la literatura: vegetacion cuidada vs "
               "vegetacion que bloquea la vista. Sensible a estacion y a hora."),
    dict(concept="tree_canopy", name_es="árboles", name_en="tree canopy",
         family="naturaleza",
         definition="Arboles de porte alto con copa visible, distinguidos de la "
                    "vegetacion baja porque su efecto sobre la visibilidad y sobre "
                    "la sensacion de cuidado es distinto.",
         grounding="Se separa de `vegetation` precisamente para poder testear si el "
                   "efecto ambiguo de la vegetacion se debe al porte: copa alta "
                   "mantiene la vision a nivel de ojo, seto denso no.",
         refs=["li_2015_gvi", "michael_hull_1994"],
         observable="deteccion de instancias 'tree' + fraccion de area de copa",
         tool="Grounding DINO / YOLO-World + SAM 2 para el area",
         output="conteo + fraccion_area", polarity="ambiguo",
         risks="Solapa con `vegetation`; hay que reportar su correlacion para no "
               "introducir features redundantes en el bottleneck."),
    dict(concept="visual_openness", name_es="apertura visual", name_en="visual openness",
         family="prospect-refuge",
         definition="Cuanto campo visual abierto ofrece la escena. Proxy operativo: "
                    "fraccion de cielo visible y profundidad del eje de la calle.",
         grounding="Teoria prospect-refuge: se prefiere poder ver sin ser visto. "
                   "Fisher & Nasar miden empiricamente prospecto, refugio y escape "
                   "como predictores del miedo al crimen. Ewing & Handy "
                   "operacionalizan 'enclosure' como cualidad de diseno medible.",
         refs=["appleton_1975", "fisher_nasar_1992", "ewing_handy_2009"],
         observable="fraccion de pixeles de cielo; ratio altura/anchura de la calle",
         tool="segmentacion semantica (clase sky/building)",
         output="fraccion_area [0,1]", polarity="positivo_esperado",
         risks="El encuadre de street view fija en parte el cielo visible; hay que "
               "controlar por heading y por camara."),
    dict(concept="sightline_obstruction", name_es="visibilidad obstruida",
         name_en="sightline obstruction", family="prospect-refuge",
         definition="Presencia de elementos que bloquean la linea de vision a "
                    "distancia media (muros ciegos, vehiculos grandes, vegetacion "
                    "densa a la altura de la vista, contenedores).",
         grounding="Complemento de la apertura visual: el miedo se asocia a lugares "
                   "con escondites potenciales y sin ruta de escape visible "
                   "(refuge para otros, no para el observador).",
         refs=["fisher_nasar_1992", "newman_1972"],
         observable="objetos grandes que ocupan la banda central del encuadre",
         tool="deteccion open-vocabulary + heuristica de posicion/area",
         output="score [0,1]", polarity="negativo_esperado",
         risks="Definicion operativa fragil; requiere validacion humana antes de "
               "usarse como concepto del bottleneck."),
    dict(concept="graffiti", name_es="grafiti", name_en="graffiti",
         family="desorden fisico",
         definition="Pintadas no autorizadas sobre fachadas, muros o mobiliario. "
                    "Se excluye el muralismo autorizado, que la literatura trata "
                    "como senal opuesta (inversion, cuidado).",
         grounding="Senal canonica de desorden fisico en la teoria de broken "
                   "windows: indicios de falta de control informal del espacio.",
         refs=["wilson_kelling_1982", "ordonez_berg_2014"],
         observable="deteccion de regiones con pintada sobre superficie vertical",
         tool="Grounding DINO / YOLO-World (prompt 'graffiti on wall')",
         output="binaria + conteo", polarity="negativo_esperado",
         risks="Distinguir grafiti de mural es dificil para un detector "
               "open-vocabulary; probable fuente de falsos positivos."),
    dict(concept="litter", name_es="basura", name_en="litter / garbage",
         family="desorden fisico",
         definition="Residuos sueltos, bolsas o acumulaciones en via publica, "
                    "excluyendo contenedores en uso normal.",
         grounding="Segundo indicador clasico de desorden fisico de broken windows; "
                   "aparece de forma recurrente en estudios de auditoria urbana.",
         refs=["wilson_kelling_1982", "zhang_2018"],
         observable="deteccion de bolsas/residuos a nivel de suelo",
         tool="Grounding DINO / YOLO-World",
         output="binaria + conteo", polarity="negativo_esperado",
         risks="Objetos pequenos y de baja resolucion en street view; recall bajo "
               "esperable."),
    dict(concept="facade_condition", name_es="deterioro de fachada",
         name_en="facade condition", family="desorden fisico",
         definition="Estado aparente de las fachadas: pintura descascarada, muros "
                    "sin revestir, ventanas rotas o tapiadas, edificios vacantes.",
         grounding="El edificio deteriorado es la imagen literal de broken windows; "
                   "en las auditorias de percepcion es de los predictores mas "
                   "consistentes.",
         refs=["wilson_kelling_1982", "naik_2014_streetscore", "zhang_2018"],
         observable="clasificacion ordinal de la fachada dominante del encuadre",
         tool="CLIP (comparacion entre prompts ordinales) o clasificador entrenado",
         output="ordinal {bueno, regular, deteriorado}", polarity="negativo_esperado",
         risks="Juicio subjetivo incluso para humanos; requiere acuerdo "
               "inter-anotador antes de aceptarse como concepto."),
    dict(concept="security_bars", name_es="rejas", name_en="security bars / grilles",
         family="fortificacion",
         definition="Rejas, barrotes o enrejados sobre ventanas, puertas y frentes "
                    "de local.",
         grounding="Signo de fortificacion defensiva. Newman lo asocia a "
                   "territorialidad; la literatura de percepcion lo lee de forma "
                   "ambivalente: senala proteccion pero tambien senala que hace "
                   "falta protegerse. Es un concepto especialmente pertinente en "
                   "ciudades latinoamericanas, y por tanto para Lima.",
         refs=["newman_1972", "salesses_2013"],
         observable="deteccion de patron de barrotes en aperturas de fachada",
         tool="Grounding DINO / YOLO-World (prompt 'window bars', 'metal grille')",
         output="binaria + conteo", polarity="ambiguo",
         risks="Ambiguedad teorica: es justamente la clase de concepto cuyo signo "
               "debe salir del dato y no del supuesto."),
    dict(concept="blank_wall", name_es="muros ciegos", name_en="blank wall",
         family="fortificacion",
         definition="Superficie de fachada sin puertas ni ventanas a nivel de calle, "
                    "en una extension apreciable del frente.",
         grounding="Ewing & Handy miden 'transparency' (proporcion de frente con "
                   "ventanas/puertas) como cualidad de diseno. Jacobs argumenta que "
                   "el frente ciego elimina los 'ojos en la calle'.",
         refs=["ewing_handy_2009", "jacobs_1961"],
         observable="proporcion del frente a nivel de calle sin aperturas",
         tool="segmentacion + deteccion de ventanas/puertas",
         output="fraccion [0,1]", polarity="negativo_esperado",
         risks="Requiere estimar el frente de calle, no solo contar ventanas."),
    dict(concept="sidewalk", name_es="veredas", name_en="sidewalk",
         family="infraestructura peatonal",
         definition="Presencia y estado aparente de acera transitable: continuidad, "
                    "anchura util, obstrucciones y pavimento roto.",
         grounding="Infraestructura peatonal basica; Ewing & Handy la tratan dentro "
                   "de las cualidades que hacen caminable una calle, y la "
                   "caminabilidad se asocia a presencia de gente.",
         refs=["ewing_handy_2009", "jacobs_1961"],
         observable="segmentacion de la clase acera + heuristica de continuidad",
         tool="segmentacion semantica",
         output="binaria + fraccion_area", polarity="positivo_esperado",
         risks="El encuadre frontal de street view puede no mostrar la acera del "
               "lado opuesto."),
    dict(concept="street_lighting", name_es="iluminación", name_en="street lighting",
         family="infraestructura",
         definition="Presencia de infraestructura de alumbrado publico (postes y "
                    "luminarias). NO es una medida de iluminacion real.",
         grounding="La revision sistematica de Welsh & Farrington documenta efecto "
                   "del alumbrado sobre el crimen; el miedo nocturno esta "
                   "fuertemente ligado a la luz.",
         refs=["welsh_farrington_2008"],
         observable="deteccion de postes y luminarias",
         tool="Grounding DINO / YOLO-World (prompt 'street lamp')",
         output="conteo", polarity="positivo_esperado",
         risks="LIMITACION FUERTE: Place Pulse y el corpus de Lima son imagenes "
               "DIURNAS. Solo puede medirse la infraestructura, nunca la "
               "iluminacion efectiva. Debe reportarse asi de explicito."),
    dict(concept="pedestrians", name_es="peatones", name_en="pedestrians",
         family="actividad humana",
         definition="Personas visibles en via publica en el encuadre.",
         grounding="'Eyes on the street' de Jacobs: la presencia de otras personas "
                   "es el mecanismo central de vigilancia informal y uno de los "
                   "correlatos mas citados de sensacion de seguridad.",
         refs=["jacobs_1961", "naik_2014_streetscore", "zhang_2018"],
         observable="deteccion de personas",
         tool="YOLO-World / Grounding DINO (clase estandar, alta fiabilidad)",
         output="conteo", polarity="positivo_esperado",
         risks="Street view desenfoca caras y capta un instante: la ausencia de "
               "peatones en una foto no implica calle vacia. Confunde hora del dia "
               "con caracteristica del lugar."),
    dict(concept="vehicles", name_es="vehículos", name_en="vehicles",
         family="actividad humana",
         definition="Automoviles, motos, buses y camiones visibles, estacionados o "
                    "en circulacion.",
         grounding="Indicador de actividad y, en parte, de nivel socioeconomico del "
                   "entorno; tambien puede obstruir la linea de vision.",
         refs=["naik_2014_streetscore", "ordonez_berg_2014"],
         observable="deteccion de vehiculos",
         tool="YOLO-World (clases estandar COCO, muy fiable)",
         output="conteo", polarity="ambiguo",
         risks="Mezcla dos mecanismos opuestos (actividad vs obstruccion); "
               "conviene separar 'estacionado' de 'en circulacion' si se puede."),
    dict(concept="commercial_activity", name_es="comercio", name_en="commercial activity",
         family="actividad humana",
         definition="Frentes comerciales activos: vitrinas, toldos, rotulos, puestos.",
         grounding="El comercio a pie de calle genera vigilancia natural y actividad "
                   "sostenida (Jacobs); Ewing & Handy lo recogen en 'transparency' y "
                   "'complexity'.",
         refs=["jacobs_1961", "ewing_handy_2009"],
         observable="deteccion de rotulos/escaparates a nivel de calle",
         tool="Grounding DINO / YOLO-World + CLIP para el tipo",
         output="binaria + conteo", polarity="positivo_esperado",
         risks="Dificil distinguir local abierto de local cerrado o clausurado, que "
               "es justamente la distincion relevante."),
    dict(concept="vacant_lot", name_es="espacios vacíos", name_en="vacant lot",
         family="desorden fisico",
         definition="Terreno baldio, solar sin edificar o predio abandonado visible "
                    "desde la calle.",
         grounding="El lote vacante es un indicador de abandono muy usado en "
                   "auditorias de desorden y en la literatura de broken windows.",
         refs=["wilson_kelling_1982", "newman_1972"],
         observable="segmentacion de suelo sin uso + ausencia de edificacion",
         tool="CLIP (prompt contrastivo) + segmentacion",
         output="binaria", polarity="negativo_esperado",
         risks="Confundible con obra en construccion, que tiene lectura opuesta."),
    dict(concept="street_signage", name_es="señales", name_en="street signage",
         family="infraestructura",
         definition="Senalizacion vial y de orientacion (no publicitaria).",
         grounding="Indicador de mantenimiento institucional y legibilidad del "
                   "espacio; 'imageability' en el marco de Ewing & Handy.",
         refs=["ewing_handy_2009"],
         observable="deteccion de senales de trafico",
         tool="YOLO-World (clase estandar)",
         output="conteo", polarity="positivo_esperado",
         risks="Senal debil; puede no aportar informacion mas alla de tipo de via."),
    dict(concept="street_furniture", name_es="mobiliario urbano",
         name_en="street furniture", family="infraestructura",
         definition="Bancos, paraderos, papeleras, bolardos, jardineras.",
         grounding="Evidencia de inversion publica y de uso estancial del espacio; "
                   "se asocia a mantenimiento percibido.",
         refs=["ewing_handy_2009", "zhang_2018"],
         observable="deteccion de mobiliario",
         tool="Grounding DINO / YOLO-World",
         output="conteo", polarity="positivo_esperado",
         risks="Categoria heterogenea; puede convenir desagregarla en v1."),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "v0",
        "date": str(date.today()),
        "purpose": "Conceptos CANDIDATOS para un Concept Bottleneck sobre percepcion "
                   "urbana de seguridad. Cada polaridad es una HIPOTESIS a testear "
                   "contra el voto humano de Place Pulse, no un supuesto.",
        "method_note": "Los conceptos se agrupan en cinco familias teoricas: "
                       "desorden fisico (broken windows), prospect-refuge, "
                       "fortificacion / territorialidad, actividad humana "
                       "(eyes on the street) e infraestructura / mantenimiento. "
                       "Un concepto entra al codebook solo si (a) tiene anclaje en "
                       "la literatura y (b) es plausible medirlo sobre una unica "
                       "imagen de street view diurna.",
        "provenance_limitation": "Los PDF del Drive no estan disponibles en esta "
                                 "maquina; las referencias son a la literatura "
                                 "canonica del area y deben verificarse contra las "
                                 "copias del Drive antes de citarse en la tesis. "
                                 "Ninguna entrada tiene citation_verified=true.",
        "n_concepts": len(CONCEPTS),
        "references": REFERENCES,
        "concepts": [{**c, "citation_verified": False} for c in CONCEPTS],
    }
    (OUT / "codebook_v0.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fam: dict[str, list] = {}
    for c in CONCEPTS:
        fam.setdefault(c["family"], []).append(c)

    L = ["# Codebook urbano v0", "",
         f"Version v0 · {date.today()} · {len(CONCEPTS)} conceptos candidatos", "",
         "## Que es este documento", "",
         payload["purpose"], "", payload["method_note"], "",
         "> **No se afirma que ningun concepto CAUSE inseguridad.** La columna "
         "*polaridad* recoge el signo que la literatura hace esperar; es la "
         "hipotesis que el experimento debe confirmar o refutar contra el voto "
         "humano `safe` de Place Pulse.", "",
         "> **Limitacion de provenance.** " + payload["provenance_limitation"], "",
         "## Resumen", "",
         "| Familia | Conceptos |", "|---|---|"]
    for f, cs in fam.items():
        L.append(f"| {f} | {', '.join(c['name_es'] for c in cs)} |")
    L += ["", "## Conceptos", ""]

    for c in CONCEPTS:
        L += [f"### `{c['concept']}` — {c['name_es']} ({c['name_en']})", "",
              f"- **Familia**: {c['family']}",
              f"- **Definicion operativa**: {c['definition']}",
              f"- **Anclaje**: {c['grounding']}",
              f"- **Como observarlo**: {c['observable']}",
              f"- **Herramienta candidata**: {c['tool']}",
              f"- **Salida**: `{c['output']}`",
              f"- **Polaridad esperada (hipotesis)**: `{c['polarity']}`",
              f"- **Riesgos / limitaciones**: {c['risks']}",
              f"- **Referencias**: " + "; ".join(REFERENCES[r] for r in c["refs"]),
              ""]

    L += ["## Referencias", ""]
    for k, v in sorted(REFERENCES.items()):
        L.append(f"- [{k}] {v}")
    L += ["", "## Siguiente paso", "",
          "El codebook no se aplica masivamente hasta validar que los conceptos se "
          "pueden medir de forma reproducible. Ese es el objetivo del "
          "proof-of-concept (fase D): comparar herramientas sobre una muestra "
          "pequena y diversa, y decidir que conceptos sobreviven a v1.", ""]

    (OUT / "CODEBOOK_V0.md").write_text("\n".join(L), encoding="utf-8")
    log(f"codebook v0: {len(CONCEPTS)} conceptos, {len(fam)} familias, "
        f"{len(REFERENCES)} referencias")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
