# Reproducibility Gaps — Perception

Levantado durante la auditoría y reproducción local (ver `RUNBOOK_LOCAL.md`),
commit base `f2c77b0` en `main`. No incluye ningún cambio de código —
solo diagnóstico.

---

## 1. Dependencias no declaradas / sin pin de versión

| Gap | Detalle | Riesgo |
|---|---|---|
| `graph/graph-requirements.txt` sin `scikit-learn` | El propio README (paso 1: dual-GMM) y `graph_pipeline.py` importan `sklearn.mixture.GaussianMixture`, pero no está en `graph-requirements.txt`. Hay que instalarlo aparte. | Bajo (documentado en el prompt maestro), pero el `requirements.txt` real está incompleto. |
| Sin pins de versión (`>=` en vez de `==`) | `graph-requirements.txt` y `distritos/requirements.txt` usan `>=X.Y.Z` para todo. En esta corrida se instaló `osmnx==2.0.7`, cuando el código (comentarios, estilo de API) sugiere que fue escrito contra osmnx 1.x. | Medio: osmnx 2.x cambió APIs internas en versiones pasadas; hoy corrió sin error, pero no hay garantía de reproducibilidad numérica exacta a futuro si PyPI publica versiones nuevas. |
| `extract_embeddings.py` requiere `torch`, `torchvision`, `timm`, `tqdm`, `pillow` | Ninguna de estas está en `graph-requirements.txt`. Solo aparecen mencionadas sueltas en el README bajo "Python Packages". | Alto para quien solo siga `graph-requirements.txt` literalmente — fallará en el Paso 2 sin previo aviso. |
| README "Requirements" es genérico | Dice `pip install torch torchvision timm scikit-learn pandas networkx pillow matplotlib numpy` — no menciona `osmnx`, `geopandas`, `scipy`, `contextily`, que sí son necesarios para Paso 1. | Medio — dos listas de dependencias inconsistentes entre sí (README vs `graph-requirements.txt`). |
| CUDA / GPU | README dice "CUDA-capable GPU (para Steps 2)" pero no hay forma de verificar sin instalar torch, y no se ha instalado (Fase 4 bloqueada). | Bajo por ahora, relevante cuando lleguen los assets visuales. |

## 2. Rutas inconsistentes

| Gap | Detalle |
|---|---|
| `graph_output_v2` vs `graph_output` | `graph_pipeline.py` (línea 56) escribe a `graph_output_v2/`. Todos los scripts posteriores (`Paso4`–`Paso9`) tienen `--graph-dir` con `default="graph_output"` (sin `_v2`). Confirmado leyendo el código, no asumido. **Cada ejecución de Pasos 4–9 necesita `--graph-dir graph_output_v2` explícito** o fallará silenciosamente buscando una carpeta que no existe (o, peor, una vieja carpeta `graph_output` residual de una corrida anterior con datos distintos). |
| `.gitignore` solo excluye `/graph/graph_output` | No excluye `/graph/graph_output_v2`, que es la carpeta real usada — por eso sí quedó versionada en git. Esto es (probablemente) un descuido del `.gitignore`, no intencional. |
| `embeddings_export/metadata.json` con rutas absolutas de otra máquina | Contiene paths tipo `/home/ricardo.acuna/Perception/research_pt2/image-extraction/...`. Si se reemplaza el `embeddings.npy`/`metadata.json` por uno nuevo generado en Windows, las rutas serán distintas (`C:\...`) — cualquier script que dependa de esas rutas absolutas (en vez de solo `crime_id`/`heading`) se rompería al cambiar de máquina. Falta confirmar en el código de `Paso4` si usa las rutas absolutas o solo `crime_id`. |
| `README.md` describe una estructura `research_pt2/` | El repo real no tiene ese prefijo de carpeta (`distritos/`, `graph/`, etc. están en la raíz). El README fue escrito asumiendo un layout de carpetas distinto al del repo clonado. |

## 3. Assets faltantes (bloquean Fases 4–9)

| Asset | Estado | Bloquea |
|---|---|---|
| `checkpoints_simclr_vit/simclr_perception_224x_224_best_epoch_50.pt` | Ausente en todo el disco | Paso 2 (`extract_embeddings.py`) |
| `image-extraction/Inseguros-Barranco-GGZ-2016/`, `Inseguros-La_Victoria-GGZ-2016/` | Ausentes | Paso 2, Paso 9 |
| `graph/embeddings_export/embeddings.npy` | Ausente (solo `metadata.json` versionado) | Paso 3 |
| `graph/neighbourhood_tensors/*.npy` | Ausente (solo `coverage_report.csv` y `node_ids_order.json` versionados) | Paso 4 |
| `graph/eigen_features/*_features_mean_std.npy`, `*_labels.npy` | Ausente (solo `*_strategy_comparison.csv` versionado) | **Paso 5 — confirmado por código (`Paso6_final_classifier.py:194`) que hace `np.load(...)` sin fallback, sin modo structural-only disponible por CLI** |
| `graph/final_classifier/*_predictions.pkl` | Ausente (solo `*_classifier_comparison.csv` versionado) | Paso 6 (Paso7 diagnostic plots) |

Consecuencia: **no hay ningún camino actual para ejecutar Pasos 2–9 sin
conseguir al menos uno de estos assets** (idealmente `embeddings.npy` +
`metadata.json` coherentes entre sí, que evitarían necesitar el checkpoint
y las imágenes crudas).

## 4. Secretos / configuración sensible

| Gap | Detalle |
|---|---|
| `image-extraction/downloadImage.py` requiere `.env` con `GOOGLE_STREETVIEW_API_KEY` y `MAPILLARY_ACCESS_TOKEN` | No existe `.env.example` en el repo — nadie sabe qué variables declarar sin leer el código fuente. No se ejecutó este script (regla no negociable). |
| No hay `.env.example` para ningún otro paso | `distritos/run.py` sí lee `RUTA_CSV` y `OUTPUT_DIR_BASE` de entorno (con default razonable), pero tampoco está documentado en ningún `.env.example`. |

## 5. Pasos manuales no documentados

| Gap | Detalle |
|---|---|
| Activación de venv en PowerShell | Puede requerir `Set-ExecutionPolicy` si la política de ejecución bloquea `Activate.ps1` — no documentado en README. |
| Selección de intérprete Python 3.10 | El sistema tiene por defecto Python 3.12 (`py -0p` → `*` en 3.12), que no fue probado contra `graph-requirements.txt`; hay que forzar `py -3.10` explícitamente. No documentado en README. |
| Encoding de consola Windows | `graph_pipeline.py --help` (y probablemente otros `--help`, no verificado uno por uno) truncan con `UnicodeEncodeError` en cp1252 por el carácter `→` en los help strings. Solución: `$env:PYTHONIOENCODING="utf-8"` antes de correr cualquier script. No documentado en README, específico de Windows. |
| Ejecución distrito por distrito si `--distrito all` no aplica en algún script | Aún no verificado en todos los Pasos 4–9 individualmente (bloqueados por assets), pendiente de confirmar cuando lleguen los assets. |

## 6. Outputs no versionados que sí deberían preservarse

Según la propia sección "Reproducibility Notes" del README, estos son
"esenciales para reproducibilidad" pero están en `.gitignore` y no se
subieron nunca:

- `embeddings_export/embeddings.npy` + `metadata.json` (30,866 embeddings esperados)
- `neighbourhood_tensors/*_neighbourhood_tensors.npy`
- `final_classifier/*_predictions.pkl`

Esto crea una contradicción interna en el propio README: dice que son
"esenciales para preservar" pero el `.gitignore` los excluye por completo
y no hay un mecanismo alternativo (release de GitHub, storage externo,
DVC/git-lfs) para distribuirlos junto al código.

## 7. Discrepancias numéricas menores vs. README (no bloqueantes)

Confirmadas al reproducir Paso 1 (ver `RUNBOOK_LOCAL.md` para detalle):

| Métrica | Reproducido | README | Causa probable |
|---|---|---|---|
| Barranco nodos | 401 | 400 | Drift de OSM (base de datos viva) |
| Barranco % peligroso | 42.1% | 42.2% | Consecuencia del drift anterior |
| La Victoria nodos | 1,456 | 1,450 | Drift de OSM |
| La Victoria % peligroso | 11.5% | 11.2% | Consecuencia del drift anterior |

No se considera un bug: el pipeline es determinista dado el mismo snapshot
de OSM (semilla fija en KMeans/GMM), pero OSM mismo cambia con el tiempo y
el pipeline no fija una fecha/versión de snapshot (no usa `overpass` con
timestamp fijo ni un extracto `.osm.pbf` congelado). Para reproducibilidad
exacta a largo plazo, habría que congelar el snapshot de OSM usado
(ej. descargar y versionar un extracto `.osm.pbf` de la fecha original).

---

## Resumen priorizado (qué resolver primero)

1. **Bloqueante crítico**: conseguir `embeddings.npy` (+ `metadata.json` coherente) o checkpoint + imágenes — sin esto, el 90% del pipeline no es evaluable.
2. **Alto impacto, bajo esfuerzo**: agregar `--graph-dir graph_output_v2` como ejemplo explícito en el README/RUNBOOK para Pasos 4–9 (no cambiar el default en código todavía sin autorización).
3. **Alto impacto, bajo esfuerzo**: completar `graph-requirements.txt` con `scikit-learn`, `torch`, `torchvision`, `timm`, `tqdm` (o separar en `graph-requirements-visual.txt`).
4. **Medio**: crear `.env.example` para `image-extraction/downloadImage.py` y `distritos/run.py`.
5. **Medio**: decidir si se congela un snapshot de OSM para reproducibilidad exacta, o se acepta el drift documentado como limitación conocida.
6. **Bajo**: documentar el workaround de `PYTHONIOENCODING=utf-8` para Windows en el README.

No se ha aplicado ningún cambio de código todavía — este archivo es
diagnóstico puro, a la espera de tu autorización para pasar a la Fase 10
(normalización mínima / Docker) o para corregir cualquiera de estos puntos.
