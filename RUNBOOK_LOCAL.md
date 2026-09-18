# Reproducción local — Perception

Registro en vivo de la reproducción local del pipeline, siguiendo
`Perception_Prompt_Maestro_Claude_Code.md`. No se modificó `main`, no se
hizo push, no se borró ningún archivo. Todo cambio de código propuesto
requiere autorización explícita antes de aplicarse.

## Entorno

| Campo | Valor |
|---|---|
| OS | Windows 11 Home, build 10.0.26200 |
| RAM | 24,299 MB |
| CPU | No verificado (wmic no disponible en este shell) |
| GPU/CUDA | No verificado aún — torch no instalado (rama visual bloqueada, ver abajo) |
| Python (venv) | 3.10.2 (`C:\Users\afp20\AppData\Local\Programs\Python\Python310\python.exe`) vía `.venv` en la raíz del repo |
| Commit SHA | `f2c77b019bc220d81d8ae085ded63eca55900bd5` |
| Branch | `main` (sin cambios locales de commits; solo working tree modificado por la corrida, ver Paso 2) |

## Dependencias instaladas (Fase 1)

Comando:
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r graph\graph-requirements.txt
pip install scikit-learn
```

Resultado: instalación exitosa. `pip freeze` completo:

```
affine==3.0.1
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
click-plugins==1.1.1.2
cligj==0.7.2
cloudpickle==3.1.2
contextily==1.7.1
contourpy==1.3.2
cycler==0.12.1
fonttools==4.65.0
geographiclib==2.1
geopandas==1.1.4
geopy==2.5.0
idna==3.19
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.10.9
mercantile==1.2.1
networkx==3.4.2
numpy==2.2.6
osmnx==2.0.7
packaging==26.3
pandas==2.3.3
pillow==12.3.0
pyogrio==0.13.0
pyparsing==3.3.2
pyproj==3.7.1
python-dateutil==2.9.0.post0
pytz==2026.3.post1
rasterio==1.4.4
requests==2.34.2
scikit-learn==1.7.2
scipy==1.15.3
shapely==2.1.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2026.3
urllib3==2.7.0
xyzservices==2026.9.1
```

Smoke test:
```powershell
python -c "import osmnx, networkx, geopandas, pandas, numpy, scipy, sklearn; print('IMPORTS_OK')"
```
→ `IMPORTS_OK`

**Nota**: `graph-requirements.txt` no fija versiones (`>=`), por lo que se
instalaron las últimas disponibles en PyPI (osmnx 2.0.7, mientras que
`graph_pipeline.py` fue escrito probablemente contra osmnx 1.x — ver
`REPRODUCIBILITY_GAPS.md`).

## Assets disponibles (Fase 3, ver detalle completo en `REPRODUCIBILITY_GAPS.md`)

- ✅ `graph_output_v2/` (grafos, nodos, edges, crímenes snapeados, plots) — versionado en git, ahora regenerado.
- ✅ `distritos/outputs/`, `image-extraction/outputs/` (clusters GMM 2016–2023) — versionado.
- ✅ `graph/embeddings_export/metadata.json` (sin el `.npy`).
- ❌ `embeddings.npy`, checkpoint SimCLR+ViT (`.pt`), carpetas `Inseguros-*-GGZ-2016/`, `neighbourhood_tensors/*.npy`, `eigen_features/*.npy`, `final_classifier/predictions.pkl` — **FALTANTES**, todos excluidos por `.gitignore` y nunca subidos al repo. Bloquean Fases 4–9 completas (rama visual).

## Paso 1 — Auditoría de código (Fase 0)

Comando:
```powershell
git status; git branch --show-current; git rev-parse HEAD; git log -1 --oneline; git remote -v
```
Resultado: branch `main`, limpio salvo el prompt maestro (untracked), remote `ricardoamiel/Perception`.

Hallazgo confirmado en código: **mismatch de rutas real**. `graph_pipeline.py`
(línea 56) define `OUTPUT_DIR = Path("graph_output_v2")`, pero
`Paso4_assemble_neighbourhood_tensors.py`, `Paso5_eigen_features.py`,
`Paso6_final_classifier.py`, `Paso7_diagnostic_plots.py`,
`Paso8_robustness_checks.py` y `Paso9_qualitative_samples.py` tienen todos
`--graph-dir` con `default="graph_output"` (sin `_v2`). Cada ejecución de
Pasos 4–9 necesita `--graph-dir graph_output_v2` explícito.

`image-extraction/downloadImage.py` usa `.env` con `GOOGLE_STREETVIEW_API_KEY`
y `MAPILLARY_ACCESS_TOKEN`. **No se ejecutó** (regla no negociable #5).

## Paso 2 — `graph_pipeline.py` (Fase 2)

Inspección de CLI:
```powershell
cd graph
$env:PYTHONIOENCODING="utf-8"
python graph_pipeline.py --help
```
(Se requirió `PYTHONIOENCODING=utf-8` porque el help text del script contiene
el carácter `→`, que la consola Windows cp1252 no puede codificar por
defecto — error `UnicodeEncodeError`. No es un bug del pipeline, es un
problema de encoding de terminal. No se modificó código.)

Confirmado por `--help`:
- `--dual-gmm` existe y es el flag documentado en el prompt.
- `--csv` default = `../distritos/crimen_distritos_seleccionados.csv`.
- `--output` default = `graph_output_v2` (confirma el mismatch de rutas).
- `--distrito` default = `all`; `--year` default = `2016`.

Comando ejecutado:
```powershell
cd graph
$env:PYTHONIOENCODING="utf-8"
python graph_pipeline.py --dual-gmm --year 2016
```

Resultado: **sin errores ni tracebacks**. Procesó Barranco y La Victoria.

### Salida — Barranco 2016
- Grafo OSM: 401 nodos, 729 aristas (dirigido original de osmnx).
- Crímenes cargados: 1,400 → 3 descartados (>150m) → 1,397 mapeados a nodos.
- Nodos con ≥1 crimen: 272/401.
- GMM dual: τ_nodo=6.8 (70/401 peligrosos, 17.5%); τ_vecindario=14.9 →
  **169/401 peligrosos (42.1%)** — label primario.
- Vecindarios exportados: 272.

### Salida — La Victoria 2016
- Grafo OSM: 1,456 nodos, 3,090 aristas.
- Crímenes cargados: 3,807 → 35 descartados (>150m) → 3,772 mapeados.
- Nodos con ≥1 crimen: 809/1,456.
- GMM dual: τ_nodo=8.5 (103/1,456, 7.1%); τ_vecindario=22.5 →
  **168/1,456 peligrosos (11.5%)** — label primario.
- Vecindarios exportados: 809.

### Outputs generados (en `graph/graph_output_v2/`)
| Archivo | Descripción |
|---|---|
| `{Distrito}_2016_crimes_snapped.csv` | crímenes ↔ nodo asignado |
| `{Distrito}_2016_dualgmm_nodes.csv` | nodos + labels (401/1456 filas × 10 cols) |
| `{Distrito}_2016_edges.csv` | aristas del grafo no dirigido (615/2295 filas) |
| `{Distrito}_2016_dualgmm_neighbourhoods.json` | dict de vecindarios (272/809 entradas) |
| `{Distrito}_2016_graph.png`, `_distribution.png`, `_dualgmm_kde_labels.png`, `_dualgmm_neighbourhoods.png` | plots |

### Validaciones realizadas
- `nodes.csv`: shape (401,10) y (1456,10), columnas
  `['node_id','lon','lat','crime_count','neigh_total','label','node_label','neighbourhood_label','threshold','strat_tag']`,
  **0 NaN** en ambos distritos.
- `edges.csv`: 615 filas (Barranco), 2295 filas (La Victoria).
- `crimes_snapped.csv`: 1397 y 3772 filas — coincide con el conteo impreso en consola.
- `neighbourhoods.json`: dict con 272 y 809 entradas — coincide con "vecindarios exportados".
- Distribución de labels: Barranco 42.1% peligroso (README reporta 42.2%);
  La Victoria 11.5% (README reporta 11.2%).

### Comparación contra la versión versionada en git (antes de la corrida)
```powershell
git status --short graph/graph_output_v2/
git diff --stat graph/graph_output_v2/
```
Todos los archivos de `graph_output_v2/` cambiaron (`git diff --stat`: 15
archivos, ~8,400 líneas +/-). El diff es **parcial, no total**: por ejemplo
`Barranco_2016_edges.csv` cambió solo 36 de ~616 líneas, y
`Barranco_2016_dualgmm_nodes.csv` cambió solo 2 líneas. Esto es consistente
con no-determinismo esperado de OSM (base de datos viva que recibe ediciones
continuas), no con un fallo del pipeline: los conteos de nodos/aristas y el
% de labels son casi idénticos a los documentados en el README (diferencias
de 0.1–0.3 puntos porcentuales).

**⚠️ Importante**: esta corrida sobrescribió los archivos versionados en
`graph/graph_output_v2/` en el working tree local. No se hizo `git add`,
`git commit` ni `git push` — los cambios son visibles solo con `git status`/
`git diff` y son reversibles con `git checkout -- graph/graph_output_v2/`
si se prefiere restaurar la versión original committeada.

### Problemas encontrados
- Ninguno bloqueante. Único inconveniente: encoding de consola (resuelto con
  `PYTHONIOENCODING=utf-8`, sin tocar código).
- `graph-requirements.txt` sin pins de versión → se instaló osmnx 2.0.7
  (posible mismatch de mayor versión vs. lo usado originalmente). El pipeline
  corrió sin errores con esta versión, pero no se puede garantizar
  reproducibilidad numérica exacta a largo plazo sin pins.

### Siguiente paso
Fase 3 (inventario de assets visuales) ya adelantada durante Fase 0 — rama
visual confirmada **BLOQUEADA** (faltan checkpoint, imágenes y
`embeddings.npy` simultáneamente). Ver `REPRODUCIBILITY_GAPS.md` para el
detalle y qué se necesita solicitar a Ricardo/Rensso antes de continuar con
Fases 4–9.

## Resultados reproducidos hasta ahora

| Métrica | Reproducido | README | Diferencia |
|---|---|---|---|
| Barranco nodos | 401 | 400 | +1 (drift OSM) |
| Barranco % peligroso | 42.1% | 42.2% | −0.1 pp |
| La Victoria nodos | 1,456 | 1,450 | +6 (drift OSM) |
| La Victoria % peligroso | 11.5% | 11.2% | +0.3 pp |

## Diferencias frente al README

- El README documenta `graph_output/` como carpeta de salida; el código real
  usa `graph_output_v2/`. Los Pasos 4–9 no saben esto por defecto.
- Los conteos de nodos y % de labels difieren levemente (ver tabla arriba),
  atribuible a que OSM es una base de datos viva.

## Bloqueadores

1. **Rama visual completa bloqueada**: faltan `checkpoint .pt`, imágenes
   Street View (`Inseguros-*-GGZ-2016/`) y `embeddings.npy`. Sin al menos
   uno de estos disponible, Fases 4, 5 (parcial), 6 (fused/visual), 7, 8
   (parcial) y 9 no pueden ejecutarse honestamente.
2. `image-extraction/downloadImage.py` no debe usarse para regenerar estos
   assets (regla no negociable #5 — revisión de permisos/licencia pendiente).
