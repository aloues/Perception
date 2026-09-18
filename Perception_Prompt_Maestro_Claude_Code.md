# PROMPT MAESTRO PARA CLAUDE CODE — PERCEPTION
## Objetivo: auditar, ejecutar localmente, reproducir y después contenerizar el proyecto sin alterar la lógica científica

Estás trabajando dentro de Visual Studio Code en Windows 11. El repositorio `Perception` ya está clonado y abierto como carpeta raíz.

Repositorio de referencia:
https://github.com/ricardoamiel/Perception

Tu tarea NO es reescribir el proyecto ni "mejorarlo" de inmediato. Primero debes conseguir que el pipeline actual sea comprensible, reproducible y verificable localmente. Después, y solo después de una corrida local satisfactoria, preparar un contenedor reproducible.

## REGLAS NO NEGOCIABLES

1. NO modifiques `main`, NO hagas push y NO borres archivos.
2. Antes de editar cualquier archivo, muestra qué vas a cambiar y por qué.
3. No inventes datos, checkpoints, imágenes, outputs ni variables de entorno faltantes.
4. No declares un paso "funcionando" solo porque el proceso terminó: valida archivos, shapes, conteos, NaN y resultados.
5. NO ejecutes `image-extraction/downloadImage.py` ni hagas nuevas descargas masivas de Google Street View. Existe una revisión pendiente sobre permisos/licencia. Usa únicamente assets locales ya disponibles.
6. Si falta un archivo que está excluido por `.gitignore`, detente en ese punto, identifica exactamente qué falta y continúa únicamente con los pasos independientes de ese asset.
7. No cambies algoritmos, thresholds, GMM, features, CV ni modelos para "hacer que funcione". Primero reproduce el comportamiento actual.
8. Mantén un registro en `RUNBOOK_LOCAL.md` de cada comando ejecutado, resultado, error y solución.
9. Todo comando debe estar pensado para PowerShell/Windows.
10. Pregunta antes de instalar software del sistema, CUDA, Docker Desktop o modificar configuración global.

---

# CONTEXTO CIENTÍFICO QUE DEBES RESPETAR

El repositorio actual está orientado principalmente a predecir `crime risk` a partir de:

- registros de crimen;
- OpenStreetMap;
- ego-networks;
- imágenes Street View;
- embeddings SimCLR + ViT;
- features estructurales;
- clasificación visual / estructural / fused;
- validación espacial.

NO asumas que las etiquetas actuales representan percepción humana de seguridad. Actualmente derivan de datos de crimen/GMM. No renombres conceptos científicos sin autorización.

La investigación futura puede estudiar percepción de seguridad, contexto espacial, ego-depth, generalización y MLLM, pero ESO NO ES PARTE DE TU TAREA INICIAL DE REPRODUCCIÓN.

---

# FASE 0 — AUDITORÍA SIN CAMBIOS

Primero ejecuta y reporta:

```powershell
git status
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git remote -v
```

Luego lista la estructura del repo y revisa como mínimo:

- `README.md`
- `.gitignore`
- `graph/graph-requirements.txt`
- `distritos/requirements.txt`
- `graph/graph_pipeline.py`
- `graph/extract_embeddings.py`
- `graph/Paso4_assemble_neighbourhood_tensors.py`
- `graph/Paso5_eigen_features.py`
- `graph/Paso6_final_classifier.py`
- `graph/Paso7_diagnostic_plots.py`
- `graph/Paso8_robustness_checks.py`
- `graph/Paso9_qualitative_samples.py`
- `image-extraction/downloadImage.py`

Genera una tabla:

| Archivo | Función | Inputs | Outputs | Dependencias | ¿Se puede ejecutar con lo que existe? |
|---|---|---|---|---|---|

También identifica todo lo que `.gitignore` excluye y que sea requerido por el pipeline, especialmente:

- `*.pt`
- `*.npy`
- `*.pkl`
- carpetas de imágenes
- `.env`
- outputs del grafo

NO avances a ejecutar hasta mostrarme esta auditoría.

---

# FASE 1 — PYTHON LOCAL

Estamos en Windows.

Primero detecta Python:

```powershell
py -0p
python --version
```

Preferencia: Python 3.10 si está disponible.

Si existe:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell bloquea la activación, NO cambies políticas globales sin preguntar. Propón la alternativa segura.

Después instala SOLO las dependencias necesarias para el Paso 1:

```powershell
pip install -r graph\graph-requirements.txt
pip install scikit-learn
```

Haz smoke test:

```powershell
python -c "import osmnx, networkx, geopandas, pandas, numpy, scipy, sklearn; print('IMPORTS_OK')"
```

Registra versiones:

```powershell
python --version
pip freeze
```

---

# FASE 2 — PRIMERA CORRIDA: GRAFO Y LABELS

Antes de ejecutar, inspecciona el CLI real:

```powershell
cd graph
python graph_pipeline.py --help
```

Confirma:

- si existe `--dual-gmm`;
- cuál es la ruta del CSV;
- cuál es la carpeta de output;
- qué distritos procesa;
- qué año usa.

El código actual observado usa `graph_output_v2`, mientras que scripts posteriores suelen esperar `graph_output`.
NO cambies código todavía.

Ejecuta:

```powershell
python graph_pipeline.py --dual-gmm --year 2016
```

Al terminar valida:

1. Qué archivos creó.
2. Número de nodos y edges por distrito.
3. Número de registros de crimen cargados y snapped.
4. Cuántos fueron descartados por distancia.
5. Distribución de labels.
6. Si existen:
   - nodes CSV
   - edges CSV
   - crimes_snapped CSV
   - neighbourhood JSON/CSV
   - plots
7. Ausencia de errores silenciosos.

Actualiza `RUNBOOK_LOCAL.md`.

Si falla, NO cambies código inmediatamente.
Primero:
- copia el traceback;
- identifica archivo/línea;
- explica la causa;
- propone la corrección mínima;
- espera autorización si implica cambiar lógica.

---

# FASE 3 — INVENTARIO DE ASSETS VISUALES

Vuelve a la raíz y busca, SIN crear ni descargar nada:

```powershell
Get-ChildItem -Recurse -Include *.pt,*.npy,*.pkl | Select-Object FullName,Length
```

Revisa si existen localmente:

- `simclr_perception_224x_224_best_epoch_50.pt`
- `embeddings.npy`
- `metadata.json`
- carpetas `Inseguros-Barranco-GGZ-2016`
- carpetas `Inseguros-La_Victoria-GGZ-2016`
- `neighbourhood_tensors`
- `eigen_features`
- `final_classifier`

Clasifica:

### DISPONIBLE
Asset presente y legible.

### FALTANTE
Está requerido por el código pero no existe.

### NO NECESARIO TODAVÍA
No bloquea el siguiente paso.

Si faltan las imágenes/checkpoint pero existen embeddings, prioriza continuar desde embeddings.

Si faltan imágenes Y embeddings, detén la rama visual y dime exactamente qué debo solicitar a Ricardo/Rensso.

NO uses `downloadImage.py` para reconstruir Google Street View.

---

# FASE 4 — EMBEDDINGS

SOLO si ya existen imágenes locales autorizadas y checkpoint local.

Instala, si faltan:

```powershell
pip install torch torchvision timm pillow
```

Comprueba:

```powershell
python -c "import torch, torchvision, timm; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Inspecciona:

```powershell
cd graph
python extract_embeddings.py --help
```

Ejecuta usando rutas reales.
El comando esperado es conceptualmente:

```powershell
python extract_embeddings.py `
  --ckpt ..\checkpoints_simclr_vit\simclr_perception_224x_224_best_epoch_50.pt `
  --data-path ..\image-extraction `
  --output embeddings_export
```

PERO debes adaptar los argumentos al CLI real del archivo, no asumirlos.

Valida:

- `embeddings.npy`
- `metadata.json`
- shape `(N, 768)` esperada por el pipeline actual
- dtype
- NaN / inf
- cantidad de metadata = N
- distritos
- headings
- IDs
- blacklist aplicada

No continúes si metadata y embeddings no alinean.

---

# FASE 5 — TENSORES DE VECINDARIO

Inspecciona:

```powershell
python Paso4_assemble_neighbourhood_tensors.py --help
```

Hay un posible mismatch de rutas:
- `graph_pipeline.py` → `graph_output_v2`
- pasos siguientes → default `graph_output`

NO renombres carpetas ni cambies código primero.
Pasa la ruta explícita cuando el CLI lo permita:

```powershell
python Paso4_assemble_neighbourhood_tensors.py `
  --distrito all `
  --year 2016 `
  --graph-dir graph_output_v2 `
  --embeddings-dir embeddings_export
```

Si `--distrito all` no está soportado por el script real, corre distrito por distrito.

Valida:

- tensor por distrito;
- shape esperado por vecindario;
- `node_ids_order.json`;
- coverage report;
- depth usado;
- oversampling;
- vecindarios insuficientes;
- IDs sin match.

El código observado usa:
- 5 puntos por vecindario;
- 4 imágenes por punto;
- 20 embeddings por vecindario;
- embedding 768-D;
- profundidad adaptativa hasta MAX_DEPTH=3.

Confirma esto en el código actual antes de reportarlo como hecho.

---

# FASE 6 — FEATURES

Inspecciona:

```powershell
python Paso5_eigen_features.py --help
```

Ejecuta con rutas explícitas:

```powershell
python Paso5_eigen_features.py `
  --distrito all `
  --year 2016 `
  --graph-dir graph_output_v2 `
  --tensors-dir neighbourhood_tensors `
  --embeddings-dir embeddings_export
```

Valida:
- archivos generados;
- `features_mean_std.npy`;
- labels;
- shapes;
- NaN;
- resultados diagnósticos.

Recuerda: Paso 5 es diagnóstico de estrategias de representación, no el clasificador final.

---

# FASE 7 — CLASIFICADOR FINAL

Inspecciona:

```powershell
python Paso6_final_classifier.py --help
```

Ejecuta:

```powershell
python Paso6_final_classifier.py `
  --distrito all `
  --year 2016 `
  --graph-dir graph_output_v2
```

Agrega otros argumentos solo si el CLI real lo requiere.

Debes verificar que compare:

- visual_only;
- structural_only;
- fused;

con:

- Logistic Regression;
- Random Forest;
- MLP;

y que use:

- GroupKFold espacial como esquema primario;
- StratifiedKFold random como referencia.

Outputs esperados:

- `classifier_comparison.csv`
- `predictions.pkl`

Reporta por distrito:
- AUC spatial;
- AUC random;
- gap de leakage;
- F1 default;
- optimal threshold;
- F1 optimal;
- mejor global;
- mejor fused.

NO compares contra los valores del README como si fueran verdad hasta haberlos reproducido.

---

# FASE 8 — VISUALIZACIONES

Inspecciona los CLIs y ejecuta:

```powershell
python Paso7_diagnostic_plots.py --distrito Barranco --year 2016
python Paso7_diagnostic_plots.py --distrito "La Victoria" --year 2016
```

Si requieren `--graph-dir` o `--output`, usa las rutas reales.

Valida visualmente los PNG:
- threshold curves;
- ROC/PR;
- confusion matrix;
- graph real vs predicted;
- error map;
- feature importance.

Luego:

```powershell
python Paso8_robustness_checks.py --distrito all --year 2016
```

Valida:
- spatial leakage;
- threshold stability;
- stacking si corresponde.

Paso 9:

```powershell
python Paso9_qualitative_samples.py --year 2016 --n-per-class 6
```

SOLO si las imágenes locales están presentes.

---

# FASE 9 — REPORTE DE REPRODUCCIÓN

Crea `RUNBOOK_LOCAL.md` con esta estructura:

```markdown
# Reproducción local — Perception

## Entorno
- OS
- Python
- Commit SHA
- Branch
- CPU
- GPU/CUDA

## Dependencias
...

## Assets disponibles
...

## Paso 1
Comando:
Resultado:
Outputs:
Validación:
Problemas:

## Paso 2
...

## Resultados reproducidos
...

## Diferencias frente al README
...

## Bloqueadores
...
```

Genera además un archivo:

`REPRODUCIBILITY_GAPS.md`

con:
- dependencies no declaradas;
- rutas inconsistentes;
- assets faltantes;
- secretos;
- pasos manuales;
- outputs no versionados.

NO arregles todo todavía.

---

# FASE 10 — NORMALIZACIÓN MÍNIMA PARA CONTENEDOR

Solo después de reproducir localmente.

Antes de tocar código, propón un plan de cambios mínimos.

Objetivo:
que otra persona pueda ejecutar el pipeline sin configurar manualmente Python.

Crear, preferentemente en una rama local nueva:

```powershell
git switch -c fabrizio/reproducibility
```

NO hagas push.

Archivos a preparar:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`
- `RUNBOOK.md`
- opcional `scripts/run_pipeline.ps1`

NO copies dentro de la imagen:
- datasets grandes;
- Street View;
- checkpoints;
- API keys;
- `.env`;
- outputs.

Usa volúmenes:

```text
/data
/checkpoints
/outputs
```

El contenedor debe poder recibir rutas por argumento o variables de entorno.

---

# FASE 11 — QUÉ SIGNIFICA "DEPLOY" POR AHORA

Hasta que el profesor Rensso confirme otra cosa, interpreta deploy como:

**pipeline reproducible dentro de Docker**, NO como aplicación web.

Éxito mínimo:

```powershell
docker compose build
docker compose run --rm perception <comando-pipeline>
```

Debe:
1. iniciar;
2. leer datos montados;
3. producir outputs montados;
4. salir sin errores;
5. generar resultados equivalentes a la corrida local.

NO construyas FastAPI, Streamlit, Gradio ni frontend todavía.

Al final pregúntame:

> “El pipeline ya corre local y en contenedor. ¿Rensso requiere además una API/interfaz web o el deploy reproducible era el entregable?”

---

# FORMATO DE RESPUESTA DURANTE EL TRABAJO

En cada fase responde así:

## Fase X — Nombre

**Estado:** OK / BLOQUEADO / ERROR / PENDIENTE

**Lo que revisé**
- ...

**Comandos ejecutados**
```powershell
...
```

**Resultado**
- ...

**Outputs generados**
- ruta
- tamaño
- descripción

**Validaciones**
- ...

**Problemas encontrados**
- ...

**Siguiente paso**
- ...

Si estás BLOQUEADO, no improvises.
Dime exactamente qué archivo, dato, permiso o decisión necesito conseguir.

---

# PRIMERA ACCIÓN AHORA

Empieza SOLO con:

1. auditoría del repositorio;
2. git status/branch/commit;
3. estructura de archivos;
4. `.gitignore`;
5. requirements;
6. identificación de assets faltantes;
7. detección de Python disponible.

NO instales ni ejecutes el pipeline todavía.

Devuélveme el diagnóstico inicial y el plan exacto de ejecución para mi máquina.
