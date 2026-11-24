# Encuesta abierta "Calles de Lima" (EXP-050)

Enlace público: **https://aloues.github.io/calles-lima/** (GitHub Pages; no pide cuenta a quien responde).
Las respuestas se guardan en una hoja de Google del investigador mediante Apps Script.

| Archivo | Qué es |
|---|---|
| `index.html` | La encuesta: consentimiento, perfil opcional, 25 pares, 2 repeticiones de control, selección activa. |
| `Code.gs` | Backend en Apps Script: `GET` conteos por estímulo; `POST` sesiones y votos a las hojas `sessions` y `votes`. |
| `publish_site.py` | Copia la página, los 400 estímulos anónimos y la portada a `../../../../calles-lima` y publica en GitHub Pages. |

## Conectar la hoja de Google (5 minutos, una sola vez)

1. En Google Drive: **Nuevo → Hojas de cálculo de Google**. Ponle de nombre `Encuesta Calles de Lima – respuestas`.
2. En la hoja: **Extensiones → Apps Script**. Borra el código de ejemplo, pega todo `Code.gs` y guarda (Ctrl+S).
3. En el editor, elige la función `setup` en la barra superior y pulsa **Ejecutar**. Acepta los permisos.
   Aparecen las pestañas `votes` y `sessions`.
4. **Implementar → Nueva implementación → ⚙ → Aplicación web**:
   - Ejecutar como: **Yo**
   - Quién tiene acceso: **Cualquier persona**
   Pulsa **Implementar** y copia la **URL de la aplicación web** (termina en `/exec`).
5. Publica la página con esa URL:

   ```
   python publish_site.py --endpoint https://script.google.com/macros/s/XXXX/exec
   ```

   Comprueba que abrir la URL `/exec` en el navegador muestre `{"ok":true,"counts":{}}`.

Si después cambias `Code.gs`, usa **Implementar → Gestionar implementaciones → editar → Nueva versión**: así la URL no cambia.

## Estado al 06/11/2026

- **Aprobada** por el estudiante (aprobación del asesor/ética comunicada el 06/11).
- **Estímulos regenerados con la X (M30):** las 4 vistas de cada punto son las diagonales a la calle (EXP-056),
  dispuestas como desde la vereda (arriba: adelante-izq. / adelante-der.; abajo: atrás-izq. / atrás-der.). Mismos
  400 puntos e IDs; la versión 0/90/180/270 queda en `outputs/survey/stimuli_abs4/`. El sitio local
  (`../../../../calles-lima/`) ya tiene los estímulos nuevos (`publish_site.py --no-push`).
- **Falta (acciones del investigador):** (1) crear la hoja y desplegar `Code.gs` (sección de arriba, 5 min);
  (2) decidir la licencia de las imágenes (abajo); (3) `python publish_site.py --endpoint <URL /exec>` publica el repo
  público `aloues/calles-lima` y activa GitHub Pages (`gh` ya está autenticado como `aloues`).

### Decisión tomada (06/11, M34): modo EN VIVO

La página ya no republica imágenes descargadas: cada foto se pide a la Street View Static API desde el navegador, con
el mismo panorama (`pano_id`) y las mismas 4 direcciones (la X) de cada estímulo (`outputs/survey/stimuli_live.csv`,
generado por `src/survey_live_panos.py` con metadata gratuita). En el sitio no queda ninguna imagen de Street View.
Costo estimado: ~220 imágenes por persona; meta de 3,000 votos (~120 personas) ≈ 26k imágenes ≈ 40–50 USD si aplica el
cupo mensual gratuito (10k por mes), ~180 USD si no. Por eso la meta es 3,000 votos y hay que poner un tope diario.

Barranco y La Victoria no tenían `pano_id` (captura de 2016 por ubicación): la encuesta muestra su panorama ACTUAL
(2013–2025). Para analizar los votos con el modelo hay que extraer esas mismas 4 vistas (≈ 800 imágenes, ≈ 6 USD o
dentro del cupo) con el collector y su ledger.

**Pasos del investigador (una sola vez):**
1. Google Cloud → Credenciales → **Crear clave de API** nueva (no usar la del `.env`): restricción de aplicación
   **Sitios web** = `https://aloues.github.io/*`; restricción de API = **Street View Static API**.
   En **Cuotas** de la Street View Static API: límite de **2,000 solicitudes por día** (protege el crédito).
2. Hoja de Google + `Code.gs` (sección "Conectar la hoja", 5 min) → URL `/exec`.
3. `python publish_site.py --endpoint <URL /exec> --live-key <clave del paso 1>` → crea/actualiza el repo público
   `aloues/calles-lima` y GitHub Pages. Abrir el enlace en el celular y comprobar que se ven las 4 fotos de cada lugar.

## Datos

- `votes`: `server_ts, sid, n, a, b, w (a|b|tie), rt (ms), rep (repetición de control), client_ts, mobile`.
  Los reintentos pueden duplicar una fila: el análisis deduplica por `(sid, n)`.
- `sessions`: perfil opcional (edad, género, vive en Lima, frecuencia de caminar, conoce los distritos).
- El estímulo `sNNN` se traduce a distrito, spot y segmento con `outputs/survey/stimuli_manifest.csv`, que NO se publica.

Para analizar: **Archivo → Descargar → CSV** de cada pestaña a `outputs/survey/responses/`.

## Difusión

Mensaje sugerido (WhatsApp, redes):

> ¿Qué calles de Lima se ven más seguras? Estoy haciendo mi tesis sobre percepción de seguridad urbana y necesito tu
> opinión: comparas pares de calles y eliges la que te parece más segura. 5 minutos, anónima, desde el celular.
> 👉 https://aloues.github.io/calles-lima/

Código QR: `outputs/survey/qr_calles_lima.png`. Meta según la simulación de potencia
(`outputs/survey/analysis/power_simulation.csv`): ≥ 120 personas (3,000 votos) para un ranking utilizable de los 400
lugares y ~200 personas (5,000 votos) para uno fiable; las comparaciones entre distritos necesitan bastante menos.
