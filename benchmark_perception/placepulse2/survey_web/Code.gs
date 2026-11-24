/**
 * EXP-050 — Backend de la encuesta abierta "Calles de Lima" (Google Apps Script sobre una hoja de cálculo).
 *
 * La página pública (GitHub Pages) hace:
 *   GET  <url /exec>                         → {ok, counts: {s001: n, ...}}  votos por estímulo (selección activa)
 *   POST <url /exec>  {kind:"session", ...}  → una fila en "sessions" (perfil opcional y anónimo)
 *   POST <url /exec>  {kind:"votes", ...}    → una fila por voto en "votes"
 *
 * Los reintentos del navegador pueden duplicar un voto: el análisis deduplica por (sid, n).
 * Despliegue: Implementar → Nueva implementación → Aplicación web, "Ejecutar como: yo", "Acceso: cualquier persona".
 */

const VOTE_COLS = ["server_ts", "sid", "n", "a", "b", "w", "rt", "rep", "client_ts", "mobile"];
const SESSION_COLS = ["server_ts", "sid", "age", "gender", "lives_lima", "walk", "knows_districts", "mobile"];
const PROFILE_KEYS = ["age", "gender", "lives_lima", "walk", "knows_districts"];
const STIM_RE = /^s\d{3}$/;
const SID_RE = /^p[a-z0-9]{6,20}$/;

function sheet_(name, cols) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(cols);
    sh.setFrozenRows(1);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

/** Crea las dos hojas con su encabezado (opcional: doPost también las crea). */
function setup() {
  sheet_("votes", VOTE_COLS);
  sheet_("sessions", SESSION_COLS);
}

function doGet() {
  const cache = CacheService.getScriptCache();
  let counts = cache.get("counts");
  if (!counts) {
    const sh = sheet_("votes", VOTE_COLS);
    const n = sh.getLastRow() - 1;
    const c = {};
    if (n > 0) sh.getRange(2, 4, n, 2).getValues().forEach(r => r.forEach(s => { c[s] = (c[s] || 0) + 1; }));
    counts = JSON.stringify(c);
    cache.put("counts", counts, 60);
  }
  return json_({ ok: true, counts: JSON.parse(counts) });
}

function doPost(e) {
  let d;
  try { d = JSON.parse(e.postData.contents); } catch (err) { return json_({ ok: false, error: "bad_json" }); }
  const sid = String(d.sid || "");
  if (!SID_RE.test(sid)) return json_({ ok: false, error: "bad_sid" });
  const now = new Date();
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    if (d.kind === "session") {
      const p = d.profile || {};
      sheet_("sessions", SESSION_COLS).appendRow(
        [now, sid].concat(PROFILE_KEYS.map(k => String(p[k] || "").slice(0, 40)), [!!d.mobile]));
      return json_({ ok: true });
    }
    if (d.kind === "votes") {
      const rows = (d.votes || []).slice(0, 100)
        .filter(v => STIM_RE.test(v.a) && STIM_RE.test(v.b) && v.a !== v.b && ["a", "b", "tie"].indexOf(v.w) >= 0)
        .map(v => [now, sid, Number(v.n) | 0, v.a, v.b, v.w, Math.max(0, Math.min(600000, Number(v.rt) | 0)),
                   !!v.rep, Number(v.t) || "", !!d.mobile]);
      if (rows.length) {
        const sh = sheet_("votes", VOTE_COLS);
        sh.getRange(sh.getLastRow() + 1, 1, rows.length, VOTE_COLS.length).setValues(rows);
      }
      return json_({ ok: true, saved: rows.length });
    }
    return json_({ ok: false, error: "bad_kind" });
  } finally {
    lock.releaseLock();
  }
}
