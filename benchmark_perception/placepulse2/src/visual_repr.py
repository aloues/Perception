"""
Representaciones visuales de un spot / segmento — las "tres para probar" de la asesoría del 23/10/2026.

    V1_12    media de las 12 vistas (idea del profesor; = agregador histórico "visual mean")
    V2_4R    media de 4 vistas ALEATORIAS distintas, con semilla (idea de la reunión; n=4 es propuesta)
    V2_4S    ablation: 4 vistas a 90° con desfase aleatorio (con FOV 90° cubren 360° sin solape)
    V3_BOVW  Visual Bag of Words: K-Means sobre descriptores de ENTRENAMIENTO → histograma L1 de
             ocurrencias de "palabras visuales" por spot / segmento / barrio
    V4_FUSION [embedding medio, histograma BoVW] (opcional; cada bloque con norma L2 = 1)

Los embeddings continuos se conservan siempre (el asesor no pidió descartarlos). Un centroide de
K-Means NO tiene significado semántico por sí mismo ("fachada", "pista"): solo se nombra tras
inspeccionar ejemplos reales.

Divergencias entre histogramas (el "KL" del boceto, SI significa Kullback–Leibler): `kl_divergence`
es ASIMÉTRICA y requiere suavizado para no dividir por cero; `js_divergence` es simétrica y acotada
por ln 2. Son métricas entre distribuciones, NO un modelo de grafo.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# ─── selección de vistas ──────────────────────────────────────────────────────

def select_random(headings: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n headings distintos al azar (sin reemplazo). Si hay menos de n, devuelve todos."""
    headings = np.asarray(headings)
    if len(headings) <= n:
        return np.sort(headings)
    return np.sort(rng.choice(headings, size=n, replace=False))


def select_stratified(headings: np.ndarray, n: int, rng: np.random.Generator, step: int = 30) -> np.ndarray:
    """n headings equiespaciados (360/n) con desfase aleatorio múltiplo de `step`; solo los disponibles."""
    headings = np.asarray(headings)
    gap = 360 // n
    offset = int(rng.integers(0, max(1, gap // step))) * step
    want = (offset + gap * np.arange(n)) % 360
    return np.sort(np.intersect1d(want, headings))


def mean_pool(E: np.ndarray) -> np.ndarray:
    return E.mean(axis=0)


def l2n(X: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


# ─── redundancia ──────────────────────────────────────────────────────────────

def angular_separation(h1, h2) -> np.ndarray:
    d = np.abs(np.asarray(h1) - np.asarray(h2)) % 360
    return np.minimum(d, 360 - d)


def effective_rank(E: np.ndarray) -> float:
    """Participación de autovalores de la Gram centrada: (Σλ)²/Σλ². 1 = todas iguales; n = independientes."""
    X = E - E.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(X @ X.T), 0, None)
    s = lam.sum()
    return float(s * s / (lam ** 2).sum()) if s > 0 else 1.0


def geometric_overlap(fov: float, sep: float) -> float:
    """Fracción del campo de visión horizontal compartida por dos vistas separadas `sep` grados."""
    return max(0.0, (fov - sep) / fov)


# ─── Visual Bag of Words ──────────────────────────────────────────────────────

class BoVW:
    """Vocabulario visual por K-Means. `fit` recibe SOLO descriptores de entrenamiento."""

    def __init__(self, k: int, seed: int = 0, n_init: int = 3, max_iter: int = 300):
        self.k, self.seed = k, seed
        self.km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        self.fitted_on: int | None = None

    def fit(self, X_train: np.ndarray) -> "BoVW":
        self.km.fit(X_train.astype(np.float64))
        self.fitted_on = len(X_train)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        return self.km.predict(X.astype(np.float64))

    def train_occupancy(self) -> np.ndarray:
        return np.bincount(self.km.labels_, minlength=self.k)


def histograms(words: np.ndarray, groups: np.ndarray, k: int, n_groups: int | None = None,
               norm: str = "l1") -> np.ndarray:
    """Histograma de palabras por grupo (fila = grupo). norm: 'l1' (distribución), 'none' (conteos)."""
    n_groups = int(groups.max()) + 1 if n_groups is None else n_groups
    H = np.zeros((n_groups, k), dtype=np.float64)
    np.add.at(H, (groups, words), 1.0)
    if norm == "l1":
        s = H.sum(axis=1, keepdims=True)
        H = H / np.where(s > 0, s, 1.0)
    elif norm != "none":
        raise ValueError(norm)
    return H


def tfidf(H_counts: np.ndarray, df_counts: np.ndarray | None = None, n_docs: int | None = None) -> np.ndarray:
    """TF-IDF con idf = log((1+N)/(1+df)) + 1 (df de ENTRENAMIENTO si se pasa)."""
    tf = H_counts / np.clip(H_counts.sum(axis=1, keepdims=True), 1, None)
    if df_counts is None:
        df_counts, n_docs = (H_counts > 0).sum(axis=0), len(H_counts)
    idf = np.log((1 + n_docs) / (1 + df_counts)) + 1
    return tf * idf


# ─── divergencias ─────────────────────────────────────────────────────────────

def smooth(P: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Suavizado aditivo (Laplace con pseudo-masa eps por bin) y renormalización: evita log(0)."""
    P = np.asarray(P, dtype=np.float64) + eps
    return P / P.sum(axis=-1, keepdims=True)


def kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """D_KL(P || Q) en nats, fila a fila, con suavizado. ASIMÉTRICA."""
    P, Q = smooth(P, eps), smooth(Q, eps)
    return np.sum(P * (np.log(P) - np.log(Q)), axis=-1)


def js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Jensen–Shannon en nats, ∈ [0, ln 2]. Sin suavizado es estable (0·log0 := 0)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if eps:
        P, Q = smooth(P, eps), smooth(Q, eps)
    M = 0.5 * (P + Q)

    def _kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * (np.log(a) - np.log(np.where(b > 0, b, 1.0))), 0.0)
        return t.sum(axis=-1)

    return np.clip(0.5 * _kl(P, M) + 0.5 * _kl(Q, M), 0.0, np.log(2))


def fuse(E: np.ndarray, H: np.ndarray) -> np.ndarray:
    """V4: [embedding, histograma], cada bloque L2-normalizado para que ninguno domine por escala."""
    return np.hstack([l2n(E), l2n(H)])
