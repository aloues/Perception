"""
Probe de ranking sobre embeddings congelados (secciones 14 y 16).

Por imagen:   z = encoder(image)   [congelado, 768-d, L2-norm]
              s = Linear(z)        [escalar]
Por par:      logit = s_left - s_right
              loss  = BCEWithLogitsLoss(logit, y),  y = 1 si gano la izquierda

La formulacion es antisimetrica por construccion: intercambiar las dos imagenes
invierte el signo del logit. El probe no puede aprender un sesgo de posicion,
que es justamente lo que se quiere dado el 50.84% de votos a la derecha medido
en la auditoria.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearRankingHead(nn.Module):
    """Probe lineal. Sin bias: un bias se cancela en la resta s_A - s_B."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
        return (self.score(z_left) - self.score(z_right)).squeeze(-1)

    @torch.no_grad()
    def score_images(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z).squeeze(-1)


class MLPRankingHead(nn.Module):
    """Variante no lineal, para una fase posterior. No es el probe por defecto."""

    def __init__(self, embed_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
        return (self.score(z_left) - self.score(z_right)).squeeze(-1)

    @torch.no_grad()
    def score_images(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z).squeeze(-1)


def build_head(kind: str = "linear", embed_dim: int = 768, **kw) -> nn.Module:
    if kind == "linear":
        return LinearRankingHead(embed_dim)
    if kind == "mlp":
        return MLPRankingHead(embed_dim, **kw)
    raise ValueError(f"head desconocido: {kind}")
