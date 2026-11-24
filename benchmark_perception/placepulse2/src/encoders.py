"""
Encoders para el benchmark de perceived safety.

Contrato del encoder OURS replicado EXACTAMENTE desde
`graph/extract_embeddings.py` (clase SimCLR_Model, lineas 87-97):
    - backbone: timm vit_base_patch16_224, pretrained=False, num_classes=0
    - projector: Linear(768,512) -> ReLU -> Linear(512,128)   [SOLO para cargar
      el state_dict; NUNCA se usa en inferencia downstream]
    - forward devuelve h (768-dim del backbone), no la salida del projector
    - checkpoint bajo la clave ckpt['model_state_dict']
    - preprocessing: Resize((224,224)) + ToTensor + Normalize(ImageNet)
    - L2-normalizacion aplicada DESPUES del forward
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimCLR_Model(nn.Module):
    """Replica exacta de graph/extract_embeddings.py:87-97."""

    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.projector = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 128)
        )

    def forward(self, x):
        return self.encoder(x)


@dataclass
class EncoderSpec:
    name: str
    embed_dim: int
    image_size: int
    model: nn.Module
    transform: transforms.Compose


def _transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_ours(ckpt_path: str, device: str = "cpu") -> EncoderSpec:
    base = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model = SimCLR_Model(base)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"El checkpoint {ckpt_path} no contiene 'model_state_dict'. "
            f"Claves presentes: {list(ckpt.keys())}"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return EncoderSpec("ours_simclr_vit_b16", 768, 224, model, _transform(224))


def load_imagenet_vit(device: str = "cpu") -> EncoderSpec:
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.to(device).eval()
    return EncoderSpec("imagenet_vit_b16", 768, 224, model, _transform(224))


def load_dinov2(device: str = "cpu", image_size: int = 224) -> EncoderSpec:
    """
    DINOv2 ViT-B/14 a 224x224, no a su 518 nativo.

    Motivo: a 518 el modelo ve (518/14)^2 = 1369 tokens frente a los
    (224/16)^2 = 196 de los otros dos encoders. Eso no seria el "mismo
    protocolo" que exige la comparacion, y ademas dispara el coste de computo.
    A 224 son (224/14)^2 = 256 tokens, comparable. timm interpola los
    embeddings posicionales al construir el modelo con otro img_size.
    """
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=image_size,
    )
    model.to(device).eval()
    return EncoderSpec("dinov2_vit_b14", 768, image_size, model, _transform(image_size))


@torch.no_grad()
def embed(spec: EncoderSpec, batch: torch.Tensor) -> torch.Tensor:
    """Forward + L2-norm, igual que extract_embeddings.py:292-293."""
    return F.normalize(spec.model(batch), dim=-1)
