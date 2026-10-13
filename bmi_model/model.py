"""
Bayesian-ViT model for BMI estimation from a face image.

Architecture
------------
ViT-B/16 backbone (timm `vit_base_patch16_224.augreg_in21k_ft_in1k`,
ImageNet-21k pre-training + ImageNet-1k fine-tuning) followed by a 2-layer
variational head (bayesian-torch `LinearReparameterization`):

    face 224x224 -> ViT-B/16 -> 768 -> Bayesian Linear -> 256 -> ReLU
                 -> Dropout -> Bayesian Linear -> 1 (BMI)

Every weight of the head is a Gaussian N(mu, sigma^2). Each forward pass
samples a different set of weights, so running the same image T times gives
T different BMI predictions. Their mean is the prediction and their variance
is the uncertainty.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from bayesian_torch.layers import LinearReparameterization

BACKBONE = "vit_base_patch16_224.augreg_in21k_ft_in1k"


class BayesianHead(nn.Module):
    def __init__(self, n_features: int, num_outputs: int = 1, dropout: float = 0.3):
        super().__init__()
        self.num_outputs = num_outputs
        self.fc1 = LinearReparameterization(n_features, 256)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = LinearReparameterization(256, num_outputs)

    def forward(self, x):
        out1, kl1 = self.fc1(x)
        out2, kl2 = self.fc2(self.drop(self.act(out1)))
        kl = kl1 + kl2
        if self.num_outputs == 1:
            return out2.squeeze(1), kl
        return out2, kl


class BayesianBMIRegressor(nn.Module):
    """Deterministic ViT backbone + variational head. forward -> (bmi [B], kl)."""

    def __init__(self, backbone: str = BACKBONE, pretrained: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = BayesianHead(self.backbone.num_features, num_outputs=1, dropout=dropout)

    def forward(self, x):
        return self.head(self.backbone(x))


def freeze_backbone(model: BayesianBMIRegressor) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: BayesianBMIRegressor) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True


def load_member(path, device="cpu") -> BayesianBMIRegressor:
    """Load one ensemble member saved by export_weights (weights-only file)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BayesianBMIRegressor(ckpt.get("backbone", BACKBONE), pretrained=False,
                                 dropout=float(ckpt.get("dropout", 0.3)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
