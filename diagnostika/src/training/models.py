"""timm model factory. Config keys map to timm architecture names (see config.yaml training.models)."""
from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(timm_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    """Only the classification head stays trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """During head warmup keep BatchNorm running statistics fixed (as in the research notebooks)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse square-root frequency computed from TRAIN labels, normalized to mean 1."""
    import numpy as np

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"A class has zero training examples: counts={counts.tolist()}")
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
