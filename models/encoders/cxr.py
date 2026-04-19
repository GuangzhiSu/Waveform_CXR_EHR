"""Unified ViT-based CXR encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTImageProcessor, ViTModel


class CXREncoder(nn.Module):
    """ViT-based CXR encoder used across unimodal/multimodal baselines."""

    def __init__(self, vit_path: str = "google/vit-base-patch16-224-in21k", hidden_dim: int = 512, freeze: bool = True):
        super().__init__()
        config = ViTConfig.from_pretrained(vit_path)
        if hasattr(config, "add_pooling_layer"):
            config.add_pooling_layer = False
        self.vit = ViTModel.from_pretrained(vit_path, config=config)
        self.processor = ViTImageProcessor.from_pretrained(vit_path, do_rescale=False)
        self.proj = nn.Linear(768, hidden_dim)
        self.hidden_dim = hidden_dim
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_trainable = any(p.requires_grad for p in self.vit.parameters())
        if vit_trainable:
            out = self.vit(x).last_hidden_state
        else:
            with torch.no_grad():
                out = self.vit(x).last_hidden_state
        cls = out[:, 0]
        return self.proj(cls)
