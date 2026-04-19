"""
ViT-based CXR encoder only — no MedTVT-R1 ``llama`` package.

CXR-only jobs should import from here (``from baseline.cxr_encoder import CXREncoder``)
so ``baseline.model`` (ECG/EHR + xresnet) is never loaded.
"""
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTImageProcessor


class CXREncoder(nn.Module):
    """ViT-based CXR encoder (MedTVT-R1 style)."""

    def __init__(self, vit_path="google/vit-base-patch16-224-in21k", hidden_dim=512, freeze=True):
        super().__init__()
        # Must use from_pretrained so checkpoint weights load; ViTModel(config) alone is random init.
        # We read CLS from last_hidden_state[:, 0] only — no pooler. Many ViT ckpts omit pooler weights;
        # add_pooling_layer=False avoids random init + "Some weights were not initialized" stderr noise.
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

    def forward(self, x):
        # x: (B, 3, 224, 224)
        # Frozen ViT: run under no_grad (saves memory; proj still trains — dL/dW uses cls as constant).
        # Unfrozen ViT: full forward in autograd so backbone updates.
        vit_trainable = any(p.requires_grad for p in self.vit.parameters())
        if vit_trainable:
            out = self.vit(x).last_hidden_state  # (B, 197, 768)
        else:
            with torch.no_grad():
                out = self.vit(x).last_hidden_state
        cls = out[:, 0]  # (B, 768)
        return self.proj(cls)  # (B, hidden_dim)
