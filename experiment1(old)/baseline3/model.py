"""
Baseline3: CXR-only → predict oxygenation.
Single modality (CXR) to predict oxygenation at matched timepoint.
"""
import os
import sys

_BASE = os.path.join(os.path.dirname(__file__), "..")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from baseline.model import CXREncoder, RegressionHead

import torch
import torch.nn as nn


class CXROnlyBaseline(nn.Module):
    """Predict oxygenation from CXR only."""

    def __init__(
        self,
        hidden_dim=512,
        vit_path="google/vit-base-patch16-224-in21k",
        freeze_encoder=True,
    ):
        super().__init__()
        self.cxr_encoder = CXREncoder(vit_path=vit_path, hidden_dim=hidden_dim, freeze=freeze_encoder)
        self.head = RegressionHead(hidden_dim, hidden_dim=hidden_dim)

    def forward(self, cxr):
        feat = self.cxr_encoder(cxr)  # (B, H)
        return self.head(feat)   # (B,)
