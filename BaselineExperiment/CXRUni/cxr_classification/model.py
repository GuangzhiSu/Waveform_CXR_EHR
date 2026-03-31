"""
CXR-only baseline for ARDS severity classification.
Uses CXREncoder + MLP classification head. Input: CXR image. Output: class (0=Severe, 1=Moderate, 2=Mild).
"""
import os
import sys

# Add paths for encoder import
# cxr_classification -> CXRUni -> BaselineExperiment -> repo root
_EXP_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "experiment1(old)")
if os.path.exists(_EXP_ROOT) and _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
from baseline.model import CXREncoder

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP head for 3-class ARDS severity classification."""

    def __init__(self, input_dim, num_classes=3, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for j, m in enumerate(linears):
            if j == len(linears) - 1:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class CXRClassificationBaseline(nn.Module):
    """CXR-only model for ARDS severity classification (Mild/Moderate/Severe)."""

    def __init__(
        self,
        num_classes=3,
        hidden_dim=512,
        vit_path="google/vit-base-patch16-224-in21k",
        freeze_encoder=True,
    ):
        super().__init__()
        self.cxr_encoder = CXREncoder(vit_path=vit_path, hidden_dim=hidden_dim, freeze=freeze_encoder)
        self.head = ClassificationHead(hidden_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, cxr):
        feat = self.cxr_encoder(cxr)  # (B, H)
        return self.head(feat)  # (B, num_classes)
