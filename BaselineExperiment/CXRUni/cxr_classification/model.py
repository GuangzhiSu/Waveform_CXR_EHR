"""
CXR-only baseline for ARDS severity classification.
Uses CXREncoder + MLP classification head. Input: CXR image. Output: class (0=Severe, 1=Moderate, 2=Mild).
"""
import os
import sys
from pathlib import Path

# cxr_classification -> CXRUni -> BaselineExperiment -> repo root (Waveform_CXR_EHR)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_ROOT = _REPO_ROOT / "experiment1(old)"
if _EXP_ROOT.is_dir():
    p = str(_EXP_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
# Import CXR encoder only — avoids ``baseline.model`` -> ``llama.xresnet1d_101`` (MedTVT-R1).
from baseline.cxr_encoder import CXREncoder

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
