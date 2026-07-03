"""Row EHR encoder + dual heads for forward s2f / p2f severity-change (3-class each)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from models.encoders.ehr import EHRMLPEncoder


class ForwardChangeRowModel(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = EHRMLPEncoder(input_dim=input_dim, embed_dim=embed_dim)
        h = embed_dim
        self.head_s2f = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )
        self.head_p2f = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head_s2f(z), self.head_p2f(z)
