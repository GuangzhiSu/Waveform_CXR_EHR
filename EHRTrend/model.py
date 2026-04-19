"""EHR trend model: EHR Transformer encoder -> pooled sequence embedding -> 3-class trend head."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from models.encoders.ehr import build_ehr_encoder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 256, dropout: float = 0.3):
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
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class EHRTrendBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        embed_dim: int = 256,
        pooling_stats=("mean", "median", "max", "min", "std"),
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        # Use unified EHR Transformer encoder directly.
        self.encoder = build_ehr_encoder("transformer", input_dim=input_dim, embed_dim=embed_dim)
        self.pooling_stats = tuple(pooling_stats)
        self.embed_dim = embed_dim
        self.head = ClassificationHead(
            input_dim=embed_dim * len(self.pooling_stats),
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
        )

    def _pool_one(self, x_valid: torch.Tensor):
        outs = []
        for s in self.pooling_stats:
            if s == "mean":
                outs.append(x_valid.mean(dim=0))
            elif s == "median":
                outs.append(x_valid.median(dim=0).values)
            elif s == "max":
                outs.append(x_valid.max(dim=0).values)
            elif s == "min":
                outs.append(x_valid.min(dim=0).values)
            elif s == "std":
                outs.append(x_valid.std(dim=0, unbiased=False))
            else:
                raise ValueError(f"Unsupported pooling stat: {s}")
        return torch.cat(outs, dim=0)

    def forward(self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor):
        # Transformer encoder returns (B, T, D) for sequence input.
        z = self.encoder(ehr_seq, attention_mask=ehr_mask)
        pooled = []
        for i in range(z.size(0)):
            valid = z[i][ehr_mask[i]]
            if valid.size(0) == 0:
                valid = z[i, :1]
            pooled.append(self._pool_one(valid))
        pooled = torch.stack(pooled, dim=0)
        return self.head(pooled)
