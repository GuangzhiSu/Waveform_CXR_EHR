"""
EHR encoder + CXR ViT encoder: CLIP alignment + 3-layer MLP on concat pooled embeddings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE = Path(__file__).resolve().parents[1]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
from models.encoders import CXREncoder, build_ehr_encoder


def _pool_stats(z: torch.Tensor, mask: torch.Tensor, stats: tuple[str, ...]) -> torch.Tensor:
    """z: (B, T, D), mask: (B, T) -> (B, D * len(stats))"""
    b = z.size(0)
    outs = []
    for i in range(b):
        valid = z[i][mask[i]]
        if valid.size(0) == 0:
            valid = z[i, :1]
        row = []
        for s in stats:
            if s == "mean":
                row.append(valid.mean(dim=0))
            elif s == "median":
                row.append(valid.median(dim=0).values)
            elif s == "max":
                row.append(valid.max(dim=0).values)
            elif s == "min":
                row.append(valid.min(dim=0).values)
            elif s == "std":
                row.append(valid.std(dim=0, unbiased=False))
            else:
                raise ValueError(f"Unknown pooling stat: {s}")
        outs.append(torch.cat(row, dim=0))
    return torch.stack(outs, dim=0)


def clip_infonce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    scale = logit_scale.exp()
    logits_ab = scale * (z_a @ z_b.T)
    targets = torch.arange(z_a.size(0), device=z_a.device, dtype=torch.long)
    return 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ab.T, targets))


class FusionMLP3(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        h2 = max(hidden // 2, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(h2, num_classes),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MultimodalEHRCXRModel(nn.Module):
    """
    Pooled EHR + pooled CXR embeddings; CLIP on projected features;
    classifier on concat(pool_ehr, pool_cxr).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        ehr_embed_dim: int = 256,
        cxr_hidden_dim: int = 512,
        contrast_dim: int = 256,
        fusion_hidden: int = 512,
        pooling_stats: tuple[str, ...] = ("mean",),
        ehr_encoder_kind: str = "mlp",
        vit_path: str = "google/vit-base-patch16-224-in21k",
        freeze_cxr_encoder: bool = True,
        logit_scale_init: float = 2.6592,
    ):
        super().__init__()
        self.pooling_stats = tuple(pooling_stats)
        self.ehr_embed_dim = ehr_embed_dim
        self.cxr_hidden_dim = cxr_hidden_dim
        self.ehr_encoder_kind = ehr_encoder_kind.lower()
        self.ehr_encoder = build_ehr_encoder(self.ehr_encoder_kind, input_dim=input_dim, embed_dim=ehr_embed_dim)
        self.cxr_encoder = CXREncoder(
            vit_path=vit_path, hidden_dim=cxr_hidden_dim, freeze=freeze_cxr_encoder
        )

        ehr_pool_dim = ehr_embed_dim * len(self.pooling_stats)
        cxr_pool_dim = cxr_hidden_dim * len(self.pooling_stats)

        self.proj_ehr = nn.Linear(ehr_pool_dim, contrast_dim)
        self.proj_cxr = nn.Linear(cxr_pool_dim, contrast_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        fused_in = ehr_pool_dim + cxr_pool_dim
        self.classifier = FusionMLP3(fused_in, fusion_hidden, num_classes)

    def encode_pooled(
        self,
        ehr_seq: torch.Tensor,
        ehr_mask: torch.Tensor,
        cxr_seq: torch.Tensor,
        cxr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f = ehr_seq.shape
        if self.ehr_encoder_kind == "transformer":
            z_e = self.ehr_encoder(ehr_seq, attention_mask=ehr_mask)
        elif self.ehr_encoder_kind == "contrastive":
            z_e = self.ehr_encoder(ehr_seq.reshape(-1, f), normalize=False).view(b, t, self.ehr_embed_dim)
        else:
            z_e = self.ehr_encoder(ehr_seq.reshape(-1, f)).view(b, t, self.ehr_embed_dim)
        pool_e = _pool_stats(z_e, ehr_mask, self.pooling_stats)

        b2, tc, c, h, w = cxr_seq.shape
        z_x = self.cxr_encoder(cxr_seq.reshape(-1, c, h, w)).view(b2, tc, self.cxr_hidden_dim)
        pool_x = _pool_stats(z_x, cxr_mask, self.pooling_stats)
        return pool_e, pool_x

    def forward(
        self,
        ehr_seq: torch.Tensor,
        ehr_mask: torch.Tensor,
        cxr_seq: torch.Tensor,
        cxr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pool_e, pool_x = self.encode_pooled(ehr_seq, ehr_mask, cxr_seq, cxr_mask)
        z_e = F.normalize(self.proj_ehr(pool_e), dim=1, p=2)
        z_x = F.normalize(self.proj_cxr(pool_x), dim=1, p=2)
        logits = self.classifier(torch.cat([pool_e, pool_x], dim=1))
        return logits, z_e, z_x, self.logit_scale
