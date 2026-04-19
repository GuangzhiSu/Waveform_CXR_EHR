"""
EHR encoder + ECG SignalEncoder, CLIP-style alignment between modalities,
and a 3-layer MLP classifier on concatenated pooled embeddings.
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
from models.encoders import build_ecg_encoder, build_ehr_encoder


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
    """Symmetric CLIP / InfoNCE for two modalities (2D normalized embeddings)."""
    scale = logit_scale.exp()
    logits_ab = scale * (z_a @ z_b.T)
    logits_ba = logits_ab.T
    targets = torch.arange(z_a.size(0), device=z_a.device, dtype=torch.long)
    return 0.5 * (
        F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets)
    )


class FusionMLP3(nn.Module):
    """Three Linear layers: in -> hidden -> hidden2 -> num_classes."""

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


class MultimodalEHRECGModel(nn.Module):
    """
    Pooled EHR + pooled ECG -> projections for contrastive loss;
    concat(pooled_ehr, pooled_ecg) -> 3-layer MLP -> class logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        ehr_embed_dim: int = 256,
        ecg_hidden_dim: int = 512,
        contrast_dim: int = 256,
        fusion_hidden: int = 512,
        pooling_stats: tuple[str, ...] = ("mean",),
        ehr_encoder_kind: str = "mlp",
        ecg_encoder_kind: str = "cnn",
        ecg_ckpt_path: str | None = None,
        freeze_ecg_encoder: bool = True,
        logit_scale_init: float = 2.6592,
    ):
        super().__init__()
        self.pooling_stats = tuple(pooling_stats)
        self.ehr_embed_dim = ehr_embed_dim
        self.ecg_hidden_dim = ecg_hidden_dim
        self.ecg_encoder_kind = ecg_encoder_kind.lower()
        self.ehr_encoder_kind = ehr_encoder_kind.lower()
        self.ehr_encoder = build_ehr_encoder(self.ehr_encoder_kind, input_dim=input_dim, embed_dim=ehr_embed_dim)
        if self.ecg_encoder_kind == "transformer":
            self.ecg_encoder = build_ecg_encoder("transformer", hidden_dim=ecg_hidden_dim)
        else:
            self.ecg_encoder = build_ecg_encoder(
                "cnn",
                hidden_dim=ecg_hidden_dim,
                ckpt_path=ecg_ckpt_path,
                freeze=freeze_ecg_encoder,
            )

        ehr_pool_dim = ehr_embed_dim * len(self.pooling_stats)
        ecg_pool_dim = ecg_hidden_dim * len(self.pooling_stats)

        self.proj_ehr = nn.Linear(ehr_pool_dim, contrast_dim)
        self.proj_ecg = nn.Linear(ecg_pool_dim, contrast_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        fused_in = ehr_pool_dim + ecg_pool_dim
        self.classifier = FusionMLP3(fused_in, fusion_hidden, num_classes)

    def encode_pooled(
        self,
        ehr_seq: torch.Tensor,
        ehr_mask: torch.Tensor,
        signal_seq: torch.Tensor,
        signal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f = ehr_seq.shape
        if self.ehr_encoder_kind == "transformer":
            z_e = self.ehr_encoder(ehr_seq, attention_mask=ehr_mask)
        elif self.ehr_encoder_kind == "contrastive":
            z_e = self.ehr_encoder(ehr_seq.reshape(-1, f), normalize=False).view(b, t, self.ehr_embed_dim)
        else:
            z_e = self.ehr_encoder(ehr_seq.reshape(-1, f)).view(b, t, self.ehr_embed_dim)
        pool_e = _pool_stats(z_e, ehr_mask, self.pooling_stats)

        b2, te, c, le = signal_seq.shape
        if self.ecg_encoder_kind == "transformer":
            z_c = self.ecg_encoder(signal_seq, attention_mask=signal_mask)
        else:
            z_c = self.ecg_encoder(signal_seq.reshape(-1, c, le)).view(b2, te, self.ecg_hidden_dim)
        pool_c = _pool_stats(z_c, signal_mask, self.pooling_stats)
        return pool_e, pool_c

    def forward(
        self,
        ehr_seq: torch.Tensor,
        ehr_mask: torch.Tensor,
        signal_seq: torch.Tensor,
        signal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits (B, num_classes)
            z_e_n, z_c_n: L2-normalized contrastive embeddings (B, contrast_dim)
            logit_scale: raw parameter (scalar tensor)
        """
        pool_e, pool_c = self.encode_pooled(ehr_seq, ehr_mask, signal_seq, signal_mask)

        z_e = F.normalize(self.proj_ehr(pool_e), dim=1, p=2)
        z_c = F.normalize(self.proj_ecg(pool_c), dim=1, p=2)

        logits = self.classifier(torch.cat([pool_e, pool_c], dim=1))
        return logits, z_e, z_c, self.logit_scale
