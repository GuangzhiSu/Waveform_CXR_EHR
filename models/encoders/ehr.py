"""Unified EHR encoder implementations (MLP / Transformer / Contrastive)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EHRMLPEncoder(nn.Module):
    """Per-row MLP encoder used by existing EHR baselines."""

    def __init__(self, input_dim: int, embed_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, embed_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.layer_norm(self.fc3(x))
        return x


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class EHRTransformerEncoder(nn.Module):
    """
    Transformer-based temporal EHR encoder inspired by `cxrgen`.

    Supports:
    - 2D input: [N, F] -> [N, D] (single-row fallback)
    - 3D input: [B, T, F] + optional mask [B, T] -> [B, T, D]
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = _PositionalEmbedding(embed_dim, max_len=max_seq_length)
        self.dropout = nn.Dropout(dropout)

        ff_dim = int(embed_dim * ff_mult)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = self.input_proj(x)
            return self.norm(x)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D/3D EHR tensor, got shape {tuple(x.shape)}")

        x = self.input_proj(x)
        x = self.pos_embed(x)
        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class EHRContrastiveEncoder(nn.Module):
    """Wrapper that adds a projection head for CLIP-style contrastive alignment."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        proj_dim: int = 256,
        base: str = "mlp",
    ):
        super().__init__()
        if base == "mlp":
            self.backbone = EHRMLPEncoder(input_dim=input_dim, embed_dim=embed_dim)
        elif base == "transformer":
            self.backbone = EHRTransformerEncoder(input_dim=input_dim, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown EHR encoder base '{base}'")
        self.projector = nn.Linear(embed_dim, proj_dim)
        self.base = base

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        z = self.backbone(x, attention_mask=attention_mask) if self.base == "transformer" else self.backbone(x)
        z = self.projector(z)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z


def build_ehr_encoder(kind: str, input_dim: int, embed_dim: int = 256) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp":
        return EHRMLPEncoder(input_dim=input_dim, embed_dim=embed_dim)
    if kind == "transformer":
        return EHRTransformerEncoder(input_dim=input_dim, embed_dim=embed_dim)
    if kind == "contrastive":
        return EHRContrastiveEncoder(input_dim=input_dim, embed_dim=embed_dim, proj_dim=embed_dim, base="mlp")
    raise ValueError(f"Unsupported EHR encoder kind: {kind}")


# Backward-compatible alias for old imports.
EHREncoder = EHRMLPEncoder
