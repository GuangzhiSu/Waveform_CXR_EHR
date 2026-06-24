"""Patient-temporal contrastive model.

CXR_t1, CXR_t2 -> shared Bio-ViL-T (frozen, precomputed) -> shared MLP proj -> c_t1, c_t2 (L2).
ECG interval embeddings (frozen, precomputed) + relative-time embedding ->
3-layer Transformer -> interval embedding h_ecg.
q = L2( fusion( concat(c_t1, h_ecg) ) ).
S = q @ c_t2^T / temperature.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CXRProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ECGTemporalEncoder(nn.Module):
    """ECG embeddings + relative-time embedding -> 3-layer Transformer -> h_ecg."""

    def __init__(self, ecg_dim: int, d_model: int, num_layers: int = 3, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, pool: str = "mean",
                 rel_dim: int = 3):
        super().__init__()
        self.pool = pool
        self.in_proj = nn.Linear(ecg_dim, d_model)
        self.rel_proj = nn.Sequential(
            nn.Linear(rel_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ecg_feats: torch.Tensor, ecg_rel: torch.Tensor,
                ecg_mask: torch.Tensor) -> torch.Tensor:
        """ecg_feats (B,L,D_ecg), ecg_rel (B,L,3), ecg_mask (B,L) True=valid -> (B,d_model)."""
        B = ecg_feats.size(0)
        h = self.in_proj(ecg_feats) + self.rel_proj(ecg_rel)
        mask = ecg_mask
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)
            cls_mask = torch.ones(B, 1, dtype=ecg_mask.dtype, device=ecg_mask.device)
            mask = torch.cat([cls_mask, ecg_mask], dim=1)
        key_padding = ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding)
        h = self.norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pool == "cls":
            return h[:, 0]
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        return (h * m).sum(dim=1) / denom


class FusionPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PatientTemporalModel(nn.Module):
    def __init__(self, cxr_dim: int, ecg_dim: int, proj_dim: int = 256,
                 cxr_proj_hidden: int = 512, d_model: int = 256, ecg_tx_layers: int = 3,
                 ecg_tx_heads: int = 4, ecg_tx_mlp_ratio: float = 4.0, fusion_hidden: int = 512,
                 dropout: float = 0.1, ecg_pool: str = "mean", temperature: float = 0.07,
                 learnable_temperature: bool = False):
        super().__init__()
        self.cxr_proj = CXRProjection(cxr_dim, cxr_proj_hidden, proj_dim, dropout)
        self.ecg_temporal = ECGTemporalEncoder(
            ecg_dim, d_model, ecg_tx_layers, ecg_tx_heads, ecg_tx_mlp_ratio, dropout, ecg_pool
        )
        self.fusion = FusionPredictor(proj_dim + d_model, fusion_hidden, proj_dim, dropout)

        self.learnable_temperature = learnable_temperature
        init_log = math.log(1.0 / temperature)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_log, dtype=torch.float32))

    def temperature_value(self) -> float:
        return float(torch.exp(-self.logit_scale).item())

    def encode(self, batch: dict):
        c1 = self.cxr_proj(batch["c1"])
        c2 = self.cxr_proj(batch["c2"])
        h_ecg = self.ecg_temporal(batch["ecg_feats"], batch["ecg_rel"], batch["ecg_mask"])
        q = self.fusion(torch.cat([c1, h_ecg], dim=-1))
        return q, c2, c1

    def forward(self, batch: dict):
        q, c2, c1 = self.encode(batch)
        scale = torch.exp(self.logit_scale).clamp(max=100.0)
        logits = (q @ c2.t()) * scale  # (B, B)
        return {"q": q, "c2": c2, "c1": c1, "logits": logits, "logit_scale": scale}
