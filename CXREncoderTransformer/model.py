"""Frozen ViT CXR encoder -> causal transformer -> dual MLP heads for s2f/p2f change."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from models.encoders.cxr import CXREncoder

from blocks_cxrgen import EncoderBlock, PositionalEmbedding


def _build_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    m = torch.triu(torch.ones(t, t, device=device, dtype=torch.float32), diagonal=1)
    return m.masked_fill(m.bool(), float("-inf"))


def _change_cls_head(d_model: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, num_classes),
    )


class CXREncoderTransformer(nn.Module):
    """
    CXR sequence in [anchor_t - 24h, anchor_t - 12h]:
    frozen ViT per image -> causal transformer -> anchor pool -> s2f/p2f MLP heads.
    """

    def __init__(
        self,
        cxr_dim: int = 512,
        d_model: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        num_classes: int = 3,
        max_seq_length: int = 512,
        anchor_pool: str = "last",
        vit_path: str = "google/vit-base-patch16-224-in21k",
        freeze_cxr: bool = True,
    ):
        super().__init__()
        self.cxr_dim = cxr_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=freeze_cxr)
        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.proj = nn.Linear(cxr_dim, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=max_seq_length)
        self.pos_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.head_s2f = _change_cls_head(d_model, num_classes, head_dropout)
        self.head_p2f = _change_cls_head(d_model, num_classes, head_dropout)

    def _pool_anchor(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = h.shape
        if self.anchor_pool == "last":
            lengths = mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(bsz, device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(self, cxr_seq: torch.Tensor, cxr_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cxr_seq: B, T, 3, 224, 224
        cxr_mask: B, T — True = valid CXR (not pad / missing)
        """
        bsz, t, _, _, _ = cxr_seq.shape
        device = cxr_seq.device
        flat = cxr_seq.reshape(bsz * t, 3, 224, 224)
        z = self.cxr_enc(flat).reshape(bsz, t, self.cxr_dim)
        m = cxr_mask.float().unsqueeze(-1)
        z = z * m + self.miss_cxr * (1.0 - m)

        h = self.proj(z)
        h = self.pos(h)
        h = self.pos_drop(h)

        pad = ~cxr_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)

        anchor_vec = self._pool_anchor(h, cxr_mask)
        return self.head_s2f(anchor_vec), self.head_p2f(anchor_vec)
