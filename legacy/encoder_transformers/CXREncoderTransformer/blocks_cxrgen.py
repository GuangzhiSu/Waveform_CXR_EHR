"""Vendored from cxrgen `transformernn.py`: position encoding + encoder block with split attn / padding masks for causal models."""
import math
from typing import Optional

import torch
import torch.nn as nn


def build_combined_attn_mask(
    key_padding_mask: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Merge causal and padding into one float additive mask for MultiheadAttention.

    key_padding_mask: (B, T) with True = ignore key position.
    Returns (B, T, T) float mask (0 = attend, -inf = block).
    """
    bsz, t = key_padding_mask.shape
    device = key_padding_mask.device
    mask = torch.zeros(bsz, t, t, device=device, dtype=torch.float32)
    if causal and t > 0:
        causal_block = torch.triu(
            torch.ones(t, t, device=device, dtype=torch.bool),
            diagonal=1,
        )
        mask = mask.masked_fill(causal_block.unsqueeze(0), float("-inf"))
    pad_keys = key_padding_mask.unsqueeze(1).expand(-1, t, -1)
    mask = mask.masked_fill(pad_keys, float("-inf"))
    return mask


class PositionalEmbedding(nn.Module):
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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        if attn_mask is not None and attn_mask.dim() == 3:
            nh = self.attention.num_heads
            b, t, _ = attn_mask.shape
            head_mask = attn_mask.unsqueeze(1).expand(b, nh, t, t).reshape(b * nh, t, t)
            attn_out, _ = self.attention(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=head_mask,
                need_weights=False,
            )
        else:
            attn_out, _ = self.attention(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.mlp(x_norm))
        return x
