"""EHR-only: row MLP -> causal transformer -> dual cls heads + anchor-embed prediction head."""
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

from blocks_cxrgen import EncoderBlock, PositionalEmbedding


def _build_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """Bool mask for nn.MultiheadAttention (True = disallow attention)."""
    return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)


def _change_cls_head(embed_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    h = embed_dim
    return nn.Sequential(
        nn.Linear(h, h),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(h, num_classes),
    )


class EHREncoderTransformerEmbedPred(nn.Module):
    """
    Per-row ``EHRMLPEncoder`` -> causal transformer on [t-24h, t-12h] lookback -> anchor pooling.
    Row input = Symile-style percentile features concatenated with presence indicators (dim 2F).

    Heads:
    - s2f / p2f change classification (same as EHREncoderTransformer)
    - anchor embed prediction: pooled transformer state -> embed_dim (target = row_encoder(t))
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        d_model: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        num_classes: int = 3,
        max_seq_length: int = 512,
        anchor_pool: str = "last",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.row_encoder = EHRMLPEncoder(input_dim=input_dim, embed_dim=embed_dim)
        self.proj = nn.Linear(embed_dim, d_model)
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
        self.head_anchor_embed = nn.Linear(d_model, embed_dim)

    def _pool_anchor(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: B,T,D; mask: B,T True=valid."""
        bsz, _, _ = h.shape
        if self.anchor_pool == "last":
            lengths = mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(bsz, device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def encode_rows(self, ehr_rows: torch.Tensor) -> torch.Tensor:
        """Encode single-row or batched EHR features -> row embeddings."""
        return self.row_encoder(ehr_rows)

    def forward_transformer(
        self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = ehr_seq.device
        z0 = self.row_encoder(ehr_seq)
        z0 = z0 * ehr_mask.unsqueeze(-1).to(dtype=z0.dtype)
        h = self.proj(z0)
        h = self.pos(h)
        h = self.pos_drop(h)

        t = h.size(1)
        pad = ~ehr_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)
        anchor_vec = self._pool_anchor(h, ehr_mask)
        return anchor_vec, z0

    def forward(
        self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor, return_pred_embed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_vec, _ = self.forward_transformer(ehr_seq, ehr_mask)
        log_s2f = self.head_s2f(anchor_vec)
        log_p2f = self.head_p2f(anchor_vec)
        if return_pred_embed:
            pred_embed = self.head_anchor_embed(anchor_vec)
            return log_s2f, log_p2f, pred_embed
        return log_s2f, log_p2f
