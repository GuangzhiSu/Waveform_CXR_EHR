"""EHR-only: row-wise MLP -> causal transformer -> next-embedding + anchor + per-step disc heads."""
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
    m = torch.triu(torch.ones(t, t, device=device, dtype=torch.float32), diagonal=1)
    return m.masked_fill(m.bool(), float("-inf"))


def _forward_change_style_disc_head(embed_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Same architecture as ``ForwardChangeRowModel.head_s2f`` / ``head_p2f`` for weight loading."""
    h = embed_dim
    return nn.Sequential(
        nn.Linear(h, h),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(h, num_classes),
    )


class StepDiscMLP(nn.Module):
    """Wider MLP head on per-step embeddings; used by ``MultimodalNextStepModel`` (concat_dim input)."""

    def __init__(self, input_dim: int, num_classes: int = 3, hidden: Tuple[int, ...] = (512, 256), dropout: float = 0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Dropout(dropout),
                ]
            )
            d = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


class EHRNextStepModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        d_model: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_length: int = 512,
        anchor_pool: str = "last",
        disc_head_dropout: float = 0.2,
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
        self.head_next = nn.Linear(d_model, embed_dim)
        hdim = d_model
        self.anchor_s2f = nn.Sequential(
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim // 2, num_classes),
        )
        self.anchor_p2f = nn.Sequential(
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim // 2, num_classes),
        )
        self.disc_s2f = _forward_change_style_disc_head(embed_dim, num_classes, disc_head_dropout)
        self.disc_p2f = _forward_change_style_disc_head(embed_dim, num_classes, disc_head_dropout)

    def _pool_anchor(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: B,T,D, mask: B,T True=valid"""
        bsz, t, d = h.shape
        if self.anchor_pool == "last":
            lengths = mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(bsz, device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(
        self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor, return_embeddings: bool = False
    ) -> Tuple[
        torch.Tensor,  # logits_s2f anchor B,C
        torch.Tensor,  # logits_p2f anchor B,C
        torch.Tensor,  # pred next B,T-1,D (padded: same T as input, only first T-1 used)
        torch.Tensor,  # logits_s2f step B,T,C
        torch.Tensor,  # logits_p2f step B,T,C
    ]:
        bsz, t, _ = ehr_seq.shape
        device = ehr_seq.device
        z0 = self.row_encoder(ehr_seq)
        h = self.proj(z0)
        h = self.pos(h)
        h = self.pos_drop(h)

        pad = ~ehr_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)

        pred_next = self.head_next(h[:, :-1, :]) if t > 1 else None
        if t == 1:
            next_padded = h.new_zeros(bsz, 1, self.embed_dim)
        else:
            next_padded = pred_next.new_zeros(bsz, t, self.embed_dim)
            next_padded[:, : t - 1, :] = pred_next

        anchor_vec = self._pool_anchor(h, ehr_mask)
        log_s2f = self.anchor_s2f(anchor_vec)
        log_p2f = self.anchor_p2f(anchor_vec)
        flat = z0.reshape(-1, self.embed_dim)
        s2f_step = self.disc_s2f(flat).reshape(bsz, t, self.num_classes)
        p2f_step = self.disc_p2f(flat).reshape(bsz, t, self.num_classes)
        if return_embeddings:
            return log_s2f, log_p2f, next_padded, s2f_step, p2f_step, z0, h
        return log_s2f, log_p2f, next_padded, s2f_step, p2f_step
