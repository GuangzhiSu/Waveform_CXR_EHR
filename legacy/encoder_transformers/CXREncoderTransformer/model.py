"""Frozen ViT CXR encoder -> causal transformer -> dual MLP heads for s2f/p2f change."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from models.encoders.cxr import CXREncoder

from blocks_cxrgen import EncoderBlock, PositionalEmbedding


def _build_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """Bool mask for nn.MultiheadAttention (True = disallow attention)."""
    return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)


def _change_cls_head(d_model: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, num_classes),
    )


class CXREncoderTransformer(nn.Module):
    """
    CXR sequence in [anchor_t - 24h, anchor_t - 12h], optional learnable anchor slot at t:
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
        include_anchor_slot: bool = True,
    ):
        super().__init__()
        self.cxr_dim = cxr_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        self.include_anchor_slot = include_anchor_slot
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=freeze_cxr)
        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.anchor_slot = nn.Parameter(torch.zeros(cxr_dim)) if include_anchor_slot else None
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
        """Pool anchor representation from transformer output h (B, T, D)."""
        if self.anchor_pool == "last":
            if self.include_anchor_slot:
                # Anchor slot is always appended as the last timestep.
                return h[:, -1, :]
            # Last True index (handles trailing pad without assuming mask.sum()-1).
            rev = mask.long().fliplr()
            has_any = rev.any(dim=1)
            last_idx = mask.size(1) - 1 - rev.argmax(dim=1)
            last_idx = torch.where(has_any, last_idx, torch.zeros_like(last_idx))
            batch_idx = torch.arange(h.size(0), device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(
        self,
        cxr_seq: torch.Tensor,
        cxr_mask: torch.Tensor,
        *,
        return_anchor_vec: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        cxr_seq: B, T, 3, 224, 224
        cxr_mask: B, T — True = valid CXR (not pad / missing)
        """
        bsz, t, _, _, _ = cxr_seq.shape
        device = cxr_seq.device
        flat = cxr_seq.reshape(bsz * t, 3, 224, 224)
        flat_mask = cxr_mask.reshape(-1)
        z = self.miss_cxr.view(1, 1, -1).expand(bsz, t, -1).clone()
        if flat_mask.any():
            enc_out = self.cxr_enc(flat[flat_mask])
            enc_out = torch.nan_to_num(enc_out, nan=0.0, posinf=0.0, neginf=0.0)
            z.reshape(-1, self.cxr_dim)[flat_mask] = enc_out

        if self.include_anchor_slot:
            slot = self.anchor_slot.view(1, 1, -1).expand(bsz, 1, self.cxr_dim)
            z = torch.cat([z, slot], dim=1)
            cxr_mask = torch.cat(
                [cxr_mask, torch.ones(bsz, 1, dtype=cxr_mask.dtype, device=device)],
                dim=1,
            )
            t = z.size(1)

        h = self.proj(z)
        h = self.pos(h)
        h = self.pos_drop(h)

        safe_mask = cxr_mask.clone()
        all_invalid = ~safe_mask.any(dim=1)
        if all_invalid.any():
            safe_mask[all_invalid, 0] = True
        pad = ~safe_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        anchor_vec = self._pool_anchor(h, cxr_mask)
        log_s = self.head_s2f(anchor_vec)
        log_p = self.head_p2f(anchor_vec)
        if return_anchor_vec:
            return log_s, log_p, anchor_vec
        return log_s, log_p
