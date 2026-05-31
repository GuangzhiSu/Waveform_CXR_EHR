"""Frozen baseline2 xresnet1d ECG encoder -> causal transformer -> dual MLP heads for s2f/p2f change."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CXR_EXP = _PROJECT_ROOT / "CXREncoderTransformer"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if _CXR_EXP.is_dir() and str(_CXR_EXP) not in sys.path:
    sys.path.insert(0, str(_CXR_EXP))

from models.encoders.ecg import SignalEncoder  # noqa: E402
from blocks_cxrgen import EncoderBlock, PositionalEmbedding  # noqa: E402


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


class ECGEncoderTransformer(nn.Module):
    """
    ECG sequence in [anchor_t - 24h, anchor_t - 12h]:
    frozen xresnet1d (baseline2 SignalEncoder) per waveform -> causal transformer -> anchor pool -> s2f/p2f MLP heads.
    """

    def __init__(
        self,
        ecg_dim: int = 512,
        d_model: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        num_classes: int = 3,
        max_seq_length: int = 512,
        anchor_pool: str = "last",
        ecg_ckpt_path: Optional[str] = None,
        input_channels: int = 12,
        sig_len: int = 1000,
        freeze_ecg: bool = True,
    ):
        super().__init__()
        self.ecg_dim = ecg_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        self.freeze_ecg = freeze_ecg
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.ecg_enc = SignalEncoder(
            ckpt_path=ecg_ckpt_path,
            input_channels=input_channels,
            sig_len=sig_len,
            hidden_dim=ecg_dim,
            freeze=False,
        )
        if freeze_ecg:
            for p in self.ecg_enc.parameters():
                p.requires_grad = False

        self.miss_ecg = nn.Parameter(torch.zeros(ecg_dim))
        self.proj = nn.Linear(ecg_dim, d_model)
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

    def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.freeze_ecg:
            with torch.no_grad():
                return self.ecg_enc(flat)
        return self.ecg_enc(flat)

    def _pool_anchor(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = h.shape
        if self.anchor_pool == "last":
            lengths = mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(bsz, device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(self, ecg_seq: torch.Tensor, ecg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ecg_seq: B, T, C, L (12-lead waveform)
        ecg_mask: B, T — True = valid ECG (not pad / missing)
        """
        bsz, t, c, length = ecg_seq.shape
        device = ecg_seq.device
        flat = ecg_seq.reshape(bsz * t, c, length)
        z = self._encode_flat(flat).reshape(bsz, t, self.ecg_dim)
        m = ecg_mask.float().unsqueeze(-1)
        z = z * m + self.miss_ecg * (1.0 - m)

        h = self.proj(z)
        h = self.pos(h)
        h = self.pos_drop(h)

        pad = ~ecg_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)

        anchor_vec = self._pool_anchor(h, ecg_mask)
        return self.head_s2f(anchor_vec), self.head_p2f(anchor_vec)
