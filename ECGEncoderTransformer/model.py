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

from models.encoders.ecg import build_ecg_encoder_from_ckpt  # noqa: E402
from blocks_cxrgen import EncoderBlock, PositionalEmbedding  # noqa: E402

_LOGIT_CLAMP = 30.0


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


def _init_cls_head(head: nn.Sequential) -> None:
    last = head[-1]
    if isinstance(last, nn.Linear):
        nn.init.xavier_uniform_(last.weight, gain=0.5)
        nn.init.zeros_(last.bias)


class ECGEncoderTransformer(nn.Module):
    """
    ECG sequence in [anchor_t - 24h, anchor_t - 12h]:
    frozen ECG encoder (MedTVT xresnet1d or Symile ResNet18 PL ckpt) per waveform -> causal transformer -> heads.
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
        anchor_pool: str = "mean",
        ecg_ckpt_path: Optional[str] = None,
        input_channels: int = 12,
        sig_len: int = 1000,
        freeze_ecg: bool = True,
        include_anchor_slot: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        self.freeze_ecg = freeze_ecg
        self.include_anchor_slot = include_anchor_slot
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.ecg_enc, self.ecg_encoder_kind = build_ecg_encoder_from_ckpt(
            ecg_ckpt_path,
            hidden_dim=ecg_dim,
            sig_len=sig_len,
            freeze=freeze_ecg,
            input_channels=input_channels,
        )
        self.ecg_dim = int(getattr(self.ecg_enc, "hidden_dim", ecg_dim))

        self.miss_ecg = nn.Parameter(torch.zeros(self.ecg_dim))
        self.anchor_slot = nn.Parameter(torch.zeros(self.ecg_dim)) if include_anchor_slot else None
        self.proj = nn.Linear(self.ecg_dim, d_model)
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
        self._init_learnable_params()

    def _init_learnable_params(self) -> None:
        nn.init.normal_(self.miss_ecg, mean=0.0, std=0.02)
        if self.anchor_slot is not None:
            nn.init.normal_(self.anchor_slot, mean=0.0, std=0.02)
        _init_cls_head(self.head_s2f)
        _init_cls_head(self.head_p2f)

    def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.freeze_ecg:
            with torch.no_grad():
                return self.ecg_enc(flat)
        return self.ecg_enc(flat)

    def _h_and_mask_for_pool(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ECG timesteps only (exclude learnable anchor slot from classification pool)."""
        if self.include_anchor_slot and h.size(1) > 1:
            return h[:, :-1, :], mask[:, :-1]
        return h, mask

    def _pool_anchor(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool anchor representation from transformer output h (B, T, D)."""
        if self.anchor_pool == "last":
            if self.include_anchor_slot:
                return h[:, -1, :]
            rev = mask.long().fliplr()
            has_any = rev.any(dim=1)
            last_idx = mask.size(1) - 1 - rev.argmax(dim=1)
            last_idx = torch.where(has_any, last_idx, torch.zeros_like(last_idx))
            batch_idx = torch.arange(h.size(0), device=h.device, dtype=torch.long)
            return h[batch_idx, last_idx]

        h_ecg, m_ecg = self._h_and_mask_for_pool(h, mask)
        denom = m_ecg.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h_ecg * m_ecg.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(
        self,
        ecg_seq: torch.Tensor,
        ecg_mask: torch.Tensor,
        *,
        return_anchor_vec: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ecg_seq: B, T, C, L (12-lead waveform)
        ecg_mask: B, T — True = valid ECG (not pad / missing)
        """
        bsz, t, c, length = ecg_seq.shape
        device = ecg_seq.device
        flat = ecg_seq.reshape(bsz * t, c, length)
        flat_mask = ecg_mask.reshape(-1)
        z = self.miss_ecg.view(1, 1, -1).expand(bsz, t, -1).clone()
        if flat_mask.any():
            enc_out = self._encode_flat(flat[flat_mask])
            enc_out = torch.nan_to_num(enc_out, nan=0.0, posinf=0.0, neginf=0.0)
            z.reshape(-1, self.ecg_dim)[flat_mask] = enc_out

        if self.include_anchor_slot:
            slot = self.anchor_slot.view(1, 1, -1).expand(bsz, 1, self.ecg_dim)
            z = torch.cat([z, slot], dim=1)
            ecg_mask = torch.cat(
                [ecg_mask, torch.ones(bsz, 1, dtype=ecg_mask.dtype, device=device)],
                dim=1,
            )
            t = z.size(1)

        h = self.proj(z)
        h = self.pos(h)
        h = self.pos_drop(h)

        safe_mask = ecg_mask.clone()
        all_invalid = ~safe_mask.any(dim=1)
        if all_invalid.any():
            safe_mask[all_invalid, 0] = True
        pad = ~safe_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        anchor_vec = self._pool_anchor(h, ecg_mask)
        log_s = self.head_s2f(anchor_vec).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        log_p = self.head_p2f(anchor_vec).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        if return_anchor_vec:
            return log_s, log_p, anchor_vec
        return log_s, log_p
