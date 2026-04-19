"""Unified xresnet1d ECG encoder."""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

# Reuse existing MedTVT path resolver from experiment1(old).
_BE = Path(__file__).resolve().parents[2]
_REPO = _BE.parent
_EXP_OLD = _REPO / "experiment1(old)"
if _EXP_OLD.is_dir() and str(_EXP_OLD) not in sys.path:
    sys.path.insert(0, str(_EXP_OLD))

from medtvt_paths import ensure_medtvt_on_syspath  # noqa: E402

ensure_medtvt_on_syspath()

from llama.xresnet1d_101 import xresnet1d101  # noqa: E402


class SignalEncoder(nn.Module):
    """xresnet1d101-based ECG encoder (12-lead waveform -> hidden embedding)."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        input_channels: int = 12,
        sig_len: int = 5000,
        hidden_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = xresnet1d101(
            num_classes=5,
            input_channels=input_channels,
            kernel_size=5,
            ps_head=0.5,
            lin_ftrs_head=[768],
            use_ecgNet_Diagnosis="other",
        )
        if ckpt_path and os.path.exists(ckpt_path):
            ecg_ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ecg_ckpt.get("ecg_model", ecg_ckpt)
            self.encoder.load_state_dict(sd, strict=False)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(768, hidden_dim)
        self.hidden_dim = hidden_dim
        self._sig_len = sig_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not self.training else torch.enable_grad():
            feats = self.encoder(x)
        pooled = self.pool(feats).squeeze(-1)
        return self.proj(pooled)


class _PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding, same style as cxrgen transformer."""

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


class ECGTransformerEncoder(nn.Module):
    """
    Transformer ECG encoder inspired by cxrgen.

    Input:
      - 4D temporal ECG: [B, T, 12, L] -> [B, T, D]
      - 3D single-step ECG: [B, 12, L] -> [B, D]
    """

    def __init__(
        self,
        input_channels: int = 12,
        signal_len: int = 1000,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.signal_len = signal_len
        token_in = input_channels * signal_len
        self.input_proj = nn.Linear(token_in, hidden_dim)
        self.pos_embed = _PositionalEmbedding(hidden_dim, max_len=max_seq_length)
        self.pos_drop = nn.Dropout(dropout)

        ff_dim = int(hidden_dim * ff_mult)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            # [B, C, L] -> [B, 1, C, L]
            x = x.unsqueeze(1)
            squeeze_back = True
        elif x.dim() == 4:
            squeeze_back = False
        else:
            raise ValueError(f"Expected ECG tensor dim 3/4, got shape {tuple(x.shape)}")

        b, t, c, l = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected ECG channels={self.input_channels}, got {c}")

        # If length differs, trim/pad to configured signal_len (keeps wiring robust).
        if l > self.signal_len:
            x = x[..., : self.signal_len]
        elif l < self.signal_len:
            pad = self.signal_len - l
            x = nn.functional.pad(x, (0, pad))

        x = x.reshape(b, t, -1)          # [B, T, C*L]
        x = self.input_proj(x)           # [B, T, D]
        x = self.pos_embed(x)
        x = self.pos_drop(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if squeeze_back:
            return x[:, 0, :]            # [B, D]
        return x                          # [B, T, D]


def build_ecg_encoder(kind: str, hidden_dim: int = 512, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in {"cnn", "xresnet", "signal"}:
        return SignalEncoder(hidden_dim=hidden_dim, **kwargs)
    if kind == "transformer":
        # Map expected kwargs from existing call sites.
        allowed = {
            "input_channels", "signal_len", "num_layers", "num_heads",
            "ff_mult", "dropout", "max_seq_length",
        }
        tfm_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return ECGTransformerEncoder(hidden_dim=hidden_dim, **tfm_kwargs)
    raise ValueError(f"Unsupported ECG encoder kind: {kind}")
