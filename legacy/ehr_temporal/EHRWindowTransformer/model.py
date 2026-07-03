"""Window transformers: EHR percentile rows, CXR images, or ECG waveforms -> TransformerEncoder -> masked mean -> s2f/p2f heads."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_exp_old = _PROJECT_ROOT / "experiment1(old)"
if _exp_old.is_dir() and str(_exp_old) not in sys.path:
    sys.path.insert(0, str(_exp_old))

from models.encoders.cxr import CXREncoder  # noqa: E402
from models.encoders.ecg import build_ecg_encoder  # noqa: E402


def _encode_with_safe_padding(
    encoder: nn.TransformerEncoder,
    h: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run transformer; ensure >=1 unmasked token per sample (all-pad rows yield NaN in PyTorch)."""
    safe_mask = mask.clone()
    all_invalid = ~safe_mask.any(dim=1)
    if all_invalid.any():
        safe_mask[all_invalid, 0] = True
    pad = ~safe_mask.bool()
    return encoder(h, src_key_padding_mask=pad)


def _pool_sequence(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over valid tokens; if none are valid, mean over miss/pad tokens (not zeros)."""
    w = mask.float().unsqueeze(-1)
    denom = w.sum(dim=1)
    pooled = (h * w).sum(dim=1)
    all_invalid = denom.squeeze(-1) == 0
    if all_invalid.any():
        pooled[all_invalid] = h[all_invalid].mean(dim=1)
    return pooled / denom.clamp(min=1.0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DirectWindowTransformer(nn.Module):
    """
    Sequence = all EHR rows in [anchor_t - 24h, anchor_t - 12h] (same window as EHRTrend EHRNextStepDataset).
    Predicts anchor-time ``severity_change_12to24h`` logits for s2f and p2f (3 classes).
    No per-row MLP encoder and no extra judging MLP: only input Linear, transformer blocks, and two Linear heads.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEmbedding(d_model, max_len=max_seq_len)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)
        self.head_s2f = nn.Linear(d_model, num_classes)
        self.head_p2f = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: B, T, input_dim — percentile features per time step
        mask: B, T — True = valid token (not pad)
        """
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.drop(h)
        h = _encode_with_safe_padding(self.encoder, h, mask)
        h = self.enc_norm(h)
        pooled = _pool_sequence(h, mask)
        return self.head_s2f(pooled), self.head_p2f(pooled)


class _WindowTransformerCore(nn.Module):
    """Sinusoidal PE + TransformerEncoder + masked mean pool + dual heads."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos = PositionalEmbedding(d_model, max_len=max_seq_len)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)
        self.head_s2f = nn.Linear(d_model, num_classes)
        self.head_p2f = nn.Linear(d_model, num_classes)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pos(h)
        h = self.drop(h)
        h = _encode_with_safe_padding(self.encoder, h, mask)
        h = self.enc_norm(h)
        pooled = _pool_sequence(h, mask)
        return self.head_s2f(pooled), self.head_p2f(pooled)


class CXRWindowTransformer(nn.Module):
    """
    Sequence = all CXR images in [anchor_t - 24h, anchor_t - 12h].
    ViT encoder per image -> linear to d_model -> transformer -> anchor s2f/p2f logits.
    """

    def __init__(
        self,
        cxr_dim: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 8192,
        vit_path: str = "google/vit-base-patch16-224-in21k",
        freeze_cxr: bool = True,
    ):
        super().__init__()
        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=freeze_cxr)
        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.input_proj = nn.Linear(cxr_dim, d_model)
        self.core = _WindowTransformerCore(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
            max_seq_len=max_seq_len,
        )

    def forward(self, cxr_seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cxr_seq: B, T, 3, 224, 224
        mask: B, T — True = valid CXR (not pad / missing)
        """
        bsz, t, _, _, _ = cxr_seq.shape
        flat = cxr_seq.reshape(bsz * t, 3, 224, 224)
        z = self.cxr_enc(flat).reshape(bsz, t, -1)
        m = mask.float().unsqueeze(-1)
        z = z * m + self.miss_cxr * (1.0 - m)
        h = self.input_proj(z)
        return self.core(h, mask)


class ECGWindowTransformer(nn.Module):
    """
    Sequence = all ECG waveforms in [anchor_t - 24h, anchor_t - 12h].
    xresnet1d encoder per waveform -> linear to d_model -> transformer -> anchor s2f/p2f logits.
    """

    def __init__(
        self,
        ecg_dim: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 8192,
        ecg_ckpt_path: Optional[str] = None,
        freeze_ecg: bool = True,
        ecg_sig_len: int = 5000,
    ):
        super().__init__()
        self.ecg_enc = build_ecg_encoder(
            "cnn",
            hidden_dim=ecg_dim,
            ckpt_path=ecg_ckpt_path,
            freeze=freeze_ecg,
            sig_len=ecg_sig_len,
        )
        self.miss_ecg = nn.Parameter(torch.zeros(ecg_dim))
        self.input_proj = nn.Linear(ecg_dim, d_model)
        self.core = _WindowTransformerCore(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
            max_seq_len=max_seq_len,
        )

    def forward(self, ecg_seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ecg_seq: B, T, 12, L
        mask: B, T — True = valid ECG (not pad / missing)
        """
        bsz, t, c, L = ecg_seq.shape
        flat = ecg_seq.reshape(bsz * t, c, L)
        z = self.ecg_enc(flat).reshape(bsz, t, -1)
        m = mask.float().unsqueeze(-1)
        z = z * m + self.miss_ecg * (1.0 - m)
        h = self.input_proj(z)
        return self.core(h, mask)
