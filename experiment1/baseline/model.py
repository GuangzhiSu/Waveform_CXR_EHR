"""
Lightweight baseline: pretrained CXR + Signal + EHR encoders + MLP head.
Uses encoders from MedTVT-R1 for consistency.
"""
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add MedTVT-R1 to path for encoder imports
_MEDTVT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "MedTVT-R1")
if _MEDTVT_ROOT not in sys.path:
    sys.path.insert(0, _MEDTVT_ROOT)

from transformers import ViTConfig, ViTModel, ViTImageProcessor
from llama.xresnet1d_101 import xresnet1d101
from llama.lab_encoder import LabsEncoder


def load_encoder_weights(model, ckpt_path, key_prefix="", strict=False):
    """Load encoder weights from checkpoint, optionally stripping a key prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    state = OrderedDict()
    for k, v in ckpt.items():
        if key_prefix and k.startswith(key_prefix):
            new_k = k[len(key_prefix):].lstrip(".")
            state[new_k] = v
        elif not key_prefix:
            state[k] = v
    model.load_state_dict(state, strict=strict)
    return model


class CXREncoder(nn.Module):
    """ViT-based CXR encoder (MedTVT-R1 style)."""

    def __init__(self, vit_path="google/vit-base-patch16-224-in21k", hidden_dim=512, freeze=True):
        super().__init__()
        config = ViTConfig.from_pretrained(vit_path)
        self.vit = ViTModel(config)
        self.processor = ViTImageProcessor.from_pretrained(vit_path, do_rescale=False)
        self.proj = nn.Linear(768, hidden_dim)
        self.hidden_dim = hidden_dim
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 224, 224)
        with torch.no_grad() if not self.training else torch.enable_grad():
            out = self.vit(x).last_hidden_state  # (B, 197, 768)
            cls = out[:, 0]  # (B, 768)
        return self.proj(cls)  # (B, hidden_dim)


class SignalEncoder(nn.Module):
    """xresnet1d101-based ECG/signal encoder (MedTVT-R1 style)."""

    def __init__(
        self,
        ckpt_path=None,
        input_channels=12,
        sig_len=5000,
        hidden_dim=512,
        freeze=True,
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
        # Output: (B, 768, T) -> pool to (B, 768)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(768, hidden_dim)
        self.hidden_dim = hidden_dim
        self._sig_len = sig_len

    def forward(self, x):
        # x: (B, 12, L) - resample to ~1000 if needed for xresnet
        with torch.no_grad() if not self.training else torch.enable_grad():
            feats = self.encoder(x)  # (B, 768, T)
        pooled = self.pool(feats).squeeze(-1)  # (B, 768)
        return self.proj(pooled)  # (B, hidden_dim)


class EHREncoder(nn.Module):
    """EHR encoder: MLP over tabular features.
    Can use LabsEncoder (100-dim) if available, or a generic MLP for variable dims.
    """

    def __init__(self, input_dim, hidden_dim=512, lab_ckpt_path=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if input_dim == 100 and lab_ckpt_path and os.path.exists(lab_ckpt_path):
            self.lab_encoder = LabsEncoder()
            load_encoder_weights(self.lab_encoder, lab_ckpt_path, key_prefix="labs_encoder.")
            self.proj = nn.Linear(1024, hidden_dim)
        else:
            self.lab_encoder = None
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, hidden_dim),
            )
            self.proj = None

    def forward(self, x):
        if self.lab_encoder is not None:
            h = self.lab_encoder(x)
            return self.proj(h)
        return self.mlp(x)


def _init_regression_head(module):
    """Xavier init for head; smaller init on last layer to avoid wild initial predictions."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RegressionHead(nn.Module):
    """Larger MLP head for regression, with Xavier init. Helps avoid collapse when encoder is frozen."""

    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )
        _init_regression_head(self.net)
        # Smaller init on last layer so initial preds are close to target mean
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            if self.net[-1].bias is not None:
                self.net[-1].bias.fill_(250.0)  # p2f_vent_fio2 (P/F ratio) typical mean

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FusionBaseline(nn.Module):
    """Predict oxygenation from ECG + CXR. EHR provides ground truth only (not model input).

    Goal: Use ECG and CXR data to predict oxygenation at the timestamp; EHR oxygenation
    (e.g. spo2, PaO2, P/F ratio) is used as ground truth for evaluation.
    """

    def __init__(
        self,
        hidden_dim=512,
        vit_path="google/vit-base-patch16-224-in21k",
        ecg_ckpt_path=None,
        freeze_encoders=True,
    ):
        super().__init__()
        self.cxr_encoder = CXREncoder(vit_path=vit_path, hidden_dim=hidden_dim, freeze=freeze_encoders)
        self.signal_encoder = SignalEncoder(
            ckpt_path=ecg_ckpt_path,
            hidden_dim=hidden_dim,
            freeze=freeze_encoders,
        )
        total_dim = hidden_dim * 2  # CXR + ECG only (no EHR input)
        self.head = RegressionHead(total_dim, hidden_dim=hidden_dim)

    def forward(self, cxr, signal):
        cxr_feat = self.cxr_encoder(cxr)      # (B, H)
        sig_feat = self.signal_encoder(signal)  # (B, H)
        fused = torch.cat([cxr_feat, sig_feat], dim=-1)  # (B, 2*H)
        return self.head(fused)   # (B,) continuous oxygenation
