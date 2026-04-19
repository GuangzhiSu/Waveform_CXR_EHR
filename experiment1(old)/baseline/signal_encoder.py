"""
xresnet1d ECG encoder only — MedTVT-R1 ``llama`` package.

ECG-only and ECG+CXR multimodal code should import from here
(``from baseline.signal_encoder import SignalEncoder``) so ``baseline.model`` (Fusion / EHR / lab) is not required.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_BASELINE_DIR = Path(__file__).resolve().parent
_EXP_ROOT = _BASELINE_DIR.parent
if str(_EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP_ROOT))
from medtvt_paths import ensure_medtvt_on_syspath

# Env VIT_PATH / ECG_CKPT / MEDTVT_ROOT are resolved inside ensure (see medtvt_paths).
ensure_medtvt_on_syspath()

from llama.xresnet1d_101 import xresnet1d101


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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(768, hidden_dim)
        self.hidden_dim = hidden_dim
        self._sig_len = sig_len

    def forward(self, x):
        with torch.no_grad() if not self.training else torch.enable_grad():
            feats = self.encoder(x)
        pooled = self.pool(feats).squeeze(-1)
        return self.proj(pooled)
