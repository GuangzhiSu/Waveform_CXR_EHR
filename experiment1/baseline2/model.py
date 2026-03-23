"""
Baseline2: ECG-only → predict oxygenation.
Single modality (ECG) to predict oxygenation at matched timepoint.
"""
import os
import sys

# Import encoders from baseline
_BASE = os.path.join(os.path.dirname(__file__), "..")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from baseline.model import SignalEncoder, RegressionHead

import torch
import torch.nn as nn


class ECGOnlyBaseline(nn.Module):
    """Predict oxygenation from ECG only."""

    def __init__(
        self,
        hidden_dim=512,
        ecg_ckpt_path=None,
        freeze_encoder=True,
    ):
        super().__init__()
        self.signal_encoder = SignalEncoder(
            ckpt_path=ecg_ckpt_path,
            hidden_dim=hidden_dim,
            freeze=freeze_encoder,
        )
        self.head = RegressionHead(hidden_dim, hidden_dim=hidden_dim)

    def forward(self, signal):
        feat = self.signal_encoder(signal)  # (B, H)
        return self.head(feat)   # (B,)
