"""
ECG-only baseline for ARDS severity classification.
Uses SignalEncoder + MLP classification head. Input: ECG waveform. Output: class (0=Severe, 1=Moderate, 2=Mild).
"""
import os
import sys

_EXP_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "experiment1(old)")
if os.path.exists(_EXP_ROOT) and _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
from baseline.model import SignalEncoder

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP head for 3-class ARDS severity classification."""

    def __init__(self, input_dim, num_classes=3, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for j, m in enumerate(linears):
            if j == len(linears) - 1:
                # Start near uniform softmax; avoid large Xavier logits that bias argmax early
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ECGClassificationBaseline(nn.Module):
    """ECG-only model for ARDS severity classification (Mild/Moderate/Severe)."""

    def __init__(
        self,
        num_classes=3,
        hidden_dim=512,
        ecg_ckpt_path=None,
        freeze_encoder=True,
        use_lora=False,
        lora_r=8,
        lora_alpha=16.0,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.signal_encoder = SignalEncoder(
            ckpt_path=ecg_ckpt_path,
            hidden_dim=hidden_dim,
            freeze=freeze_encoder,
        )
        self.head = ClassificationHead(hidden_dim, num_classes=num_classes, hidden_dim=hidden_dim)
        if use_lora:
            from ECGUni.lora_inject import (
                inject_lora_into_xresnet,
                inject_lora_signal_encoder_proj,
                set_trainable_lora_and_head,
            )

            self._lora_encoder_layers = inject_lora_into_xresnet(
                self.signal_encoder.encoder, r=lora_r, alpha=lora_alpha
            )
            self.signal_encoder.proj = inject_lora_signal_encoder_proj(
                self.signal_encoder.proj, lora_r, lora_alpha
            )
            if freeze_encoder:
                set_trainable_lora_and_head(self)
        else:
            self._lora_encoder_layers = 0

    def forward(self, signal):
        feat = self.signal_encoder(signal)  # (B, H)
        return self.head(feat)  # (B, num_classes)
