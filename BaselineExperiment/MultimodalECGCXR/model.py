"""ECG + CXR multimodal baseline: unified encoder folder + concat + MLP head."""

import torch
import torch.nn as nn

from models.encoders import CXREncoder, build_ecg_encoder


class ClassificationHead(nn.Module):
    """Same MLP head as CXR/ECG unimodal baselines; ``input_dim`` is fused size (e.g. 1024)."""

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
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MultimodalECGCXRBaseline(nn.Module):
    """Concatenate CXR and ECG 512-d embeddings -> 3-class logits."""

    def __init__(
        self,
        num_classes=3,
        hidden_dim=512,
        vit_path="google/vit-base-patch16-224-in21k",
        ecg_ckpt_path=None,
        ecg_encoder_kind: str = "cnn",
        freeze_encoder=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ecg_encoder_kind = ecg_encoder_kind.lower()
        self.cxr_encoder = CXREncoder(vit_path=vit_path, hidden_dim=hidden_dim, freeze=freeze_encoder)
        if self.ecg_encoder_kind == "transformer":
            self.signal_encoder = build_ecg_encoder("transformer", hidden_dim=hidden_dim)
        else:
            self.signal_encoder = build_ecg_encoder(
                "cnn",
                hidden_dim=hidden_dim,
                ckpt_path=ecg_ckpt_path,
                freeze=freeze_encoder,
            )
        fused_dim = hidden_dim * 2
        self.head = ClassificationHead(fused_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, cxr, signal):
        z_cxr = self.cxr_encoder(cxr)
        if self.ecg_encoder_kind == "transformer":
            z_ecg = self.signal_encoder(signal)  # [B, D]
        else:
            z_ecg = self.signal_encoder(signal)  # [B, D]
        z = torch.cat([z_cxr, z_ecg], dim=-1)
        return self.head(z)
