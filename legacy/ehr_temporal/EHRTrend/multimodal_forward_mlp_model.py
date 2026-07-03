"""Three encoders -> concat -> StepDisc heads (same interface slice as ``MultimodalNextStepModel``)."""
from __future__ import annotations

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

from models.encoders.cxr import CXREncoder
from models.encoders.ehr import EHRMLPEncoder
from models.encoders.ecg import build_ecg_encoder

from model_nextstep import StepDiscMLP


class MultimodalForwardMLPModel(nn.Module):
    """
    Per-anchor: EHR row + CXR image + ECG waveform -> encoders -> ``proj_*`` -> concat ``3*fuse_dim``
    -> ``disc_s2f`` / ``disc_p2f`` (``StepDiscMLP``). Attribute names match a subset of
    ``MultimodalNextStepModel`` for checkpoint transfer.
    """

    def __init__(
        self,
        input_dim: int,
        ehr_embed_dim: int = 256,
        cxr_dim: int = 512,
        ecg_dim: int = 512,
        fuse_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
        vit_path: str = "google/vit-base-patch16-224-in21k",
        freeze_cxr: bool = True,
        ecg_ckpt_path: Optional[str] = None,
        freeze_ecg: bool = True,
        ecg_sig_len: int = 5000,
    ):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.num_classes = num_classes
        concat_dim = 3 * fuse_dim

        self.ehr_enc = EHRMLPEncoder(input_dim=input_dim, embed_dim=ehr_embed_dim)
        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=freeze_cxr)
        self.ecg_enc = build_ecg_encoder(
            "cnn",
            hidden_dim=ecg_dim,
            ckpt_path=ecg_ckpt_path,
            freeze=freeze_ecg,
            sig_len=ecg_sig_len,
        )
        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.miss_ecg = nn.Parameter(torch.zeros(ecg_dim))
        self.proj_e = nn.Linear(ehr_embed_dim, fuse_dim)
        self.proj_x = nn.Linear(cxr_dim, fuse_dim)
        self.proj_s = nn.Linear(ecg_dim, fuse_dim)

        self.disc_s2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)
        self.disc_p2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)

    def forward(
        self,
        ehr: torch.Tensor,
        cxr: torch.Tensor,
        ecg: torch.Tensor,
        cxr_valid: torch.Tensor,
        ecg_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ehr: [B, F], cxr: [B,3,224,224], ecg: [B,12,L], *_valid: [B] bool
        """
        ze = self.ehr_enc(ehr)
        zx_raw = self.cxr_enc(cxr)
        m_x = cxr_valid.float().unsqueeze(-1)
        zx = zx_raw * m_x + self.miss_cxr * (1.0 - m_x)

        zs_raw = self.ecg_enc(ecg)
        m_s = ecg_valid.float().unsqueeze(-1)
        zs = zs_raw * m_s + self.miss_ecg * (1.0 - m_s)

        fe = self.proj_e(ze)
        fx = self.proj_x(zx)
        fs = self.proj_s(zs)
        cat = torch.cat([fe, fx, fs], dim=-1)
        return self.disc_s2f(cat), self.disc_p2f(cat)
