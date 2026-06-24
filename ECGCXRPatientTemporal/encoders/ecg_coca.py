"""Frozen ECG-CoCa encoder wrapper (PKUDigitalHealth/ECG-R1 -> ECG-Chat).

Builds only the ECG tower of the CoCa model (avoids the text tower, which would
otherwise download ``ncbi/MedCPT-Query-Encoder`` from HuggingFace), and loads
the ``ecg.*`` weights from the ``cpt_wfep_epoch_20.pt`` CoCa checkpoint.

forward(x) where x is (B, 12, 5000) -> (B, 512) L2-normalized ECG latent,
matching CoCa's ``encode_ecg``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env_setup  # noqa: E402,F401

from ecg_coca.open_clip.model import _build_ecg_tower  # noqa: E402


def _clean_state_dict(raw) -> dict:
    """Pull a flat tensor state-dict out of various checkpoint container layouts."""
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "module", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    sd = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        sd[nk] = v
    return sd


def _extract_ecg_tower_sd(sd: dict) -> dict:
    """Keep only the ECG tower params and strip the ``ecg.`` prefix."""
    out = {k[len("ecg."):]: v for k, v in sd.items() if k.startswith("ecg.")}
    if not out:
        # Some checkpoints may already be the bare tower (no ecg. prefix).
        out = sd
    return out


class ECGCoCaEncoder(nn.Module):
    """Frozen ECG-CoCa ECG tower. forward(x)->(B, 512) normalized latent."""

    feat_dim = 512

    def __init__(self, ckpt_path: str, config_path: str, freeze: bool = True,
                 normalize: bool = True, strict: bool = False):
        super().__init__()
        cfg = json.load(open(config_path))
        self.embed_dim = int(cfg["embed_dim"])
        self.ecg = _build_ecg_tower(embed_dim=self.embed_dim, ecg_cfg=cfg["ecg_cfg"])
        self.seq_length = int(cfg["ecg_cfg"]["seq_length"])
        self.lead_num = int(cfg["ecg_cfg"]["lead_num"])
        self.normalize = normalize
        self.loaded_pretrained = False

        if ckpt_path and Path(ckpt_path).is_file():
            raw = torch.load(ckpt_path, map_location="cpu")
            sd = _extract_ecg_tower_sd(_clean_state_dict(raw))
            missing, unexpected = self.ecg.load_state_dict(sd, strict=strict)
            n_loaded = len(sd) - len(unexpected)
            print(
                f"  ECG-CoCa: loaded {n_loaded}/{len(sd)} tower tensors from {Path(ckpt_path).name} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if unexpected:
                print(f"    unexpected[:5]={unexpected[:5]}")
            if missing:
                print(f"    missing[:5]={missing[:5]}")
            self.loaded_pretrained = True
        else:
            print(
                f"  WARNING: ECG-CoCa checkpoint not found at {ckpt_path!r}; "
                "encoder is RANDOM-INIT (download cpt_wfep_epoch_20.pt for real features)."
            )

        if freeze:
            for p in self.ecg.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, seq_length) -> (B, embed_dim)."""
        out = self.ecg(x)
        pooled = out[0] if isinstance(out, (tuple, list)) else out
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled
