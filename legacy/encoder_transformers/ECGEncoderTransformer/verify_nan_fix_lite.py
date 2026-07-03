#!/usr/bin/env python3
"""Verify NaN fixes: encode-only-valid, split attn masks, GPU forward."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_EXP = Path(__file__).resolve().parent
PROJECT_ROOT = _EXP.parents[0]
for p in (
    _EXP,
    PROJECT_ROOT / "CXREncoderTransformer",
    PROJECT_ROOT,
    PROJECT_ROOT / "EHRWindowTransformer",
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "experiment1(old)",
):
    if p.is_dir():
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("ecg_enc_model", _EXP / "model.py")
_ecg_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ecg_model)
ECGEncoderTransformer = _ecg_model.ECGEncoderTransformer


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ecg_seq = torch.randn(2, 2, 12, 1000, device=device)
    ecg_mask = torch.tensor([[True, False], [True, True]], device=device)

    model = ECGEncoderTransformer(ecg_ckpt_path=None, ecg_dim=512).to(device)
    model.train()

    ecg_mask_lead = torch.tensor([[False, True]], device=device)
    with torch.no_grad():
        log_lead, _ = model(ecg_seq[:1], ecg_mask_lead)
    print(f"[1] leading-invalid [F,T] mask -> logits finite={torch.isfinite(log_lead).all().item()}")

    orig_encode = model._encode_flat

    def encode_with_nan(flat):
        out = orig_encode(flat)
        if out.numel() > 0:
            out = out.clone()
            out[0] = float("nan")
        return out

    model._encode_flat = encode_with_nan
    log_s, log_p = model(ecg_seq, ecg_mask)
    model._encode_flat = orig_encode
    print(f"[2] NaN enc on invalid slot -> logits finite={torch.isfinite(log_s).all().item()}")

    def train_steps(steps=3):
        m = ECGEncoderTransformer(ecg_ckpt_path=None, ecg_dim=512).to(device)
        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-4)
        y = torch.tensor([1, 0], device=device)
        losses = []
        for _ in range(steps):
            ls, _ = m(ecg_seq, ecg_mask)
            loss = F.cross_entropy(ls, y)
            losses.append(float(loss))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 5.0)
            opt.step()
        return losses

    losses = train_steps()
    print(f"[3] 3 train steps losses={[round(x, 4) for x in losses]}")

    ok = (
        torch.isfinite(log_lead).all()
        and torch.isfinite(log_s).all()
        and all(np.isfinite(x) for x in losses)
    )
    print(f"VERIFIED={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
