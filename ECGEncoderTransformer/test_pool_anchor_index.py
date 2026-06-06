#!/usr/bin/env python3
"""Regression: last-pool uses anchor slot; mean-pool aggregates ECG tokens only."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_EXP = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXP.parent
for _p in (
    _EXP,
    _PROJECT_ROOT / "CXREncoderTransformer",
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiment1(old)",
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import ECGEncoderTransformer  # noqa: E402


def _legacy_last_idx(mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1)
    return (lengths - 1).clamp(min=0)


def main() -> int:
    device = torch.device("cpu")
    d_model = 64

    model_last = ECGEncoderTransformer(
        ecg_ckpt_path=None,
        ecg_dim=512,
        d_model=d_model,
        num_transformer_layers=1,
        num_heads=2,
        include_anchor_slot=True,
        anchor_pool="last",
    ).to(device)

    mask = torch.tensor(
        [
            [True, False, False, True],
            [True, True, True, True],
        ],
        device=device,
    )
    h = torch.randn(2, 4, d_model, device=device)

    legacy_idx = _legacy_last_idx(mask)
    assert int(legacy_idx[0]) == 1, f"legacy bug index should be 1 (pad), got {legacy_idx[0]}"
    assert int(legacy_idx[1]) == 3, f"legacy ok row last idx 3, got {legacy_idx[1]}"

    pooled_last = model_last._pool_anchor(h, mask)
    expected_last = h[:, -1, :]
    assert torch.allclose(pooled_last, expected_last), "last pool must match h[:, -1] when include_anchor_slot"

    legacy_pooled = h[torch.arange(2), legacy_idx]
    assert not torch.allclose(legacy_pooled[0], expected_last[0]), "legacy pools wrong row for padded sample"

    model_mean = ECGEncoderTransformer(
        ecg_ckpt_path=None,
        ecg_dim=512,
        d_model=d_model,
        num_transformer_layers=1,
        num_heads=2,
        include_anchor_slot=True,
        anchor_pool="mean",
    ).to(device)
    h_ecg, m_ecg = model_mean._h_and_mask_for_pool(h, mask)
    denom = m_ecg.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    expected_mean = (h_ecg * m_ecg.unsqueeze(-1).float()).sum(dim=1) / denom
    pooled_mean = model_mean._pool_anchor(h, mask)
    assert torch.allclose(pooled_mean, expected_mean), "mean pool must average ECG tokens only"
    assert not torch.allclose(pooled_mean[0], expected_last[0]), "mean pool != anchor slot for row0"

    model2 = ECGEncoderTransformer(
        ecg_ckpt_path=None,
        ecg_dim=512,
        d_model=d_model,
        num_transformer_layers=1,
        num_heads=2,
        include_anchor_slot=False,
        anchor_pool="last",
    ).to(device)
    mask2 = torch.tensor([[True, False, True]], device=device)
    h2 = torch.randn(1, 3, d_model, device=device)
    pooled2 = model2._pool_anchor(h2, mask2)
    assert torch.allclose(pooled2, h2[:, 2, :]), "last True at index 2 when no anchor slot"

    print("test_pool_anchor_index: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
