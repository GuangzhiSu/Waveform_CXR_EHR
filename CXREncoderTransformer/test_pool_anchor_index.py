#!/usr/bin/env python3
"""Regression: last-pool must use final anchor slot, not padding between CXR and anchor."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_EXP = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXP.parent
for _p in (_EXP, _PROJECT_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import CXREncoderTransformer  # noqa: E402


def _legacy_last_idx(mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1)
    return (lengths - 1).clamp(min=0)


def main() -> int:
    device = torch.device("cpu")
    model = CXREncoderTransformer(
        vit_path="google/vit-base-patch16-224-in21k",
        cxr_dim=512,
        d_model=64,
        num_transformer_layers=1,
        num_heads=2,
        include_anchor_slot=True,
        anchor_pool="last",
    ).to(device)

    # After collate + anchor append: [CXR valid, pad..., anchor valid]
    mask = torch.tensor(
        [
            [True, False, False, True],
            [True, True, True, True],
        ],
        device=device,
    )
    h = torch.randn(2, 4, model.d_model, device=device)

    legacy_idx = _legacy_last_idx(mask)
    assert int(legacy_idx[0]) == 1, f"legacy bug index should be 1 (pad), got {legacy_idx[0]}"
    assert int(legacy_idx[1]) == 3, f"legacy ok row last idx 3, got {legacy_idx[1]}"

    pooled = model._pool_anchor(h, mask)
    expected = h[:, -1, :]
    assert torch.allclose(pooled, expected), "pool must match h[:, -1] when include_anchor_slot"

    legacy_pooled = h[torch.arange(2), legacy_idx]
    assert not torch.allclose(legacy_pooled[0], expected[0]), "legacy pools wrong row for padded sample"

    assert not torch.allclose(pooled[0], pooled[1]), "random h should differ across batch rows"

    # Without anchor slot: last True index
    model2 = CXREncoderTransformer(
        vit_path="google/vit-base-patch16-224-in21k",
        cxr_dim=512,
        d_model=64,
        num_transformer_layers=1,
        num_heads=2,
        include_anchor_slot=False,
        anchor_pool="last",
    ).to(device)
    mask2 = torch.tensor([[True, False, True]], device=device)
    h2 = torch.randn(1, 3, model2.d_model, device=device)
    pooled2 = model2._pool_anchor(h2, mask2)
    assert torch.allclose(pooled2, h2[:, 2, :]), "last True at index 2 when no anchor slot"

    print("test_pool_anchor_index: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
