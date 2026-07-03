#!/usr/bin/env python3
"""Verify CXREncoderTransformer loads pretrained (frozen) ViT weights, not random init."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "BaselineExperiment" / "CXRUni",
    _EXP,
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from config import VIT_PATH  # noqa: E402
from cxr_classification.model import CXRClassificationBaseline  # noqa: E402
from model import CXREncoderTransformer  # noqa: E402
from models.encoders.cxr import CXREncoder  # noqa: E402


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def _compare_vit_state_dicts(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    label: str,
    atol: float = 1e-5,
) -> bool:
    left_keys = set(left.keys())
    right_keys = set(right.keys())
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)
        print(f"  FAIL [{label}]: key mismatch missing={missing[:5]} extra={extra[:5]}")
        return False

    worst_key = ""
    worst_diff = 0.0
    for key in left_keys:
        diff = _max_abs_diff(left[key], right[key])
        if diff > worst_diff:
            worst_diff = diff
            worst_key = key
    ok = worst_diff <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  {status} [{label}]: max_abs_diff={worst_diff:.2e}  worst_key={worst_key}")
    return ok


def _vit_state_dict(vit: ViTModel) -> dict[str, torch.Tensor]:
    return {k: v.cpu() for k, v in vit.state_dict().items()}


def verify_pretrained_loaded(vit_path: str, freeze_cxr: bool = True) -> int:
    print("=== CXREncoderTransformer ViT load smoke test ===")
    print(f"  vit_path={vit_path}")

    config = ViTConfig.from_pretrained(vit_path)
    if hasattr(config, "add_pooling_layer"):
        config.add_pooling_layer = False

    ref_vit = ViTModel.from_pretrained(vit_path, config=config)
    ref_sd = _vit_state_dict(ref_vit)

    random_vit = ViTModel(config)
    random_sd = _vit_state_dict(random_vit)
    random_diff = max(_max_abs_diff(ref_sd[k], random_sd[k]) for k in ref_sd)
    print(f"  pretrained vs random-init max_abs_diff={random_diff:.4f} (expect >> 1e-3)")
    if random_diff < 1e-3:
        print("  FAIL: pretrained weights look identical to random init")
        return 1

    cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=512, freeze=freeze_cxr)
    ok_enc = _compare_vit_state_dicts(
        ref_sd, _vit_state_dict(cxr_enc.vit), "CXREncoder.vit vs ViTModel.from_pretrained"
    )

    baseline = CXRClassificationBaseline(
        num_classes=3, hidden_dim=512, vit_path=vit_path, freeze_encoder=freeze_cxr
    )
    ok_base = _compare_vit_state_dicts(
        ref_sd,
        _vit_state_dict(baseline.cxr_encoder.vit),
        "CXRClassificationBaseline.vit vs ViTModel.from_pretrained",
    )
    ok_cross = _compare_vit_state_dicts(
        _vit_state_dict(cxr_enc.vit),
        _vit_state_dict(baseline.cxr_encoder.vit),
        "CXREncoderTransformer stack vs baseline CXREncoder",
    )

    model = CXREncoderTransformer(vit_path=vit_path, freeze_cxr=freeze_cxr, cxr_dim=512)
    ok_model = _compare_vit_state_dicts(
        ref_sd,
        _vit_state_dict(model.cxr_enc.vit),
        "CXREncoderTransformer.cxr_enc.vit vs ViTModel.from_pretrained",
    )

    vit_params = list(model.cxr_enc.vit.parameters())
    n_frozen = sum(p.numel() for p in vit_params if not p.requires_grad)
    n_total = sum(p.numel() for p in vit_params)
    print(f"  ViT frozen params: {n_frozen:,}/{n_total:,}  freeze_cxr={freeze_cxr}")
    if freeze_cxr and n_frozen != n_total:
        print("  FAIL: expected all ViT params frozen")
        return 1

    proj_trainable = model.cxr_enc.proj.weight.requires_grad
    print(f"  cxr_enc.proj trainable={proj_trainable} (expected True)")
    if not proj_trainable:
        print("  FAIL: projection layer should remain trainable")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    x = torch.randn(2, 3, 224, 224, device=device)
    mask = torch.ones(2, 1, dtype=torch.bool, device=device)
    seq = x.unsqueeze(1)
    log_s, log_p = model(seq, mask)
    loss = log_s.mean() + log_p.mean()
    loss.backward()
    vit_grads = [p.grad for p in model.cxr_enc.vit.parameters() if p.grad is not None]
    proj_grad = model.cxr_enc.proj.weight.grad
    print(f"  backward: vit_grad_tensors={len(vit_grads)} (expect 0 when frozen)  proj_grad={'ok' if proj_grad is not None else 'missing'}")
    if freeze_cxr and vit_grads:
        print("  FAIL: frozen ViT received gradients")
        return 1
    if proj_grad is None:
        print("  FAIL: trainable projection did not receive gradients")
        return 1
    if not torch.isfinite(log_s).all() or not torch.isfinite(log_p).all():
        print("  FAIL: non-finite logits")
        return 1
    print(f"  forward ok: log_s2f={tuple(log_s.shape)} log_p2f={tuple(log_p.shape)}")

    if ok_enc and ok_base and ok_cross and ok_model:
        print("  PASS: CXREncoderTransformer uses pretrained frozen ViT weights")
        return 0
    print("  FAIL: ViT weight checks did not all pass")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit_path", default=VIT_PATH)
    ap.add_argument("--unfreeze_cxr", action="store_true")
    args = ap.parse_args()
    return verify_pretrained_loaded(args.vit_path, freeze_cxr=not args.unfreeze_cxr)


if __name__ == "__main__":
    raise SystemExit(main())
