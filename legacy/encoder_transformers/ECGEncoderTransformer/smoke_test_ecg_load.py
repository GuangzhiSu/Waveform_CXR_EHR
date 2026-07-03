#!/usr/bin/env python3
"""Smoke test: WFDB ECG load + frozen SignalEncoder (same stack as train.py)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "EHRWindowTransformer",
    _EXP,
    PROJECT_ROOT / "experiment1(old)",
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from common import collate_ecg_window_batch  # noqa: E402
from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT, ECG_DIM, ECG_TARGET_LEN  # noqa: E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    ap.add_argument("--n_anchors", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    print("=== ECGEncoderTransformer smoke test ===")
    print(f"  csv={args.ecg_labeled_csv}")

    ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv, ecg_target_len=ECG_TARGET_LEN)
    n = min(args.n_anchors, len(ds))
    loader = DataLoader(
        range(n),
        batch_size=min(args.batch_size, n),
        collate_fn=lambda idxs: collate_ecg_window_batch([ds[int(i)] for i in idxs]),
    )
    batch = next(iter(loader))
    mask_sum = int(batch["ecg_mask"].sum())
    mask_total = batch["ecg_mask"].numel()
    lens = batch["ecg_mask"].long().sum(dim=1)
    sig_mean = float(batch["ecg_seq"].abs().mean())
    print(f"  batch ecg_seq={tuple(batch['ecg_seq'].shape)}  mask={mask_sum}/{mask_total}")
    print(f"  seq_len min/median/max={int(lens.min())}/{int(lens.median())}/{int(lens.max())}  |ecg|_mean={sig_mean:.6f}")

    if mask_sum == 0:
        print("  FAIL: no valid ECG slots — check WFDB paths / group mount")
        return 1

    ckpt = args.ecg_ckpt if args.ecg_ckpt and os.path.isfile(args.ecg_ckpt) else None
    if ckpt:
        print(f"  ECG ckpt: {ckpt}")
    else:
        print(
            "  WARNING: ECG ckpt missing — encoder random init. "
            f"Expected under MedTVT-R1/CKPTS/ (set MEDTVT_ROOT or --ecg_ckpt)."
        )

    ecg_dim = ECG_DIM
    if ckpt and ckpt.endswith(".ckpt") and ecg_dim < 1024:
        ecg_dim = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim, freeze_ecg=True).to(device)
    n_frozen = sum(p.numel() for p in model.ecg_enc.parameters() if not p.requires_grad)
    n_enc = sum(p.numel() for p in model.ecg_enc.parameters())
    print(f"  encoder kind={model.ecg_encoder_kind}  ecg_dim={model.ecg_dim}  frozen={n_frozen}/{n_enc}")
    if ckpt and model.ecg_encoder_kind == "symile":
        w = model.ecg_enc.resnet.conv1.weight.detach()
        print(f"  symile conv1 |w|_mean={float(w.abs().mean()):.6f}  shape={tuple(w.shape)}")

    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
    if not torch.isfinite(log_s).all() or not torch.isfinite(log_p).all():
        print("  FAIL: non-finite logits")
        return 1
    print(f"  forward ok: log_s2f={tuple(log_s.shape)} finite=True  log_p2f finite=True")
    print("  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
