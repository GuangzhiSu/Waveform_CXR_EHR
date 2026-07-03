#!/usr/bin/env python
"""Smoke-test ECGLabeledCatalogDataset + collate + one forward pass."""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
_EWT = PROJECT_ROOT / "EHRWindowTransformer"
for _p in (PROJECT_ROOT, _EWT, _EXP):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from common import collate_ecg_window_batch  # noqa: E402
from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT, ECG_TARGET_LEN  # noqa: E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--max_samples", type=int, default=8)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    args = ap.parse_args()

    ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv, ecg_target_len=ECG_TARGET_LEN)
    n = min(args.max_samples, len(ds))
    print(f"Dataset len={len(ds):,}  smoke n={n}")

    loader = DataLoader(
        range(n),
        batch_size=min(4, n),
        collate_fn=lambda idxs: collate_ecg_window_batch([ds[int(i)] for i in idxs]),
    )
    batch = next(iter(loader))
    print(f"  ecg_seq={tuple(batch['ecg_seq'].shape)}  mask sum={int(batch['ecg_mask'].sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ecg_ckpt if Path(args.ecg_ckpt).is_file() else None
    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt).to(device)
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
    print(f"  forward ok: log_s2f={tuple(log_s.shape)} log_p2f={tuple(log_p.shape)}  device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
