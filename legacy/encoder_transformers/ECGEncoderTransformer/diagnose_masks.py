#!/usr/bin/env python3
"""Scan ECG mask patterns (leading-invalid / internal gaps) and link to non-finite forward."""
from __future__ import annotations

import argparse
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
    PROJECT_ROOT / "CXREncoderTransformer",
    PROJECT_ROOT / "EHRWindowTransformer",
    _EXP,
    PROJECT_ROOT / "experiment1(old)",
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from common import collate_ecg_window_batch  # noqa: E402
from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT  # noqa: E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def _mask_stats(m: torch.Tensor) -> dict:
    """Per-sample mask diagnostics (1D bool tensor)."""
    n = int(m.numel())
    n_valid = int(m.sum())
    if n_valid == 0:
        return {"n_valid": 0, "leading_invalid": False, "internal_gap": False, "first_true": -1}
    idx = m.nonzero(as_tuple=False).view(-1).cpu().numpy()
    first_true = int(idx[0])
    leading_invalid = first_true > 0
    internal_gap = False
    if len(idx) > 1:
        for i in range(len(idx) - 1):
            if idx[i + 1] - idx[i] > 1:
                internal_gap = True
                break
    return {
        "n_valid": n_valid,
        "leading_invalid": leading_invalid,
        "internal_gap": internal_gap,
        "first_true": first_true,
    }


def scan_dataset(ds, max_samples: int = 0) -> dict:
    n = len(ds)
    if max_samples and max_samples < n:
        n = max_samples
    n_leading = n_gap = n_all_invalid = 0
    leading_idxs: list = []
    for i in range(n):
        item = ds[i]
        st = _mask_stats(item["ecg_mask"])
        if st["n_valid"] == 0:
            n_all_invalid += 1
        if st["leading_invalid"]:
            n_leading += 1
            if len(leading_idxs) < 10:
                leading_idxs.append(i)
        if st["internal_gap"]:
            n_gap += 1
    return {
        "n_scanned": n,
        "n_leading_invalid": n_leading,
        "n_internal_gap": n_gap,
        "n_all_invalid": n_all_invalid,
        "leading_example_idxs": leading_idxs,
    }


@torch.no_grad()
def forward_finite(model, item: dict, device: torch.device) -> bool:
    ecg_seq = item["ecg_seq"].unsqueeze(0).to(device)
    ecg_mask = item["ecg_mask"].unsqueeze(0).to(device)
    log_s, log_p = model(ecg_seq, ecg_mask)
    return bool(torch.isfinite(log_s).all() and torch.isfinite(log_p).all())


def scan_batches_for_nan(model, loader, device, max_batches: int = 500) -> dict:
    n_nan = 0
    n_leading_in_batch = 0
    first_nan = None
    for bi, batch in enumerate(loader):
        if bi >= max_batches or batch is None:
            break
        for i in range(batch["ecg_mask"].size(0)):
            st = _mask_stats(batch["ecg_mask"][i])
            if st["leading_invalid"]:
                n_leading_in_batch += 1
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
        finite = torch.isfinite(log_s).all() and torch.isfinite(log_p).all()
        if not finite:
            n_nan += 1
            if first_nan is None:
                bad_rows = []
                for i in range(b["ecg_mask"].size(0)):
                    ls = log_s[i]
                    if not torch.isfinite(ls).all():
                        bad_rows.append(i)
                first_nan = {
                    "batch_idx": bi,
                    "bad_rows": bad_rows,
                    "masks": [batch["ecg_mask"][r].tolist() for r in bad_rows[:3]],
                }
    return {
        "n_batches": min(bi + 1, max_batches) if batch is not None else 0,
        "n_nan_batches": n_nan,
        "n_leading_invalid_rows_in_batches": n_leading_in_batch,
        "first_nan": first_nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"\nLoading dataset (scan up to {args.max_samples} anchors)...")
    ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv)
    stats = scan_dataset(ds, args.max_samples)
    print(f"[mask scan] n={stats['n_scanned']:,}")
    print(f"  leading_invalid (first True idx > 0): {stats['n_leading_invalid']:,}")
    print(f"  internal_gap (False between True):    {stats['n_internal_gap']:,}")
    print(f"  all_invalid rows:                     {stats['n_all_invalid']:,}")
    print(f"  leading_invalid examples: {stats['leading_example_idxs']}")

    ckpt = args.ecg_ckpt if Path(args.ecg_ckpt).is_file() else None
    ecg_dim = 1024 if ckpt and ckpt.endswith(".ckpt") else 512
    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim, freeze_ecg=True).to(device)
    model.eval()

    if stats["leading_example_idxs"]:
        print("\n[forward on leading-invalid examples]")
        for idx in stats["leading_example_idxs"][:5]:
            item = ds[idx]
            ok = forward_finite(model, item, device)
            m = item["ecg_mask"].tolist()
            print(f"  idx={idx} mask={m} finite={ok}")

    loader = DataLoader(
        range(min(len(ds), args.max_samples)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda idxs: collate_ecg_window_batch([ds[int(i)] for i in idxs]),
    )
    print(f"\n[batch forward scan] up to {args.max_batches} batches, batch_size={args.batch_size}")
    bstats = scan_batches_for_nan(model, loader, device, args.max_batches)
    print(f"  batches scanned: {bstats['n_batches']}")
    print(f"  NaN logit batches: {bstats['n_nan_batches']}")
    print(f"  leading-invalid rows seen: {bstats['n_leading_invalid_rows_in_batches']}")
    if bstats["first_nan"]:
        print(f"  first NaN batch: {bstats['first_nan']}")

    ok = stats["n_leading_invalid"] == 0 or bstats["n_nan_batches"] > 0
    print(f"\nDIAGNOSIS={'leading-invalid correlates with NaN risk' if ok else 'inconclusive'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
