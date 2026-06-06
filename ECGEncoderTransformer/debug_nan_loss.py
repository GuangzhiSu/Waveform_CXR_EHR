#!/usr/bin/env python3
"""Locate first training batch that yields NaN loss / logits in ECGEncoderTransformer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
_EWT = PROJECT_ROOT / "EHRWindowTransformer"
for _p in (PROJECT_ROOT, PROJECT_ROOT / "BaselineExperiment", _EWT, _EXP):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from common import collate_ecg_window_batch, stratify_labels_from_anchor  # noqa: E402
from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT, SEED, TRAIN_SPLIT, VAL_SPLIT  # noqa: E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def masked_ce(logits, y, valid):
    if not valid.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[valid], y[valid])


def scan_all_invalid_masks(ds) -> dict:
    n_all_invalid = 0
    n_partial = 0
    examples = []
    for i in range(len(ds)):
        item = ds[i]
        m = item["ecg_mask"]
        n_valid = int(m.sum())
        if n_valid == 0:
            n_all_invalid += 1
            if len(examples) < 5:
                examples.append(i)
        elif n_valid < int(m.numel()):
            n_partial += 1
    return {
        "n_samples": len(ds),
        "n_all_invalid": n_all_invalid,
        "n_partial_invalid": n_partial,
        "examples": examples,
    }


def check_forward(model, b, device):
    ecg_seq = b["ecg_seq"].to(device)
    ecg_mask = b["ecg_mask"].to(device)
    with torch.set_grad_enabled(model.training):
        log_s, log_p = model(ecg_seq, ecg_mask)
    s_ok = b["anchor_has_s2f"].to(device) & (b["anchor_s2f"].to(device) >= 0)
    p_ok = b["anchor_has_p2f"].to(device) & (b["anchor_p2f"].to(device) >= 0)
    loss_s = masked_ce(log_s, b["anchor_s2f"].to(device), s_ok)
    loss_p = masked_ce(log_p, b["anchor_p2f"].to(device), p_ok)
    loss = loss_s + loss_p
    all_invalid = (~ecg_mask.any(dim=1)).sum().item()
    return {
        "loss": float(loss),
        "loss_s2f": float(loss_s),
        "loss_p2f": float(loss_p),
        "finite_logits_s": bool(torch.isfinite(log_s).all()),
        "finite_logits_p": bool(torch.isfinite(log_p).all()),
        "n_all_invalid_rows": int(all_invalid),
        "ecg_seq_finite": bool(torch.isfinite(ecg_seq).all()),
        "ecg_nan_count": int((~torch.isfinite(ecg_seq)).sum()),
    }


def test_all_pad_attention(device):
    """Minimal repro: MultiheadAttention with all key_padding_mask=True -> NaN."""
    d_model, nhead, t, bsz = 256, 4, 3, 2
    attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True).to(device)
    x = torch.randn(bsz, t, d_model, device=device)
    pad = torch.ones(bsz, t, dtype=torch.bool, device=device)  # all padded
    out, _ = attn(x, x, x, key_padding_mask=pad, need_weights=False)
    return bool(torch.isfinite(out).all()), float(out.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    attn_ok, attn_max = test_all_pad_attention(device)
    print(
        f"[attention repro] all key_padding_mask=True -> finite={attn_ok}  |out|_max={attn_max}"
    )

    print(f"\nLoading dataset: {args.ecg_labeled_csv}")
    full_ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv)
    stats = scan_all_invalid_masks(full_ds)
    print(
        f"[mask scan] n={stats['n_samples']:,}  all_invalid={stats['n_all_invalid']:,}  "
        f"partial={stats['n_partial_invalid']:,}  example_idxs={stats['examples']}"
    )

    base = full_ds
    y = stratify_labels_from_anchor(
        base.anchor_has_p2f, base.anchor_p2f_cls, base.anchor_has_s2f, base.anchor_s2f_cls
    )
    test_split = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, _, _ = stratified_train_val_test_indices(y, TRAIN_SPLIT, VAL_SPLIT, test_split, args.seed)
    train_ds = make_subset(full_ds, idx_train)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ecg_window_batch,
    )
    torch.manual_seed(args.seed)

    ckpt = args.ecg_ckpt if Path(args.ecg_ckpt).is_file() else None
    ecg_dim = 1024 if ckpt and ckpt.endswith(".ckpt") else 512
    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim, freeze_ecg=True).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-3)

    print(f"\nScanning up to {args.max_batches} train batches (batch_size={args.batch_size})...")
    first_bad = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        b = batch
        model.eval()
        ev = check_forward(model, b, device)
        if not ev["finite_logits_s"] or not ev["finite_logits_p"] or not np.isfinite(ev["loss"]):
            first_bad = (batch_idx, "eval", ev, b)
            break

        model.train()
        b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        log_s, log_p = model(b_dev["ecg_seq"], b_dev["ecg_mask"])
        s_ok = b_dev["anchor_has_s2f"] & (b_dev["anchor_s2f"] >= 0)
        p_ok = b_dev["anchor_has_p2f"] & (b_dev["anchor_p2f"] >= 0)
        loss = masked_ce(log_s, b_dev["anchor_s2f"], s_ok) + masked_ce(log_p, b_dev["anchor_p2f"], p_ok)
        if not torch.isfinite(loss):
            first_bad = (batch_idx, "train_forward", check_forward(model, b, device), b)
            break
        opt.zero_grad()
        loss.backward()
        gn = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        if not np.isfinite(gn):
            first_bad = (batch_idx, "grad_nan", {"grad_norm": gn}, b)
            break
        if batch_idx == 0:
            print(f"  batch0 ok: loss={float(loss):.4f} grad_norm={gn:.4f} {ev}")

    if first_bad is None:
        print(f"No NaN in first {min(batch_idx + 1, args.max_batches)} batches.")
        return 0

    bi, stage, info, batch = first_bad
    print(f"\n*** FIRST FAILURE batch_idx={bi} stage={stage} ***")
    for k, v in info.items():
        print(f"  {k}: {v}")
    lens = batch["ecg_mask"].long().sum(dim=1)
    valid_per_row = batch["ecg_mask"].sum(dim=1)
    print(f"  seq_len min/med/max={int(lens.min())}/{int(lens.median())}/{int(lens.max())}")
    print(f"  valid_ecg per row: min={int(valid_per_row.min())} max={int(valid_per_row.max())}")
    all_inv = (~batch["ecg_mask"].any(dim=1)).nonzero(as_tuple=False).view(-1).tolist()
    print(f"  rows with ALL ecg_mask=False: {all_inv}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
