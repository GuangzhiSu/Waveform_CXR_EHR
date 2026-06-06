#!/usr/bin/env python3
"""Verify ECG collapse fixes: mask patterns, finite forward, mini-train pred diversity."""
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

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from common import collate_ecg_window_batch, stratify_labels_from_anchor  # noqa: E402
from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT, SEED, TRAIN_SPLIT, VAL_SPLIT  # noqa: E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def _synthetic_leading_invalid_forward(model, device) -> dict:
    """Mask [False, True] + anchor should be finite with split-mask forward."""
    ecg_seq = torch.randn(1, 2, 12, 1000, device=device)
    ecg_mask = torch.tensor([[False, True]], device=device)
    model.eval()
    with torch.no_grad():
        log_s, log_p = model(ecg_seq, ecg_mask)
    return {
        "finite": bool(torch.isfinite(log_s).all() and torch.isfinite(log_p).all()),
        "log_s": log_s.cpu().tolist(),
    }


@torch.no_grad()
def _anchor_vec_stats(model, batch, device) -> dict:
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    _, _, av = model(b["ecg_seq"], b["ecg_mask"], return_anchor_vec=True)
    return {
        "anchor_vec_std_mean": float(av.std(dim=0).mean()),
        "anchor_vec_std_min": float(av.std(dim=0).min()),
    }


def _mini_train_pred_diversity(
    model,
    train_loader,
    val_loader,
    device,
    *,
    steps: int = 80,
    lr: float = 1e-4,
) -> dict:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    model.train()
    n_skipped = 0
    losses = []
    step = 0
    for batch in train_loader:
        if batch is None:
            continue
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        loss_s = F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok]) if s_ok.any() else log_s.new_tensor(0.0)
        loss_p = F.cross_entropy(log_p[p_ok], b["anchor_p2f"][p_ok]) if p_ok.any() else log_p.new_tensor(0.0)
        loss = loss_s + loss_p
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        losses.append(float(loss))
        step += 1
        if step >= steps:
            break

    model.eval()
    pred_s = pred_p = np.zeros(3, dtype=np.int64)
    n_nan = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
            if not (torch.isfinite(log_s).all() and torch.isfinite(log_p).all()):
                n_nan += 1
                continue
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
            if s_ok.any():
                for c in log_s[s_ok].argmax(1).cpu().numpy():
                    pred_s[int(c)] += 1
            if p_ok.any():
                for c in log_p[p_ok].argmax(1).cpu().numpy():
                    pred_p[int(c)] += 1

    return {
        "n_steps": step,
        "n_skipped_nonfinite": n_skipped,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "val_nan_batches": n_nan,
        "n_unique_pred_s2f": int(np.count_nonzero(pred_s)),
        "n_unique_pred_p2f": int(np.count_nonzero(pred_p)),
        "pred_hist_s2f": pred_s.tolist(),
        "pred_hist_p2f": pred_p.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mini_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt = args.ecg_ckpt if Path(args.ecg_ckpt).is_file() else None
    ecg_dim = 1024 if ckpt and ckpt.endswith(".ckpt") else 512
    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim, freeze_ecg=True).to(device)

    syn = _synthetic_leading_invalid_forward(model, device)
    print(f"[1] synthetic [F,T] mask forward finite={syn['finite']}")

    print(f"\nLoading dataset max_samples={args.max_samples}...")
    full_ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv)
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        idxs = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, idxs.tolist())

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    y = stratify_labels_from_anchor(
        base.anchor_has_p2f, base.anchor_p2f_cls, base.anchor_has_s2f, base.anchor_s2f_cls
    )
    if isinstance(full_ds, Subset):
        y = y[np.array(full_ds.indices, dtype=np.int64)]
    test_split = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, idx_val, _ = stratified_train_val_test_indices(
        y, TRAIN_SPLIT, VAL_SPLIT, test_split, args.seed
    )
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ecg_window_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ecg_window_batch,
    )

    batch = next(iter(train_loader))
    av = _anchor_vec_stats(model, batch, device)
    print(f"[2] train batch0 anchor_vec std mean={av['anchor_vec_std_mean']:.6f}")

    mt = _mini_train_pred_diversity(
        model, train_loader, val_loader, device, steps=args.mini_steps
    )
    print(f"[3] mini-train steps={mt['n_steps']} skipped={mt['n_skipped_nonfinite']} "
          f"loss {mt['loss_first']:.4f}->{mt['loss_last']:.4f}")
    print(f"    val unique pred s2f/p2f={mt['n_unique_pred_s2f']}/{mt['n_unique_pred_p2f']} "
          f"val_nan_batches={mt['val_nan_batches']}")
    print(f"    pred_hist_s2f={mt['pred_hist_s2f']} pred_hist_p2f={mt['pred_hist_p2f']}")

    ok = (
        syn["finite"]
        and mt["n_skipped_nonfinite"] == 0
        and mt["val_nan_batches"] == 0
        and mt["n_unique_pred_s2f"] >= 2
        and mt["loss_last"] is not None
        and mt["loss_last"] < (mt["loss_first"] or 999)
    )
    print(f"\nVERIFIED={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
