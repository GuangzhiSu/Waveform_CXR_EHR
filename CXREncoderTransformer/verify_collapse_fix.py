#!/usr/bin/env python3
"""Verify CXR collapse fixes: finite forward, mini-train diversity, image-shuffle sensitivity."""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
_EWT = PROJECT_ROOT / "EHRWindowTransformer"
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "EHRTrend",
    _EWT,
    _EXP,
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import stratified_train_val_test_indices  # noqa: E402
from common import collate_cxr_window_batch, stratify_labels_from_anchor  # noqa: E402
from config import (  # noqa: E402
    CXR_CATALOG_LABELED_CSV,
    CXR_ROOT,
    LR,
    METADATA_PATH,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    VIT_PATH,
)
from cxr_labeled_dataset import CXRLabeledCatalogDataset, CXRLabeledCatalogView  # noqa: E402
from model import CXREncoderTransformer  # noqa: E402
from train import _make_split_datasets  # noqa: E402


def _shuffle_images_batch(b: dict) -> dict:
    out = deepcopy(b)
    bsz = out["cxr_seq"].size(0)
    perm = torch.randperm(bsz, device=out["cxr_seq"].device)
    out["cxr_seq"] = out["cxr_seq"][perm]
    out["cxr_mask"] = out["cxr_mask"][perm]
    return out


def _synthetic_padded_forward(model: CXREncoderTransformer, device: torch.device) -> dict:
    """One valid CXR + padding + anchor slot — must stay finite."""
    cxr_seq = torch.randn(1, 3, 3, 224, 224, device=device)
    cxr_mask = torch.tensor([[True, False, False]], device=device)
    model.eval()
    with torch.no_grad():
        log_s, log_p = model(cxr_seq, cxr_mask)
    return {
        "finite": bool(torch.isfinite(log_s).all() and torch.isfinite(log_p).all()),
    }


def _mini_train(
    model: CXREncoderTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    steps: int,
    lr: float,
) -> dict:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    model.train()
    n_skipped = 0
    losses: list[float] = []
    step = 0
    for batch in train_loader:
        if batch is None:
            continue
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["cxr_seq"], b["cxr_mask"])
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        loss = (
            F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok])
            if s_ok.any()
            else log_s.new_tensor(0.0)
        )
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss))
        step += 1
        if step >= steps:
            break

    model.eval()
    y_true: list[int] = []
    y_pred_real: list[int] = []
    y_pred_shuf: list[int] = []
    pred_s = np.zeros(3, dtype=np.int64)
    n_nan = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, _ = model(b["cxr_seq"], b["cxr_mask"])
            b_shuf = _shuffle_images_batch(b)
            log_s_shuf, _ = model(b_shuf["cxr_seq"], b_shuf["cxr_mask"])
            if not (torch.isfinite(log_s).all() and torch.isfinite(log_s_shuf).all()):
                n_nan += 1
                continue
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            if s_ok.any():
                y_true.extend(b["anchor_s2f"][s_ok].cpu().numpy().tolist())
                y_pred_real.extend(log_s[s_ok].argmax(1).cpu().numpy().tolist())
                y_pred_shuf.extend(log_s_shuf[s_ok].argmax(1).cpu().numpy().tolist())
                for c in log_s[s_ok].argmax(1).cpu().numpy():
                    pred_s[int(c)] += 1

    y_true_a = np.asarray(y_true, dtype=np.int64)
    pred_real_a = np.asarray(y_pred_real, dtype=np.int64)
    pred_shuf_a = np.asarray(y_pred_shuf, dtype=np.int64)
    acc_real = float((pred_real_a == y_true_a).mean()) if y_true_a.size else 0.0
    acc_shuf = float((pred_shuf_a == y_true_a).mean()) if y_true_a.size else 0.0
    pred_change_rate = (
        float((pred_real_a != pred_shuf_a).mean()) if pred_real_a.size else 0.0
    )
    return {
        "n_steps": step,
        "n_skipped_nonfinite": n_skipped,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "val_nan_batches": n_nan,
        "n_unique_pred_s2f": int(np.count_nonzero(pred_s)),
        "pred_hist_s2f": pred_s.tolist(),
        "val_acc_s2f_real": acc_real,
        "val_acc_s2f_shuffled": acc_shuf,
        "acc_drop": acc_real - acc_shuf,
        "pred_change_rate": pred_change_rate,
    }


def _verify_config(
    include_anchor_slot: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    vit_path: str,
    steps: int,
    lr: float,
) -> dict:
    label = f"include_anchor_slot={include_anchor_slot}"
    print(f"\n=== {label} ===")
    model = CXREncoderTransformer(
        vit_path=vit_path,
        freeze_cxr=True,
        include_anchor_slot=include_anchor_slot,
    ).to(device)
    syn = _synthetic_padded_forward(model, device)
    print(f"  synthetic padded forward finite={syn['finite']}")
    mt = _mini_train(model, train_loader, val_loader, device, steps=steps, lr=lr)
    print(
        f"  mini-train steps={mt['n_steps']} skipped={mt['n_skipped_nonfinite']} "
        f"loss {mt['loss_first']:.4f}->{mt['loss_last']:.4f}"
        if mt["loss_first"] is not None and mt["loss_last"] is not None
        else f"  mini-train steps={mt['n_steps']}"
    )
    print(
        f"  val unique s2f={mt['n_unique_pred_s2f']}  "
        f"pred_hist={mt['pred_hist_s2f']}  acc_drop={mt['acc_drop']:.4f}  "
        f"pred_change_rate={mt['pred_change_rate']:.4f}"
    )
    image_sensitive = mt["acc_drop"] >= 0.05 or mt["pred_change_rate"] >= 0.15
    ok = (
        syn["finite"]
        and mt["n_skipped_nonfinite"] == 0
        and mt["val_nan_batches"] == 0
        and mt["n_unique_pred_s2f"] >= 2
        and mt["loss_last"] is not None
        and mt["loss_first"] is not None
        and mt["loss_last"] < mt["loss_first"]
        and image_sensitive
    )
    score = (
        mt["n_unique_pred_s2f"] * 10.0
        + mt["acc_drop"] * 100.0
        + mt["pred_change_rate"] * 50.0
        + (1.09 - mt["loss_last"]) * 10.0
        if mt["loss_last"] is not None
        else 0.0
    )
    return {
        "include_anchor_slot": include_anchor_slot,
        "verified": ok,
        "score": score,
        **mt,
        **syn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify CXR collapse fixes")
    ap.add_argument("--cxr_labeled_csv", default=CXR_CATALOG_LABELED_CSV)
    ap.add_argument("--cxr_root", default=CXR_ROOT)
    ap.add_argument("--metadata_path", default=METADATA_PATH)
    ap.add_argument("--vit_path", default=VIT_PATH)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mini_steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  max_samples={args.max_samples}  mini_steps={args.mini_steps}")

    full_ds = CXRLabeledCatalogDataset(
        labeled_csv=args.cxr_labeled_csv,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        cxr_split="train",
        imagenet_normalize=True,
    )
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        pick = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, pick.tolist())

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    y = stratify_labels_from_anchor(
        base.anchor_has_p2f, base.anchor_p2f_cls, base.anchor_has_s2f, base.anchor_s2f_cls
    )
    if isinstance(full_ds, Subset):
        y = y[np.array(full_ds.indices, dtype=np.int64)]
    test_split = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, TRAIN_SPLIT, VAL_SPLIT, test_split, args.seed
    )
    train_ds, val_ds, _, _ = _make_split_datasets(full_ds, idx_train, idx_val, idx_test)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cxr_window_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cxr_window_batch,
    )

    results = []
    for slot in (True, False):
        results.append(
            _verify_config(
                slot,
                train_loader,
                val_loader,
                device,
                vit_path=args.vit_path,
                steps=args.mini_steps,
                lr=args.lr,
            )
        )

    best = max(results, key=lambda r: r["score"])
    all_ok = all(r["verified"] for r in results)
    any_ok = any(r["verified"] for r in results)
    print(f"\nRecommended include_anchor_slot={best['include_anchor_slot']} (score={best['score']:.2f})")
    print(f"VERIFIED={any_ok}  (both_configs={all_ok})")
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
