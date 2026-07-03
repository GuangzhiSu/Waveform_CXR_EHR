#!/usr/bin/env python3
"""ECGEncoderTransformer collapse / NaN diagnostics: short-train A–E ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
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
from config import (  # noqa: E402
    ECG_CATALOG_LABELED_CSV,
    ECG_CKPT,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402
from train import (  # noqa: E402
    _head_class_weights,
    _hist_str,
    _param_l2,
    _restore_trainable,
    _snapshot_trainable,
    _trainable_finite,
    eval_loader,
    forward_loss_parts,
)


def _majority_baseline(labels: np.ndarray, num_classes: int = 3) -> float:
    if labels.size == 0:
        return 0.0
    cnt = np.bincount(labels, minlength=num_classes)
    return float(cnt.max()) / len(labels)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    if y_true.size == 0:
        return 0.0
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
    )


def _resolve_ckpt(path: Optional[str]) -> Optional[str]:
    if path and Path(path).is_file():
        return path
    if ECG_CKPT and Path(ECG_CKPT).is_file():
        return ECG_CKPT
    return None


def _build_loaders(args, device: torch.device):
    full_ds = ECGLabeledCatalogDataset(args.ecg_labeled_csv)
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
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ecg_window_batch,
    )
    return train_loader, val_loader, base, idx_train


def _train_mini(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    p2f_loss_weight: float,
    s2f_class_weight: Optional[torch.Tensor],
    p2f_class_weight: Optional[torch.Tensor],
    grad_clip: float,
    use_rollback: bool,
) -> dict:
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3
    )
    loss_kw = dict(
        p2f_loss_weight=p2f_loss_weight,
        s2f_class_weight=s2f_class_weight,
        p2f_class_weight=p2f_class_weight,
    )
    n_skipped_loss = n_skipped_grad = n_rollback = 0
    epoch_hist = []

    for epoch in range(epochs):
        epoch_snap = _snapshot_trainable(model) if use_rollback else None
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            step_snap = _snapshot_trainable(model) if use_rollback else None
            log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
            loss, _ = forward_loss_parts(b, log_s, log_p, **loss_kw)
            if not torch.isfinite(loss):
                n_skipped_loss += 1
                if use_rollback and step_snap is not None:
                    _restore_trainable(model, step_snap)
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            gn = sum(
                float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None
            ) ** 0.5
            if not np.isfinite(gn):
                n_skipped_grad += 1
                if use_rollback and step_snap is not None:
                    _restore_trainable(model, step_snap)
                opt.zero_grad()
                continue
            opt.step()
            if use_rollback and not _trainable_finite(model):
                n_rollback += 1
                if step_snap is not None:
                    _restore_trainable(model, step_snap)
                opt.zero_grad()
        if use_rollback and epoch_snap is not None and not _trainable_finite(model):
            _restore_trainable(model, epoch_snap)

        st = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)
        epoch_hist.append(
            {
                "epoch": epoch + 1,
                "val_acc_s2f": st["acc_s2f"],
                "val_loss": st["loss"],
                "pred_hist_s2f": st["pred_hist_s2f"].tolist(),
                "n_unique_pred_s2f": st.get("n_unique_pred_s2f", 0),
                "n_skipped_nonfinite": st.get("n_skipped_nonfinite", 0),
            }
        )

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, _ = model(b["ecg_seq"], b["ecg_mask"])
            if not torch.isfinite(log_s).all():
                continue
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            if s_ok.any():
                y_true.extend(b["anchor_s2f"][s_ok].cpu().numpy().tolist())
                y_pred.extend(log_s[s_ok].argmax(1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_unique = len(np.unique(y_pred)) if y_pred.size else 0
    param_l2 = _param_l2(model)
    st_final = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)
    return {
        "val_acc_s2f": float((y_pred == y_true).mean()) if y_true.size else 0.0,
        "majority_baseline": _majority_baseline(y_true),
        "macro_f1_s2f": _macro_f1(y_true, y_pred),
        "pred_hist_s2f": np.bincount(y_pred, minlength=3).tolist() if y_pred.size else [0, 0, 0],
        "n_unique_predictions": n_unique,
        "collapsed_to_one_class": n_unique <= 1,
        "param_l2_finite": bool(np.isfinite(param_l2)),
        "param_l2": param_l2,
        "val_loss_finite": bool(np.isfinite(st_final["loss"])),
        "n_skipped_nonfinite_loss": n_skipped_loss,
        "n_skipped_nonfinite_grad": n_skipped_grad,
        "n_rollback": n_rollback,
        "epoch_history": epoch_hist,
    }


def _passes(res: dict) -> bool:
    return (
        res["param_l2_finite"]
        and res["val_loss_finite"]
        and res["n_skipped_nonfinite_loss"] == 0
        and res["n_unique_predictions"] >= 3
        and res["macro_f1_s2f"] > 0.20
    )


def _new_model(ckpt: Optional[str], device: torch.device) -> ECGEncoderTransformer:
    ecg_dim = 1024 if ckpt and ckpt.endswith(".ckpt") else 512
    return ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim, freeze_ecg=True).to(device)


def run_experiments(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_ckpt(args.ecg_ckpt)
    print(
        f"diagnose_collapse  device={device}  ckpt={ckpt or 'none'}  "
        f"max_samples={args.max_samples}  epochs={args.epochs}"
    )
    train_loader, val_loader, base, idx_train = _build_loaders(args, device)
    s2f_w = _head_class_weights(
        base, idx_train, "anchor_has_s2f", "anchor_s2f_cls", 3, device
    )
    p2f_w = _head_class_weights(
        base, idx_train, "anchor_has_p2f", "anchor_p2f_cls", 3, device
    )

    specs = {
        "A_legacy_defaults": dict(
            lr=5e-4,
            p2f_loss_weight=10.0,
            s2f_class_weight=s2f_w,
            p2f_class_weight=p2f_w,
            grad_clip=5.0,
            use_rollback=False,
        ),
        "B_mild_loss": dict(
            lr=5e-4,
            p2f_loss_weight=1.0,
            s2f_class_weight=None,
            p2f_class_weight=None,
            grad_clip=5.0,
            use_rollback=False,
        ),
        "C_low_lr": dict(
            lr=1e-4,
            p2f_loss_weight=1.0,
            s2f_class_weight=None,
            p2f_class_weight=None,
            grad_clip=1.0,
            use_rollback=False,
        ),
        "D_combined_mask_is_default": dict(
            lr=1e-4,
            p2f_loss_weight=1.0,
            s2f_class_weight=None,
            p2f_class_weight=None,
            grad_clip=1.0,
            use_rollback=False,
        ),
        "E_stable_with_rollback": dict(
            lr=1e-4,
            p2f_loss_weight=1.0,
            s2f_class_weight=None,
            p2f_class_weight=None,
            grad_clip=1.0,
            use_rollback=True,
        ),
    }

    results = {
        "device": str(device),
        "ecg_ckpt": ckpt,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "experiments": {},
    }

    for name, kw in specs.items():
        print(f"\n=== {name} ===")
        model = _new_model(ckpt, device)
        res = _train_mini(model, train_loader, val_loader, device, epochs=args.epochs, **kw)
        res["pass"] = _passes(res)
        results["experiments"][name] = res
        print(
            f"  acc={res['val_acc_s2f']:.4f}  maj={res['majority_baseline']:.4f}  "
            f"macro_f1={res['macro_f1_s2f']:.4f}  unique={res['n_unique_predictions']}  "
            f"param_l2_finite={res['param_l2_finite']}  skipped_loss={res['n_skipped_nonfinite_loss']}"
        )
        print(f"  pred_s2f: {_hist_str(np.array(res['pred_hist_s2f']))}  pass={res['pass']}")

    winners = [k for k, v in results["experiments"].items() if v.get("pass")]
    results["winners"] = winners
    print(f"\nWinners (pass all criteria): {winners or 'none'}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnose_collapse.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    ap.add_argument("--ecg_ckpt", default=ECG_CKPT)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output_dir", default=str(_EXP / "output" / "diagnose_collapse"))
    args = ap.parse_args()
    res = run_experiments(args)
    any_pass = any(v.get("pass") for v in res["experiments"].values())
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
