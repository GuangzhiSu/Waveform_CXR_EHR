#!/usr/bin/env python3
"""Diagnose EHREncoderTransformer flat-loss / prediction-collapse (quick experiments)."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
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
for _p in (PROJECT_ROOT, PROJECT_ROOT / "BaselineExperiment", PROJECT_ROOT / "EHRTrend", _EXP):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import *  # noqa: F401,F403,E402
from ehr_symile_dataset import EHRNextStepDatasetSymile  # noqa: E402
from model import EHREncoderTransformer  # noqa: E402
from models.encoders.ehr import EHRMLPEncoder  # noqa: E402
from train import (  # noqa: E402
    _grad_norm,
    _head_class_weights,
    _hist_str,
    _param_delta_l2,
    _snapshot_trainable,
    _stratify_labels_from_dataset,
    collate_anchor_batch,
    eval_loader,
    forward_loss_parts,
    masked_ce,
)


class EHRRowLinearProbe(nn.Module):
    """Row MLP mean-pool + linear s2f head (no transformer; diagnostic only)."""

    def __init__(self, input_dim: int, embed_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.row_encoder = EHRMLPEncoder(input_dim=input_dim, embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor) -> torch.Tensor:
        z = self.row_encoder(ehr_seq) * ehr_mask.unsqueeze(-1).to(dtype=ehr_seq.dtype)
        denom = ehr_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = z.sum(dim=1) / denom
        return self.head(pooled)


def _majority_baseline(labels: np.ndarray, num_classes: int = 3) -> float:
    if labels.size == 0:
        return 0.0
    cnt = np.bincount(labels, minlength=num_classes)
    return float(cnt.max()) / len(labels)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    if y_true.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0))


def _build_loaders(args, device: torch.device):
    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        enr = None

    full_ds = EHRNextStepDatasetSymile(
        anchor_source_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        enriched_csv=enr,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        include_anchor_row=args.include_anchor_row,
    )
    y = _stratify_labels_from_dataset(full_ds)
    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    full_ds.fit_preprocess(idx_train)

    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        pick = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, pick.tolist())
        y = y[pick]
        idx_train, idx_val, idx_test = stratified_train_val_test_indices(
            y, args.train_split, args.val_split, test_split, args.seed
        )

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    if isinstance(full_ds, Subset):
        _map = np.asarray(full_ds.indices, dtype=np.int64)
        idx_train_base = _map[idx_train]
    else:
        idx_train_base = idx_train

    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
    )
    s2f_w = _head_class_weights(
        base, idx_train_base, "anchor_has_s2f", "anchor_s2f_cls", args.num_classes, device
    )
    input_dim = base.input_dim
    return train_loader, val_loader, base, idx_train_base, s2f_w, input_dim


def _make_model(input_dim: int, args) -> EHREncoderTransformer:
    return EHREncoderTransformer(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        anchor_pool=args.anchor_pool,
    )


def _train_mini(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
    p2f_loss_weight: float = 10.0,
    grad_clip: float = 0.0,
    label_smoothing: float = 0.0,
    subset_loader: Optional[DataLoader] = None,
) -> dict:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    loss_kw = dict(
        p2f_loss_weight=p2f_loss_weight,
        s2f_class_weight=s2f_class_weight,
        p2f_class_weight=p2f_class_weight,
        label_smoothing=label_smoothing,
    )
    epoch_hist = []
    tr_loader = subset_loader or train_loader

    for epoch in range(epochs):
        model.train()
        tr_loss = tr_uw_s = 0.0
        n_batches = 0
        for batch in tr_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if isinstance(model, EHRRowLinearProbe):
                log_s = model(b["ehr_seq"], b["ehr_mask"])
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                loss = masked_ce(log_s, b["anchor_s2f"], s_ok, s2f_class_weight, label_smoothing)
                uw = masked_ce(log_s, b["anchor_s2f"], s_ok, None, label_smoothing)
            else:
                loss, parts = forward_loss_parts(model, b, **loss_kw)
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
                uw_s = masked_ce(parts["log_s2f"], b["anchor_s2f"], s_ok, None, label_smoothing)
                uw_p = masked_ce(parts["log_p2f"], b["anchor_p2f"], p_ok, None, label_smoothing)
                n_s, n_p = int(s_ok.sum()), int(p_ok.sum())
                if n_s and n_p:
                    uw = (uw_s * n_s + uw_p * p2f_loss_weight * n_p) / (n_s + p2f_loss_weight * n_p)
                elif n_s:
                    uw = uw_s
                elif n_p:
                    uw = uw_p
                else:
                    uw = uw_s + uw_p
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
            tr_loss += float(loss)
            tr_uw_s += float(uw)
            n_batches += 1

        if isinstance(model, EHRRowLinearProbe):
            model.eval()
            ce_sum = acc_n = acc_d = 0.0
            pred_hist = np.zeros(3, dtype=np.int64)
            with torch.no_grad():
                for batch in val_loader:
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    log_s = model(b["ehr_seq"], b["ehr_mask"])
                    s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                    if s_ok.any():
                        ce_sum += float(F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok]))
                        pred = log_s[s_ok].argmax(1)
                        acc_n += (pred == b["anchor_s2f"][s_ok]).float().sum().item()
                        acc_d += int(s_ok.sum())
                        for c in pred.cpu().numpy():
                            pred_hist[int(c)] += 1
            st = {"loss": ce_sum / max(len(val_loader), 1), "acc_s2f": acc_n / max(acc_d, 1), "pred_hist_s2f": pred_hist}
        else:
            st = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)

        epoch_hist.append(
            {
                "epoch": epoch + 1,
                "train_loss": tr_loss / max(n_batches, 1),
                "train_loss_uw": tr_uw_s / max(n_batches, 1),
                "val_acc_s2f": st["acc_s2f"],
                "val_loss": st["loss"],
                "pred_hist_s2f": st["pred_hist_s2f"].tolist(),
            }
        )

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if isinstance(model, EHRRowLinearProbe):
                log_s = model(b["ehr_seq"], b["ehr_mask"])
            else:
                log_s, _ = model(b["ehr_seq"], b["ehr_mask"])
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            if s_ok.any():
                y_true.extend(b["anchor_s2f"][s_ok].cpu().numpy().tolist())
                y_pred.extend(log_s[s_ok].argmax(1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    maj = _majority_baseline(y_true)
    n_unique = len(np.unique(y_pred)) if y_pred.size else 0
    return {
        "val_acc_s2f": float((y_pred == y_true).mean()) if y_true.size else 0.0,
        "majority_baseline": maj,
        "macro_f1_s2f": _macro_f1(y_true, y_pred),
        "pred_hist_s2f": np.bincount(y_pred, minlength=3).tolist() if y_pred.size else [0, 0, 0],
        "n_unique_predictions": n_unique,
        "collapsed_to_one_class": bool(n_unique == 1),
        "epoch_history": epoch_hist,
    }


def _exp_a_gradient_check(model, train_loader, device, s2f_w, args) -> dict:
    """Single-step gradient and parameter update check."""
    model.train()
    batch = next(iter(train_loader))
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss_kw = dict(
        p2f_loss_weight=args.p2f_loss_weight,
        s2f_class_weight=s2f_w,
        p2f_class_weight=None,
    )
    snap = _snapshot_trainable(model)
    loss, parts = forward_loss_parts(model, b, **loss_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    opt.zero_grad()
    loss.backward()
    grad_norm = _grad_norm(model)
    opt.step()
    param_delta = _param_delta_l2(model, snap)
    s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
    pred_hist = np.bincount(parts["log_s2f"][s_ok].argmax(1).cpu().numpy(), minlength=3) if s_ok.any() else np.zeros(3)
    return {
        "loss_weighted": float(loss),
        "loss_s2f": float(parts["loss_s2f"]),
        "loss_p2f": float(parts["loss_p2f"]),
        "grad_norm": grad_norm,
        "param_delta_after_1step": param_delta,
        "batch_pred_s2f": pred_hist.tolist(),
        "pass": bool(grad_norm > 1e-6 and param_delta > 1e-8),
    }


def _exp_e_weighted_vs_unweighted(model, train_loader, device, s2f_w, args) -> dict:
    """Compare weighted vs unweighted CE over first N batches."""
    model.eval()
    w_losses, uw_losses = [], []
    loss_kw = dict(
        p2f_loss_weight=args.p2f_loss_weight,
        s2f_class_weight=s2f_w,
        p2f_class_weight=None,
    )
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= args.ce_compare_batches:
                break
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, parts = forward_loss_parts(model, b, **loss_kw)
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
            w_total, _ = forward_loss_parts(model, b, **loss_kw)
            uw_s = masked_ce(parts["log_s2f"], b["anchor_s2f"], s_ok, None)
            uw_p = masked_ce(parts["log_p2f"], b["anchor_p2f"], p_ok, None)
            n_s, n_p = int(s_ok.sum()), int(p_ok.sum())
            if n_s and n_p:
                uw = (uw_s * n_s + uw_p * args.p2f_loss_weight * n_p) / (n_s + args.p2f_loss_weight * n_p)
            elif n_s:
                uw = uw_s
            elif n_p:
                uw = uw_p
            else:
                uw = uw_s + uw_p
            w_losses.append(float(w_total))
            uw_losses.append(float(uw))

    w_mean = float(np.mean(w_losses)) if w_losses else 0.0
    uw_mean = float(np.mean(uw_losses)) if uw_losses else 0.0
    near_ln3 = bool(abs(w_mean - np.log(3)) < 0.05)
    passed = bool(uw_mean < w_mean - 0.05 or uw_mean < 1.0)
    return {
        "n_batches": len(w_losses),
        "weighted_ce_mean": w_mean,
        "unweighted_ce_mean": uw_mean,
        "near_ln3_weighted": near_ln3,
        "pass": passed,
    }


def run_experiments(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"diagnose_training  device={device}  max_samples={args.max_samples}")
    train_loader, val_loader, base, idx_train, s2f_w, input_dim = _build_loaders(args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "device": str(device),
        "max_samples": args.max_samples,
        "input_dim": input_dim,
        "experiments": {},
    }

    print("\n=== A: gradient / parameter update check ===")
    model_a = _make_model(input_dim, args).to(device)
    res_a = _exp_a_gradient_check(model_a, train_loader, device, s2f_w, args)
    results["experiments"]["A_gradient_check"] = res_a
    print(
        f"  grad_norm={res_a['grad_norm']:.6f}  param_delta={res_a['param_delta_after_1step']:.8f}  "
        f"pass={res_a['pass']}"
    )

    print("\n=== E: weighted vs unweighted CE (untrained model) ===")
    model_e0 = _make_model(input_dim, args).to(device)
    res_e = _exp_e_weighted_vs_unweighted(model_e0, train_loader, device, s2f_w, args)
    results["experiments"]["E_weighted_vs_unweighted"] = res_e
    print(
        f"  weighted={res_e['weighted_ce_mean']:.4f}  unweighted={res_e['unweighted_ce_mean']:.4f}  "
        f"near_ln3={res_e['near_ln3_weighted']}"
    )

    print("\n=== B: train 3 epochs WITHOUT class weights ===")
    model_b = _make_model(input_dim, args).to(device)
    res_b = _train_mini(
        model_b, train_loader, val_loader, device,
        epochs=args.epochs_no_cw, lr=args.lr,
        s2f_class_weight=None, p2f_class_weight=None,
        p2f_loss_weight=args.p2f_loss_weight, grad_clip=args.grad_clip,
    )
    hist = res_b["epoch_history"]
    loss_drop = 0.0
    if len(hist) >= 2:
        loss_drop = (hist[0]["train_loss_uw"] - hist[-1]["train_loss_uw"]) / max(hist[0]["train_loss_uw"], 1e-6)
    res_b["train_loss_uw_drop_pct"] = float(loss_drop * 100.0)
    res_b["pass"] = bool(loss_drop > 0.05 and res_b["val_acc_s2f"] >= res_b["majority_baseline"] - 0.02)
    results["experiments"]["B_no_class_weights"] = res_b
    print(
        f"  acc={res_b['val_acc_s2f']:.4f}  maj={res_b['majority_baseline']:.4f}  "
        f"uw_loss_drop={loss_drop*100:.1f}%  pass={res_b['pass']}"
    )
    if hist:
        print(f"  epoch1 uw={hist[0]['train_loss_uw']:.4f}  epoch{len(hist)} uw={hist[-1]['train_loss_uw']:.4f}")

    print("\n=== C: row-encoder linear probe (no transformer) ===")
    probe = EHRRowLinearProbe(input_dim, embed_dim=args.embed_dim, num_classes=args.num_classes).to(device)
    res_c = _train_mini(
        probe, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, s2f_class_weight=None, grad_clip=args.grad_clip,
    )
    res_c["pass"] = bool(res_c["val_acc_s2f"] > 0.35)
    results["experiments"]["C_row_linear_probe"] = res_c
    print(
        f"  acc={res_c['val_acc_s2f']:.4f}  maj={res_c['majority_baseline']:.4f}  "
        f"macro_f1={res_c['macro_f1_s2f']:.4f}  pass={res_c['pass']}"
    )
    print(f"  pred_s2f: {_hist_str(np.array(res_c['pred_hist_s2f']))}")

    print("\n=== D: mini overfit (2k train subset) ===")
    overfit_n = min(args.overfit_samples, len(train_loader.dataset))
    rng = np.random.RandomState(args.seed)
    sub_idx = rng.choice(len(train_loader.dataset), size=overfit_n, replace=False)
    overfit_ds = Subset(train_loader.dataset, sub_idx.tolist())
    overfit_loader = DataLoader(
        overfit_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
    )
    model_d = _make_model(input_dim, args).to(device)
    res_d = _train_mini(
        model_d, train_loader, val_loader, device,
        epochs=args.overfit_epochs, lr=args.lr,
        s2f_class_weight=None, p2f_class_weight=None,
        p2f_loss_weight=1.0, grad_clip=args.grad_clip,
        subset_loader=overfit_loader,
    )
    tr_end = res_d["epoch_history"][-1]["train_loss_uw"] if res_d["epoch_history"] else 999.0
    res_d["pass"] = bool(tr_end < 0.5)
    results["experiments"]["D_mini_overfit"] = res_d
    print(f"  final_train_uw={tr_end:.4f}  pass={res_d['pass']}")

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {report_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="EHREncoderTransformer training diagnostics")
    p.add_argument("--anchor_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--history_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--enriched_csv", default=ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true")
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--num_transformer_layers", type=int, default=NUM_TRANSFORMER_LAYERS)
    p.add_argument("--num_heads", type=int, default=NUM_HEADS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--anchor_pool", type=str, default=ANCHOR_POOL)
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--overfit_samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--epochs_no_cw", type=int, default=3)
    p.add_argument("--overfit_epochs", type=int, default=5)
    p.add_argument("--ce_compare_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--p2f_loss_weight", type=float, default=10.0)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--include_anchor_row", action=argparse.BooleanOptionalAction, default=INCLUDE_ANCHOR_ROW)
    p.add_argument("--output_dir", default=str(EXP_DIR / "output_diagnose"))
    run_experiments(p.parse_args())


if __name__ == "__main__":
    main()
