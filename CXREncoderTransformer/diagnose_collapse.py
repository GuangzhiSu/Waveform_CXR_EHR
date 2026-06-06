#!/usr/bin/env python3
"""Diagnose CXREncoderTransformer majority-class collapse (5 quick experiments)."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
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
from config import *  # noqa: F401,F403,E402
from cxr_labeled_dataset import CXRLabeledCatalogDataset, CXRLabeledCatalogView  # noqa: E402
from model import CXREncoderTransformer  # noqa: E402
from models.encoders.cxr import CXREncoder  # noqa: E402
from train import (  # noqa: E402
    _head_class_weights,
    _hist_str,
    _make_split_datasets,
    eval_loader,
    forward_loss_parts,
)


class CXRLinearProbe(nn.Module):
    """Frozen ViT + trainable proj + masked mean pool + linear s2f head (diagnostic only)."""

    def __init__(self, vit_path: str, cxr_dim: int = 512, num_classes: int = 3):
        super().__init__()
        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=True)
        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.head = nn.Linear(cxr_dim, num_classes)

    def forward(self, cxr_seq: torch.Tensor, cxr_mask: torch.Tensor) -> torch.Tensor:
        bsz, t, _, _, _ = cxr_seq.shape
        flat = cxr_seq.reshape(bsz * t, 3, 224, 224)
        flat_mask = cxr_mask.reshape(-1)
        z = self.miss_cxr.view(1, 1, -1).expand(bsz, t, -1).clone()
        if flat_mask.any():
            enc = self.cxr_enc(flat[flat_mask])
            enc = torch.nan_to_num(enc, nan=0.0, posinf=0.0, neginf=0.0)
            z.reshape(-1, z.size(-1))[flat_mask] = enc
        w = cxr_mask.float().unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)
        pooled = (z * w).sum(dim=1) / denom
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


def _pred_hist(logits: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    if not valid.any():
        return np.zeros(3, dtype=np.int64)
    pred = logits[valid].argmax(1).cpu().numpy()
    return np.bincount(pred, minlength=3)


def _build_loaders(args, device: torch.device):
    full_ds = CXRLabeledCatalogDataset(
        labeled_csv=args.cxr_labeled_csv,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        cxr_split="train",
        imagenet_normalize=True,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
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

    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    if isinstance(full_ds, Subset):
        _map = np.asarray(full_ds.indices, dtype=np.int64)
        idx_train_base = _map[idx_train]
    else:
        idx_train_base = idx_train
    train_ds, val_ds, _, _ = _make_split_datasets(full_ds, idx_train, idx_val, idx_test)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_window_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_window_batch,
    )
    s2f_w = _head_class_weights(base, idx_train_base, "anchor_has_s2f", "anchor_s2f_cls", 3, device)
    return train_loader, val_loader, base, idx_train_base, s2f_w


def _train_mini(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    batch_transform: Optional[Callable[[dict], dict]] = None,
    loss_fn: Optional[Callable] = None,
    grad_clip: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
) -> dict:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    loss_kw = dict(p2f_loss_weight=10.0, s2f_class_weight=s2f_class_weight, p2f_class_weight=None)
    epoch_hist = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if batch_transform is not None:
                b = batch_transform(b)
            if isinstance(model, CXRLinearProbe):
                log_s = model(b["cxr_seq"], b["cxr_mask"])
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                loss = F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok], weight=s2f_class_weight)
            else:
                log_s, log_p = model(b["cxr_seq"], b["cxr_mask"])
                loss, _ = forward_loss_parts(b, log_s, log_p, **loss_kw)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()

        if isinstance(model, CXRLinearProbe):
            model.eval()
            ce_sum = acc_n = acc_d = 0.0
            pred_hist = np.zeros(3, dtype=np.int64)
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    if batch_transform is not None:
                        b = batch_transform(b)
                    log_s = model(b["cxr_seq"], b["cxr_mask"])
                    s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                    if s_ok.any():
                        ce_sum += float(F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok], weight=s2f_class_weight))
                        pred = log_s[s_ok].argmax(1)
                        acc_n += (pred == b["anchor_s2f"][s_ok]).float().sum().item()
                        acc_d += int(s_ok.sum())
                        for c in pred.cpu().numpy():
                            pred_hist[int(c)] += 1
            st = {
                "loss": ce_sum / max(len(val_loader), 1),
                "acc_s2f": acc_n / max(acc_d, 1),
                "pred_hist_s2f": pred_hist,
            }
        else:
            st = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)
        epoch_hist.append(
            {
                "epoch": epoch + 1,
                "val_acc_s2f": st["acc_s2f"],
                "val_loss": st["loss"],
                "pred_hist_s2f": st["pred_hist_s2f"].tolist(),
            }
        )

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if batch_transform is not None:
                b = batch_transform(b)
            if isinstance(model, CXRLinearProbe):
                log_s = model(b["cxr_seq"], b["cxr_mask"])
            else:
                log_s, _ = model(b["cxr_seq"], b["cxr_mask"])
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            if s_ok.any():
                y_true.extend(b["anchor_s2f"][s_ok].cpu().numpy().tolist())
                y_pred.extend(log_s[s_ok].argmax(1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    maj = _majority_baseline(y_true)
    n_unique = len(np.unique(y_pred)) if y_pred.size else 0
    collapsed = n_unique == 1
    return {
        "val_acc_s2f": float((y_pred == y_true).mean()) if y_true.size else 0.0,
        "majority_baseline": maj,
        "macro_f1_s2f": _macro_f1(y_true, y_pred),
        "pred_hist_s2f": np.bincount(y_pred, minlength=3).tolist() if y_pred.size else [0, 0, 0],
        "n_unique_predictions": n_unique,
        "collapsed_to_one_class": collapsed,
        "epoch_history": epoch_hist,
    }


def _shuffle_labels_batch(b: dict) -> dict:
    out = deepcopy(b)
    s_ok = out["anchor_has_s2f"] & (out["anchor_s2f"] >= 0)
    if s_ok.any():
        n = int(s_ok.sum())
        perm = torch.randperm(n, device=out["anchor_s2f"].device)
        out["anchor_s2f"] = out["anchor_s2f"].clone()
        out["anchor_s2f"][s_ok] = out["anchor_s2f"][s_ok][perm]
    p_ok = out["anchor_has_p2f"] & (out["anchor_p2f"] >= 0)
    if p_ok.any():
        n = int(p_ok.sum())
        perm = torch.randperm(n, device=out["anchor_p2f"].device)
        out["anchor_p2f"] = out["anchor_p2f"].clone()
        out["anchor_p2f"][p_ok] = out["anchor_p2f"][p_ok][perm]
    return out


def _shuffle_images_batch(b: dict) -> dict:
    out = deepcopy(b)
    bsz = out["cxr_seq"].size(0)
    perm = torch.randperm(bsz, device=out["cxr_seq"].device)
    out["cxr_seq"] = out["cxr_seq"][perm]
    out["cxr_mask"] = out["cxr_mask"][perm]
    return out


@torch.no_grad()
def _anchor_vec_stats(model: CXREncoderTransformer, val_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    vecs = []
    for batch in val_loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _, _, av = model(b["cxr_seq"], b["cxr_mask"], return_anchor_vec=True)
        vecs.append(av.cpu())
    all_v = torch.cat(vecs, dim=0)
    per_dim_std = all_v.std(dim=0)
    return {
        "n_samples": int(all_v.size(0)),
        "anchor_vec_std_mean": float(per_dim_std.mean()),
        "anchor_vec_std_min": float(per_dim_std.min()),
        "anchor_vec_norm_mean": float(all_v.norm(dim=1).mean()),
    }


def run_experiments(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"diagnose_collapse  device={device}  max_samples={args.max_samples}  epochs={args.epochs}")
    train_loader, val_loader, base, idx_train, s2f_w = _build_loaders(args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"device": str(device), "max_samples": args.max_samples, "epochs": args.epochs, "experiments": {}}

    # E — full model baseline
    print("\n=== E: full model baseline ===")
    model_e = CXREncoderTransformer(vit_path=args.vit_path, freeze_cxr=True).to(device)
    res_e = _train_mini(
        model_e, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, s2f_class_weight=s2f_w, grad_clip=args.grad_clip,
    )
    res_e["pass_note"] = "baseline collapse reproduced" if res_e["collapsed_to_one_class"] else "no single-class collapse"
    results["experiments"]["E_full_baseline"] = res_e
    print(f"  acc={res_e['val_acc_s2f']:.4f}  maj={res_e['majority_baseline']:.4f}  macro_f1={res_e['macro_f1_s2f']:.4f}")
    print(f"  pred_s2f: {_hist_str(np.array(res_e['pred_hist_s2f']))}")

    # A — label shuffle
    print("\n=== A: label shuffle ===")
    model_a = CXREncoderTransformer(vit_path=args.vit_path, freeze_cxr=True).to(device)
    res_a = _train_mini(
        model_a, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, batch_transform=_shuffle_labels_batch,
        s2f_class_weight=s2f_w, grad_clip=args.grad_clip,
    )
    res_a["pass"] = res_a["macro_f1_s2f"] < 0.35 and abs(res_a["val_acc_s2f"] - res_a["majority_baseline"]) < 0.05
    results["experiments"]["A_label_shuffle"] = res_a
    print(f"  acc={res_a['val_acc_s2f']:.4f}  macro_f1={res_a['macro_f1_s2f']:.4f}  pass={res_a['pass']}")

    # B — image shuffle (train on real, eval on shuffled images)
    print("\n=== B: image shuffle at eval ===")
    model_b = CXREncoderTransformer(vit_path=args.vit_path, freeze_cxr=True).to(device)
    res_b_real = _train_mini(
        model_b, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, s2f_class_weight=s2f_w, grad_clip=args.grad_clip,
    )
    # Re-eval trained model with shuffled vs real images
    model_b.eval()
    y_true, y_pred_shuf, y_pred_real = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, _ = model_b(b["cxr_seq"], b["cxr_mask"])
            b_shuf = _shuffle_images_batch(b)
            log_s_shuf, _ = model_b(b_shuf["cxr_seq"], b_shuf["cxr_mask"])
            s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            if s_ok.any():
                y_true.extend(b["anchor_s2f"][s_ok].cpu().numpy().tolist())
                y_pred_real.extend(log_s[s_ok].argmax(1).cpu().numpy().tolist())
                y_pred_shuf.extend(log_s_shuf[s_ok].argmax(1).cpu().numpy().tolist())
    y_true = np.asarray(y_true, dtype=np.int64)
    acc_real = float((np.asarray(y_pred_real) == y_true).mean()) if y_true.size else 0.0
    acc_shuf = float((np.asarray(y_pred_shuf) == y_true).mean()) if y_true.size else 0.0
    res_b = {
        "val_acc_s2f_real_images": acc_real,
        "val_acc_s2f_shuffled_images": acc_shuf,
        "acc_drop": acc_real - acc_shuf,
        "pass": (acc_real - acc_shuf) >= 0.05,
        "train_summary": res_b_real,
    }
    results["experiments"]["B_image_shuffle"] = res_b
    print(f"  acc real={acc_real:.4f}  shuffled={acc_shuf:.4f}  drop={acc_real - acc_shuf:.4f}  pass={res_b['pass']}")

    # C — anchor_vec variance (untrained vs after brief train)
    print("\n=== C: anchor_vec variance ===")
    model_c0 = CXREncoderTransformer(vit_path=args.vit_path, freeze_cxr=True).to(device)
    stats_init = _anchor_vec_stats(model_c0, val_loader, device)
    _train_mini(
        model_c0, train_loader, val_loader, device,
        epochs=max(1, args.epochs // 2), lr=args.lr, s2f_class_weight=s2f_w, grad_clip=args.grad_clip,
    )
    stats_trained = _anchor_vec_stats(model_c0, val_loader, device)
    res_c = {
        "init": stats_init,
        "after_train": stats_trained,
        "pass": stats_init["anchor_vec_std_mean"] > 1e-4,
    }
    results["experiments"]["C_anchor_vec_variance"] = res_c
    print(f"  init std_mean={stats_init['anchor_vec_std_mean']:.6f}  after={stats_trained['anchor_vec_std_mean']:.6f}")

    # D — linear probe
    print("\n=== D: linear probe (frozen ViT + mean pool) ===")
    probe = CXRLinearProbe(vit_path=args.vit_path).to(device)
    res_d = _train_mini(
        probe, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, s2f_class_weight=s2f_w, grad_clip=args.grad_clip,
    )
    res_d["pass"] = res_d["val_acc_s2f"] > res_d["majority_baseline"] + 0.03
    results["experiments"]["D_linear_probe"] = res_d
    print(
        f"  acc={res_d['val_acc_s2f']:.4f}  maj={res_d['majority_baseline']:.4f}  "
        f"macro_f1={res_d['macro_f1_s2f']:.4f}  pass={res_d['pass']}"
    )
    print(f"  pred_s2f: {_hist_str(np.array(res_d['pred_hist_s2f']))}")

    summary_path = out_dir / "diagnose_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {summary_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="CXR collapse diagnostics")
    p.add_argument("--cxr_labeled_csv", default=CXR_CATALOG_LABELED_CSV)
    p.add_argument("--cxr_root", default=CXR_ROOT)
    p.add_argument("--metadata_path", default=METADATA_PATH)
    p.add_argument("--vit_path", default=VIT_PATH)
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--max_samples", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", default=str(Path(OUTPUT_DIR) / "diagnose"))
    args = p.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
