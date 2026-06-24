"""End-to-end train: frozen ViT + causal transformer + dual s2f/p2f change MLP heads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

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

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from common import collate_cxr_window_batch, stratify_labels_from_anchor  # noqa: E402
from config import *  # noqa: F401,F403,E402
from cxr_labeled_dataset import CXRLabeledCatalogDataset, CXRLabeledCatalogView  # noqa: E402
from eval_reports import evaluate_split_both_heads  # noqa: E402
from model import CXREncoderTransformer  # noqa: E402


def masked_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if not valid.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(
        logits[valid],
        y[valid],
        weight=class_weight,
        label_smoothing=label_smoothing,
    )


def _inverse_freq_weights(
    counts: np.ndarray,
    num_classes: int,
    device: torch.device,
    *,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> torch.Tensor:
    c = np.bincount(counts, minlength=num_classes).astype(np.float64)
    c = np.maximum(c, 1.0)
    w = 1.0 / c
    w = w * (num_classes / w.sum())
    w = np.clip(w, clip_min, clip_max)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32, device=device)


def _head_class_weights(
    ds,
    indices: np.ndarray,
    has_attr: str,
    cls_attr: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    has = getattr(ds, has_attr)
    cls = getattr(ds, cls_attr)
    labels = []
    for i in indices:
        if has[i] and cls[i] >= 0:
            labels.append(int(cls[i]))
    if not labels:
        return torch.ones(num_classes, device=device)
    return _inverse_freq_weights(np.asarray(labels, dtype=np.int64), num_classes, device)


def forward_loss_parts(
    batch: dict,
    log_s: torch.Tensor,
    log_p: torch.Tensor,
    *,
    p2f_loss_weight: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    device = log_s.device
    s_tgt = batch["anchor_s2f"].to(device)
    p_tgt = batch["anchor_p2f"].to(device)
    s_ok = batch["anchor_has_s2f"].to(device) & (s_tgt >= 0)
    p_ok = batch["anchor_has_p2f"].to(device) & (p_tgt >= 0)
    loss_s = masked_ce(log_s, s_tgt, s_ok, s2f_class_weight, label_smoothing=label_smoothing)
    loss_p = masked_ce(log_p, p_tgt, p_ok, p2f_class_weight, label_smoothing=label_smoothing)
    n_s = int(s_ok.sum())
    n_p = int(p_ok.sum())
    if n_s and n_p:
        total = (loss_s * n_s + loss_p * p2f_loss_weight * n_p) / (n_s + p2f_loss_weight * n_p)
    elif n_s:
        total = loss_s
    elif n_p:
        total = loss_p
    else:
        total = loss_s + loss_p
    return total, {"loss_s2f": loss_s, "loss_p2f": loss_p, "log_s2f": log_s, "log_p2f": log_p}


def forward_loss_from_logits_parts(
    batch: dict,
    log_s: torch.Tensor,
    log_p: torch.Tensor,
    *,
    p2f_loss_weight: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
) -> tuple:
    return forward_loss_parts(
        batch,
        log_s,
        log_p,
        p2f_loss_weight=p2f_loss_weight,
        s2f_class_weight=s2f_class_weight,
        p2f_class_weight=p2f_class_weight,
    )


def _count_params(model: CXREncoderTransformer) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _grad_norm(model: CXREncoderTransformer) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        sq += float(p.grad.detach().pow(2).sum())
    return sq**0.5


def _param_l2(model: CXREncoderTransformer) -> float:
    sq = sum(float(p.detach().pow(2).sum()) for p in model.parameters() if p.requires_grad)
    return sq**0.5


def _snapshot_trainable(model: CXREncoderTransformer) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _param_delta_l2(model: CXREncoderTransformer, before: dict) -> float:
    sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in before:
            continue
        sq += float((p.detach() - before[n]).pow(2).sum())
    return sq**0.5


def _restore_trainable(model: CXREncoderTransformer, snap: dict) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in snap:
                p.copy_(snap[n])


def _trainable_finite(model: CXREncoderTransformer) -> bool:
    for p in model.parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            return False
    return True


def _print_module_trainable(model: CXREncoderTransformer) -> None:
    print("  Trainable modules (params with requires_grad=True):")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        t = sum(p.numel() for p in mod.parameters())
        print(f"    {name}: trainable={n:,} / total={t:,}")


def _anchor_label_stats(ds, indices: np.ndarray, split_name: str) -> None:
    s_cnt = np.zeros(3, dtype=np.int64)
    p_cnt = np.zeros(3, dtype=np.int64)
    n_s = n_p = 0
    for i in indices:
        if ds.anchor_has_s2f[i] and ds.anchor_s2f_cls[i] >= 0:
            s_cnt[int(ds.anchor_s2f_cls[i])] += 1
            n_s += 1
        if ds.anchor_has_p2f[i] and ds.anchor_p2f_cls[i] >= 0:
            p_cnt[int(ds.anchor_p2f_cls[i])] += 1
            n_p += 1
    print(
        f"  [{split_name}] anchor labels  s2f n={n_s:,} counts={s_cnt.tolist()}  "
        f"majority_acc={float(s_cnt.max()) / max(n_s, 1):.6f}"
    )
    print(
        f"  [{split_name}] anchor labels  p2f n={n_p:,} counts={p_cnt.tolist()}  "
        f"majority_acc={float(p_cnt.max()) / max(n_p, 1):.6f}"
    )


def _hist_str(counts: np.ndarray) -> str:
    tot = max(int(counts.sum()), 1)
    pct = [100.0 * int(c) / tot for c in counts]
    return f"counts={counts.tolist()} pct=[{', '.join(f'{x:.1f}' for x in pct)}]%"


def _n_unique_preds(counts: np.ndarray) -> int:
    return int(np.count_nonzero(counts))


@torch.no_grad()
def eval_loader(
    model,
    loader,
    device,
    collect_pred_hist: bool = False,
    *,
    p2f_loss_weight: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
) -> dict:
    model.eval()
    tot = 0.0
    n_batches = 0
    n_skipped_nonfinite = 0
    acc_s_n = acc_s_d = acc_p_n = acc_p_d = 0.0
    ce_s_sum = ce_p_sum = 0.0
    n_ce_s = n_ce_p = 0
    pred_s = np.zeros(3, dtype=np.int64)
    pred_p = np.zeros(3, dtype=np.int64)
    for batch in loader:
        if batch is None:
            continue
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["cxr_seq"], b["cxr_mask"])
        if not (torch.isfinite(log_s).all() and torch.isfinite(log_p).all()):
            n_skipped_nonfinite += 1
            continue
        loss, _ = forward_loss_parts(
            b,
            log_s,
            log_p,
            p2f_loss_weight=p2f_loss_weight,
            s2f_class_weight=s2f_class_weight,
            p2f_class_weight=p2f_class_weight,
        )
        if not torch.isfinite(loss):
            n_skipped_nonfinite += 1
            continue
        tot += float(loss)
        n_batches += 1
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        if s_ok.any():
            ce_s_sum += float(F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok], weight=s2f_class_weight))
            n_ce_s += 1
            pred = log_s[s_ok].argmax(1)
            acc_s_n += (pred == b["anchor_s2f"][s_ok]).float().sum().item()
            acc_s_d += int(s_ok.sum())
            if collect_pred_hist:
                for c in pred.cpu().numpy():
                    pred_s[int(c)] += 1
        if p_ok.any():
            ce_p_sum += float(F.cross_entropy(log_p[p_ok], b["anchor_p2f"][p_ok], weight=p2f_class_weight))
            n_ce_p += 1
            pred = log_p[p_ok].argmax(1)
            acc_p_n += (pred == b["anchor_p2f"][p_ok]).float().sum().item()
            acc_p_d += int(p_ok.sum())
            if collect_pred_hist:
                for c in pred.cpu().numpy():
                    pred_p[int(c)] += 1
    out = {
        "loss": tot / max(n_batches, 1),
        "ce_s2f": ce_s_sum / max(n_ce_s, 1),
        "ce_p2f": ce_p_sum / max(n_ce_p, 1),
        "acc_s2f": acc_s_n / max(acc_s_d, 1),
        "acc_p2f": acc_p_n / max(acc_p_d, 1),
        "n_skipped_nonfinite": n_skipped_nonfinite,
    }
    if collect_pred_hist:
        out["pred_hist_s2f"] = pred_s
        out["pred_hist_p2f"] = pred_p
        out["n_unique_pred_s2f"] = _n_unique_preds(pred_s)
        out["n_unique_pred_p2f"] = _n_unique_preds(pred_p)
    return out


def _make_split_datasets(full_ds, idx_train, idx_val, idx_test):
    """Train uses RandomCrop; val/test use CenterCrop (cxr_split val/test)."""
    if isinstance(full_ds, Subset):
        base = full_ds.dataset
        map_idx = np.asarray(full_ds.indices, dtype=np.int64)
        idx_train = map_idx[idx_train]
        idx_val = map_idx[idx_val]
        idx_test = map_idx[idx_test]
    else:
        base = full_ds
    if isinstance(base, CXRLabeledCatalogDataset):
        return (
            CXRLabeledCatalogView(base, idx_train, "train"),
            CXRLabeledCatalogView(base, idx_val, "val"),
            CXRLabeledCatalogView(base, idx_test, "test"),
            base,
        )
    train_ds = make_subset(full_ds, idx_train) if not isinstance(full_ds, Subset) else make_subset(base, idx_train)
    val_ds = make_subset(full_ds, idx_val) if not isinstance(full_ds, Subset) else make_subset(base, idx_val)
    test_ds = make_subset(full_ds, idx_test) if not isinstance(full_ds, Subset) else make_subset(base, idx_test)
    return train_ds, val_ds, test_ds, base


def _build_dataset(args):
    if args.use_runtime_catalog:
        from cxr_catalog_dataset import CXRCatalogWindowDataset  # noqa: E402

        enr = None if args.no_enriched else args.enriched_csv
        if enr and (not str(enr).strip() or not Path(enr).is_file()):
            print(f"  enriched_csv missing; using catalog hadm map: {enr!r}")
            enr = None
        print(f"  Dataset: CXRCatalogWindowDataset (runtime window)  catalog={args.cxr_catalog_csv}")
        return CXRCatalogWindowDataset(
            anchor_source_csv=args.anchor_csv,
            cxr_catalog_csv=args.cxr_catalog_csv,
            label_lookup_csv=args.label_lookup_csv,
            enriched_csv=enr,
            cxr_root=args.cxr_root,
            metadata_path=args.metadata_path,
            lookback_min_hours=args.lookback_min_hours,
            lookback_max_hours=args.lookback_max_hours,
            cxr_split=args.cxr_split,
            imagenet_normalize=not args.no_imagenet_normalize,
        )

    print(f"  Dataset: CXRLabeledCatalogDataset  csv={args.cxr_labeled_csv}")
    return CXRLabeledCatalogDataset(
        labeled_csv=args.cxr_labeled_csv,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        cxr_split=args.cxr_split,
        imagenet_normalize=not args.no_imagenet_normalize,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        require_hours_in_window=not args.no_hours_filter,
    )


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(
            "WARNING: CUDA not available; training on CPU. "
            "Use Slurm partition gpu-common (-p gpu-common) and request a GPU (-G 1)."
        )
    print(f"CXREncoderTransformer  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ViT path: {args.vit_path}  freeze_cxr={not args.unfreeze_cxr}")
    print(
        f"  Loss: p2f_weight={args.p2f_loss_weight}  class_weights={args.use_class_weights}  "
        f"include_anchor_slot={args.include_anchor_slot}"
    )

    if not args.use_runtime_catalog and not Path(args.cxr_labeled_csv).is_file():
        raise FileNotFoundError(f"Labeled CXR CSV not found: {args.cxr_labeled_csv}")

    full_ds = _build_dataset(args)
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        idxs = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, idxs.tolist())
        print(f"  Subset max_samples={args.max_samples}")

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
        idx_val_base = _map[idx_val]
        idx_test_base = _map[idx_test]
    else:
        idx_train_base = idx_train
        idx_val_base = idx_val
        idx_test_base = idx_test
    train_ds, val_ds, test_ds, base = _make_split_datasets(full_ds, idx_train, idx_val, idx_test)
    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    if isinstance(base, CXRLabeledCatalogDataset):
        print("  cxr_split: train=RandomCrop  val/test=CenterCrop")

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_window_batch,
    )

    model = CXREncoderTransformer(
        cxr_dim=args.cxr_dim,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        anchor_pool=args.anchor_pool,
        vit_path=args.vit_path,
        freeze_cxr=not args.unfreeze_cxr,
        include_anchor_slot=args.include_anchor_slot,
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_train, n_total = _count_params(model)
    print(f"  Parameters: trainable={n_train:,}  total={n_total:,}  lr={args.lr}  weight_decay={args.weight_decay}")
    if args.max_grad_norm and args.max_grad_norm > 0:
        print(f"  max_grad_norm={args.max_grad_norm}")
    _print_module_trainable(model)
    _anchor_label_stats(base, idx_val_base, "val")
    _anchor_label_stats(base, idx_train_base, "train")

    s2f_w = p2f_w = None
    if args.use_class_weights:
        s2f_w = _head_class_weights(
            base, idx_train_base, "anchor_has_s2f", "anchor_s2f_cls", args.num_classes, device
        )
        p2f_w = _head_class_weights(
            base, idx_train_base, "anchor_has_p2f", "anchor_p2f_cls", args.num_classes, device
        )
        print(f"  s2f class weights: {s2f_w.detach().cpu().tolist()}")
        print(f"  p2f class weights: {p2f_w.detach().cpu().tolist()}")

    loss_kw = dict(
        p2f_loss_weight=args.p2f_loss_weight,
        s2f_class_weight=s2f_w,
        p2f_class_weight=p2f_w,
        label_smoothing=args.label_smoothing,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False
    param_snap_start = _snapshot_trainable(model)
    w_before_step = None
    logged_nonfinite_loss = False
    logged_nonfinite_grad = False

    for epoch in range(args.epochs):
        model.train()
        epoch_snap = _snapshot_trainable(model)
        tr = tr_s = tr_p = 0.0
        n_tr_batches = 0
        n_skipped_nonfinite_loss = 0
        n_skipped_nonfinite_grad = 0
        n_rollback = 0
        last_grad_norm = 0.0
        epoch_aborted_nan = False
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            step_snap = _snapshot_trainable(model)
            log_s, log_p = model(b["cxr_seq"], b["cxr_mask"])
            loss, parts = forward_loss_parts(b, log_s, log_p, **loss_kw)
            if not torch.isfinite(loss):
                n_skipped_nonfinite_loss += 1
                _restore_trainable(model, step_snap)
                opt.zero_grad()
                if not logged_nonfinite_loss:
                    logged_nonfinite_loss = True
                    print(
                        f"  WARNING: non-finite loss at epoch {epoch + 1} batch {batch_idx} "
                        f"(rollback step). loss={float(loss)}"
                    )
                if not _trainable_finite(model):
                    epoch_aborted_nan = True
                    print(f"  ERROR: non-finite trainable weights after rollback at epoch {epoch + 1}")
                    break
                continue
            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            last_grad_norm = _grad_norm(model)
            if not np.isfinite(last_grad_norm):
                n_skipped_nonfinite_grad += 1
                _restore_trainable(model, step_snap)
                opt.zero_grad()
                if not logged_nonfinite_grad:
                    logged_nonfinite_grad = True
                    print(
                        f"  WARNING: non-finite grad at epoch {epoch + 1} batch {batch_idx} "
                        f"(rollback step). grad_norm={last_grad_norm}  loss={float(loss):.4f}"
                    )
                if not _trainable_finite(model):
                    epoch_aborted_nan = True
                    print(f"  ERROR: non-finite trainable weights after grad rollback at epoch {epoch + 1}")
                    break
                continue
            if epoch == 0 and batch_idx == 0:
                w_before_step = _snapshot_trainable(model)
            opt.step()
            if not _trainable_finite(model):
                _restore_trainable(model, step_snap)
                opt.zero_grad()
                n_rollback += 1
                epoch_aborted_nan = True
                print(
                    f"  ERROR: non-finite weights after opt.step at epoch {epoch + 1} batch {batch_idx} "
                    f"(rolled back)"
                )
                break
            if epoch == 0 and batch_idx == 0:
                step_delta = _param_delta_l2(model, w_before_step)
                lens = b["cxr_mask"].long().sum(dim=1)
                valid_cxr = int(b["cxr_mask"].sum())
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
                with torch.no_grad():
                    _, _, anchor_vec = model(
                        b["cxr_seq"], b["cxr_mask"], return_anchor_vec=True
                    )
                    anchor_std = float(anchor_vec.std(dim=0).mean())
                    log_std_s = float(parts["log_s2f"][s_ok].std(dim=0).mean()) if s_ok.any() else 0.0
                print(
                    "  [train check epoch1 batch0] "
                    f"loss={float(loss):.4f}  loss_s2f={float(parts['loss_s2f']):.4f}  "
                    f"loss_p2f={float(parts['loss_p2f']):.4f}  grad_norm={last_grad_norm:.6f}  "
                    f"param_delta_after_1step={step_delta:.8f}"
                )
                print(
                    f"    batch: cxr_seq={tuple(b['cxr_seq'].shape)}  "
                    f"valid_cxr_slots={valid_cxr}/{b['cxr_mask'].numel()}  "
                    f"seq_len min/median/max={int(lens.min())}/{int(lens.median())}/{int(lens.max())}  "
                    f"n_s2f={int(s_ok.sum())}  n_p2f={int(p_ok.sum())}  "
                    f"anchor_vec_std={anchor_std:.6f}  log_s2f_std={log_std_s:.6f}"
                )
                if s_ok.any():
                    ps = parts["log_s2f"][s_ok].argmax(1).cpu().numpy()
                    print(f"    batch pred_s2f (valid only): {_hist_str(np.bincount(ps, minlength=3))}")
                if p_ok.any():
                    pp = parts["log_p2f"][p_ok].argmax(1).cpu().numpy()
                    print(f"    batch pred_p2f (valid only): {_hist_str(np.bincount(pp, minlength=3))}")
            tr += float(loss)
            tr_s += float(parts["loss_s2f"])
            tr_p += float(parts["loss_p2f"])
            n_tr_batches += 1
        if n_skipped_nonfinite_loss or n_skipped_nonfinite_grad or n_rollback:
            print(
                f"  train skipped: non-finite loss={n_skipped_nonfinite_loss}  "
                f"non-finite grad={n_skipped_nonfinite_grad}  rollbacks={n_rollback}"
            )
        if epoch_aborted_nan:
            _restore_trainable(model, epoch_snap)
            print(f"  Restored trainable weights to start-of-epoch snapshot (epoch {epoch + 1})")
            break
        tr /= max(n_tr_batches, 1)
        tr_s /= max(n_tr_batches, 1)
        tr_p /= max(n_tr_batches, 1)
        epoch_param_delta = _param_delta_l2(model, param_snap_start) if epoch == 0 else None
        st = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={tr:.4f}  "
            f"(s2f={tr_s:.4f} p2f={tr_p:.4f})  val_loss={st['loss']:.4f}  "
            f"val_acc_s2f={st['acc_s2f']:.6f}  val_acc_p2f={st['acc_p2f']:.6f}"
        )
        print(
            f"  val pred_s2f: {_hist_str(st['pred_hist_s2f'])}  "
            f"val pred_p2f: {_hist_str(st['pred_hist_p2f'])}  "
            f"unique_classes(s2f/p2f)={st.get('n_unique_pred_s2f', '?')}/{st.get('n_unique_pred_p2f', '?')}"
        )
        if st.get("n_skipped_nonfinite", 0):
            print(f"  val skipped non-finite batches: {st['n_skipped_nonfinite']}")
        print(
            f"  train diagnostics: last_batch_grad_norm={last_grad_norm:.6f}  "
            f"param_l2={_param_l2(model):.4f}"
            + (
                f"  epoch1_param_delta_vs_init={epoch_param_delta:.8f}"
                if epoch_param_delta is not None
                else ""
            )
        )
        improved = np.isfinite(st["loss"]) and st["loss"] < best_val - args.early_stop_min_delta
        if improved:
            best_val = st["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val},
                out_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}/{args.epochs} "
                f"(best epoch {best_epoch + 1}, best val_loss={best_val:.4f})"
            )
            stopped_early = True
            break

    torch.save(model.state_dict(), out_dir / "last.pt")
    has_best = (out_dir / "best.pt").is_file()
    if has_best:
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"\nEvaluating with best checkpoint (epoch {best_epoch + 1})")
    elif not _trainable_finite(model):
        print("\nWARNING: no valid best.pt and last.pt has non-finite weights; test metrics unreliable")
    test_st = eval_loader(model, test_loader, device, **loss_kw)
    print(
        f"\nTest (summary): loss={test_st['loss']:.4f}  "
        f"acc_s2f={test_st['acc_s2f']:.4f}  acc_p2f={test_st['acc_p2f']:.4f}"
    )
    print(f"  (best checkpoint epoch {best_epoch + 1 if best_epoch >= 0 else 'n/a'})")

    val_reports = evaluate_split_both_heads(
        model, val_loader, device, "Val (best checkpoint)", args.num_classes
    )
    test_reports = evaluate_split_both_heads(
        model, test_loader, device, "Test (best checkpoint)", args.num_classes
    )

    results = {
        "task": "cxr_encoder_transformer_labeled_catalog",
        "cxr_labeled_csv": args.cxr_labeled_csv,
        "use_runtime_catalog": args.use_runtime_catalog,
        "lookback_hours": [args.lookback_max_hours, args.lookback_min_hours],
        "include_anchor_slot": args.include_anchor_slot,
        "p2f_loss_weight": args.p2f_loss_weight,
        "use_class_weights": args.use_class_weights,
        "label_smoothing": args.label_smoothing,
        "max_grad_norm": args.max_grad_norm,
        "anchor_pool": args.anchor_pool,
        "freeze_cxr": not args.unfreeze_cxr,
        "best_val_loss": best_val,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "stopped_early": stopped_early,
        "test_summary": test_st,
        "val": val_reports,
        "test": test_reports,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report_test.json", "w") as f:
        json.dump(test_reports, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CXREncoderTransformer: frozen ViT + causal transformer + MLP heads")
    p.add_argument(
        "--cxr_labeled_csv",
        default=CXR_CATALOG_LABELED_CSV,
        help="Pre-joined CXR–anchor labeled catalog (default data source)",
    )
    p.add_argument(
        "--use_runtime_catalog",
        action="store_true",
        help="Use CXRCatalogWindowDataset + raw catalog instead of labeled CSV",
    )
    p.add_argument("--anchor_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--label_lookup_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--cxr_catalog_csv", default=CXR_CATALOG_CSV)
    p.add_argument("--enriched_csv", default=ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true")
    p.add_argument("--no_hours_filter", action="store_true", help="Keep all labeled pairs regardless of hours")
    p.add_argument("--history_csv", default=ENRICHED_CSV)
    p.add_argument("--cxr_root", default=CXR_ROOT)
    p.add_argument("--metadata_path", default=METADATA_PATH)
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--cxr_split", default=CXR_SPLIT)
    p.add_argument("--no_imagenet_normalize", action="store_true")
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--cxr_dim", type=int, default=CXR_DIM)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--num_transformer_layers", type=int, default=NUM_TRANSFORMER_LAYERS)
    p.add_argument("--num_heads", type=int, default=NUM_HEADS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--anchor_pool", type=str, default=ANCHOR_POOL, choices=["last", "mean"])
    p.add_argument("--vit_path", default=VIT_PATH)
    p.add_argument("--unfreeze_cxr", action="store_true")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    p.add_argument("--include_anchor_slot", action=argparse.BooleanOptionalAction, default=INCLUDE_ANCHOR_SLOT)
    p.add_argument("--p2f_loss_weight", type=float, default=P2F_LOSS_WEIGHT)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=USE_CLASS_WEIGHTS)
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM, help="Max grad norm (0=disable)")
    p.add_argument("--grad_clip", type=float, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.grad_clip is not None:
        a.max_grad_norm = a.grad_clip
    if not a.max_samples:
        a.max_samples = 0
    main(a)
