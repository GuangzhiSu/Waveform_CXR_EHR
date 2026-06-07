"""End-to-end train: EHRMLPEncoder + causal transformer + dual s2f/p2f change heads."""
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BaselineExperiment"))
sys.path.insert(0, str(PROJECT_ROOT / "EHRTrend"))
sys.path.insert(0, str(_EXP))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import *  # noqa: F401,F403,E402
from ehr_nextstep_dataset import EHRNextStepDataset  # noqa: E402
from ehr_symile_dataset import EHRNextStepDatasetSymile  # noqa: E402
from model import EHREncoderTransformer  # noqa: E402


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


def _class_weights_from_counts(
    counts: np.ndarray,
    num_classes: int,
    device: torch.device,
    mode: str,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    c = np.bincount(counts, minlength=num_classes).astype(np.float64)
    c = np.maximum(c, 1.0)
    if mode == "sqrt_inverse":
        w = 1.0 / np.sqrt(c)
    else:
        w = 1.0 / c
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32, device=device)


def _head_class_weights(
    ds: EHRNextStepDataset | EHRNextStepDatasetSymile,
    indices: np.ndarray,
    has_attr: str,
    cls_attr: str,
    num_classes: int,
    device: torch.device,
    mode: str = "inverse_freq",
) -> Optional[torch.Tensor]:
    has = getattr(ds, has_attr)
    cls = getattr(ds, cls_attr)
    labels = []
    for i in indices:
        if has[i] and cls[i] >= 0:
            labels.append(int(cls[i]))
    if not labels:
        return None if mode == "none" else torch.ones(num_classes, device=device)
    return _class_weights_from_counts(np.asarray(labels, dtype=np.int64), num_classes, device, mode)


def collate_anchor_batch(batch):
    lengths = [b["ehr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    feat = batch[0]["ehr_seq"].shape[1]
    bsz = len(batch)
    seq = torch.zeros(bsz, max_len, feat, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    anchor_s2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_p2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_has_s2f = torch.zeros(bsz, dtype=torch.bool)
    anchor_has_p2f = torch.zeros(bsz, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["ehr_seq"].shape[0]
        seq[i, :t] = b["ehr_seq"]
        mask[i, :t] = True
        c = b["anchor_s2f_cls"]
        anchor_s2f[i] = c if c >= 0 else -1
        c2 = b["anchor_p2f_cls"]
        anchor_p2f[i] = c2 if c2 >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])
    return {
        "ehr_seq": seq,
        "ehr_mask": mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
    }


def _stratify_labels_from_dataset(ds: EHRNextStepDataset | EHRNextStepDatasetSymile) -> np.ndarray:
    n = len(ds)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ds.anchor_has_p2f[i] and ds.anchor_p2f_cls[i] >= 0:
            y[i] = int(ds.anchor_p2f_cls[i])
        elif ds.anchor_has_s2f[i] and ds.anchor_s2f_cls[i] >= 0:
            y[i] = 3 + int(ds.anchor_s2f_cls[i])
        else:
            y[i] = 0
    return y


def forward_loss(
    model: EHREncoderTransformer,
    batch: dict,
    *,
    p2f_loss_weight: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total, _ = forward_loss_parts(
        model,
        batch,
        p2f_loss_weight=p2f_loss_weight,
        s2f_class_weight=s2f_class_weight,
        p2f_class_weight=p2f_class_weight,
    )
    return total


def forward_loss_parts(
    model: EHREncoderTransformer,
    batch: dict,
    *,
    p2f_loss_weight: float = 1.0,
    s2f_class_weight: Optional[torch.Tensor] = None,
    p2f_class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    log_s2f: Optional[torch.Tensor] = None,
    log_p2f: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    if log_s2f is None or log_p2f is None:
        log_s2f, log_p2f = model(batch["ehr_seq"], batch["ehr_mask"])
    device = log_s2f.device
    s_tgt = batch["anchor_s2f"].to(device)
    p_tgt = batch["anchor_p2f"].to(device)
    s_ok = batch["anchor_has_s2f"].to(device) & (s_tgt >= 0)
    p_ok = batch["anchor_has_p2f"].to(device) & (p_tgt >= 0)
    loss_s = masked_ce(log_s2f, s_tgt, s_ok, s2f_class_weight, label_smoothing=label_smoothing)
    loss_p = masked_ce(log_p2f, p_tgt, p_ok, p2f_class_weight, label_smoothing=label_smoothing)
    loss_s_uw = masked_ce(log_s2f, s_tgt, s_ok, None, label_smoothing=label_smoothing)
    loss_p_uw = masked_ce(log_p2f, p_tgt, p_ok, None, label_smoothing=label_smoothing)
    n_s = int(s_ok.sum())
    n_p = int(p_ok.sum())
    if n_s and n_p:
        total = (loss_s * n_s + loss_p * p2f_loss_weight * n_p) / (n_s + p2f_loss_weight * n_p)
        total_uw = (loss_s_uw * n_s + loss_p_uw * p2f_loss_weight * n_p) / (n_s + p2f_loss_weight * n_p)
    elif n_s:
        total = loss_s
        total_uw = loss_s_uw
    elif n_p:
        total = loss_p
        total_uw = loss_p_uw
    else:
        total = loss_s + loss_p
        total_uw = loss_s_uw + loss_p_uw
    return total, {
        "loss_s2f": loss_s,
        "loss_p2f": loss_p,
        "loss_s2f_uw": loss_s_uw,
        "loss_p2f_uw": loss_p_uw,
        "loss_uw": total_uw,
        "log_s2f": log_s2f,
        "log_p2f": log_p2f,
    }


def _count_params(model: EHREncoderTransformer) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _grad_norm(model: EHREncoderTransformer) -> float:
    sq = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float(g.pow(2).sum())
        n += g.numel()
    return (sq ** 0.5) if n else 0.0


def _param_l2(model: EHREncoderTransformer) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            sq += float(p.detach().pow(2).sum())
    return sq ** 0.5


def _snapshot_trainable(model: EHREncoderTransformer) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _param_delta_l2(model: EHREncoderTransformer, before: dict) -> float:
    sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in before:
            continue
        d = (p.detach() - before[n]).pow(2).sum()
        sq += float(d)
    return sq ** 0.5


def _print_module_trainable(model: EHREncoderTransformer) -> None:
    print("  Trainable modules (params with requires_grad=True):")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        t = sum(p.numel() for p in mod.parameters())
        print(f"    {name}: trainable={n:,} / total={t:,}")


def _anchor_label_stats(ds: EHRNextStepDataset, indices: np.ndarray, split_name: str) -> None:
    """Label counts + majority-class accuracy baseline on a split."""
    n_cls = 3
    s_cnt = np.zeros(n_cls, dtype=np.int64)
    p_cnt = np.zeros(n_cls, dtype=np.int64)
    n_s = n_p = 0
    for i in indices:
        if ds.anchor_has_s2f[i] and ds.anchor_s2f_cls[i] >= 0:
            s_cnt[int(ds.anchor_s2f_cls[i])] += 1
            n_s += 1
        if ds.anchor_has_p2f[i] and ds.anchor_p2f_cls[i] >= 0:
            p_cnt[int(ds.anchor_p2f_cls[i])] += 1
            n_p += 1
    maj_s = float(s_cnt.max()) / max(n_s, 1)
    maj_p = float(p_cnt.max()) / max(n_p, 1)
    print(f"  [{split_name}] anchor labels  s2f n={n_s:,} counts={s_cnt.tolist()}  majority_acc={maj_s:.6f}")
    print(f"  [{split_name}] anchor labels  p2f n={n_p:,} counts={p_cnt.tolist()}  majority_acc={maj_p:.6f}")


def _hist_str(counts: np.ndarray) -> str:
    tot = max(int(counts.sum()), 1)
    pct = [100.0 * int(c) / tot for c in counts]
    return f"counts={counts.tolist()} pct=[{', '.join(f'{x:.1f}' for x in pct)}]%"


def _change_class_names(num_classes: int) -> list:
    """Labels for severity_change_12to24h (0/1/2 in CSV)."""
    return [f"change_{i}" for i in range(num_classes)]


@torch.no_grad()
def _collect_head_preds(
    model: EHREncoderTransformer,
    loader: DataLoader,
    device: torch.device,
    head: str,
) -> tuple:
    """Return (y_true, y_pred) for anchor s2f or p2f on a split."""
    labels: list = []
    preds: list = []
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s2f, log_p2f = model(b["ehr_seq"], b["ehr_mask"])
        if head == "s2f":
            ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            logits = log_s2f
            tgt = b["anchor_s2f"]
        elif head == "p2f":
            ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
            logits = log_p2f
            tgt = b["anchor_p2f"]
        else:
            raise ValueError(f"head must be 's2f' or 'p2f', got {head!r}")
        if not ok.any():
            continue
        pred = logits[ok].argmax(1).cpu().numpy()
        y = tgt[ok].cpu().numpy()
        preds.extend(pred.tolist())
        labels.extend(y.tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def _label_count_list(y: np.ndarray, num_classes: int) -> list:
    c = np.bincount(y, minlength=num_classes)
    return [int(x) for x in c]


def _print_head_classification(
    split_name: str,
    head: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    names = _change_class_names(num_classes)
    n = len(y_true)
    if n == 0:
        print(f"\n=== {split_name} — {head.upper()} (no valid samples) ===")
        return {}

    acc = float((y_pred == y_true).mean())
    true_cnt = _label_count_list(y_true, num_classes)
    pred_cnt = _label_count_list(y_pred, num_classes)
    maj_acc = float(np.max(true_cnt)) / n
    n_pred_classes = len(np.unique(y_pred))

    print(f"\n=== {split_name} — {head.upper()} severity_change_12to24h ===")
    print(f"  n={n:,}  accuracy={acc:.4f}  majority_baseline={maj_acc:.4f}")
    print(f"  true counts [{', '.join(names)}]: {true_cnt}")
    print(f"  pred counts [{', '.join(names)}]: {pred_cnt}")
    if n_pred_classes == 1:
        print(f"  WARNING: all predictions are class {int(y_pred[0])} (collapsed to single class)")
    elif acc >= maj_acc - 0.002 and acc <= maj_acc + 0.002:
        print("  NOTE: accuracy ≈ majority baseline — model may be mostly guessing the majority class")
    print("  Classification report:")
    print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(cm)

    report = classification_report(
        y_true, y_pred, target_names=names, output_dict=True, zero_division=0
    )
    return {
        "n": n,
        "accuracy": acc,
        "majority_baseline": maj_acc,
        "true_counts": true_cnt,
        "pred_counts": pred_cnt,
        "n_unique_predictions": int(n_pred_classes),
        "collapsed_to_one_class": n_pred_classes == 1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_split_both_heads(
    model: EHREncoderTransformer,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    num_classes: int,
) -> dict:
    model.eval()
    out = {}
    for head in ("s2f", "p2f"):
        y_true, y_pred = _collect_head_preds(model, loader, device, head)
        out[head] = _print_head_classification(split_name, head, y_true, y_pred, num_classes)
    return out


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
    label_smoothing: float = 0.0,
) -> dict:
    model.eval()
    tot = 0.0
    n_batches = 0
    acc_s_n = acc_s_d = acc_p_n = acc_p_d = 0.0
    ce_s_sum = ce_p_sum = 0.0
    n_ce_s = n_ce_p = 0
    pred_s = np.zeros(3, dtype=np.int64)
    pred_p = np.zeros(3, dtype=np.int64)
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s2f, log_p2f = model(b["ehr_seq"], b["ehr_mask"])
        loss, _ = forward_loss_parts(
            model,
            b,
            p2f_loss_weight=p2f_loss_weight,
            s2f_class_weight=s2f_class_weight,
            p2f_class_weight=p2f_class_weight,
            label_smoothing=label_smoothing,
            log_s2f=log_s2f,
            log_p2f=log_p2f,
        )
        tot += float(loss)
        n_batches += 1
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        if s_ok.any():
            ce_s_sum += float(F.cross_entropy(log_s2f[s_ok], b["anchor_s2f"][s_ok]))
            n_ce_s += 1
            pred = log_s2f[s_ok].argmax(1)
            acc_s_n += (pred == b["anchor_s2f"][s_ok]).float().sum().item()
            acc_s_d += int(s_ok.sum())
            if collect_pred_hist:
                for c in pred.cpu().numpy():
                    pred_s[int(c)] += 1
        if p_ok.any():
            ce_p_sum += float(F.cross_entropy(log_p2f[p_ok], b["anchor_p2f"][p_ok]))
            n_ce_p += 1
            pred = log_p2f[p_ok].argmax(1)
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
    }
    if collect_pred_hist:
        out["pred_hist_s2f"] = pred_s
        out["pred_hist_p2f"] = pred_p
    return out


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(
            "WARNING: CUDA not available; training on CPU. "
            "For Slurm, use partition gpu-common (-p gpu-common) and request a GPU (-G 1)."
        )
    print(f"EHREncoderTransformer  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  No enriched join (missing file): {enr!r}")
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
    cw_mode = "none" if not args.use_class_weights else args.class_weight_mode
    print(
        f"  Loss: p2f_weight={args.p2f_loss_weight}  class_weights={args.use_class_weights}  "
        f"class_weight_mode={cw_mode}  label_smoothing={args.label_smoothing}  "
        f"grad_clip={args.grad_clip}  include_anchor_row={args.include_anchor_row}  "
        f"preprocess=symile_pct+indicator"
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
        idxs = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, idxs.tolist())
        print(f"  Subset max_samples={args.max_samples}")

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    input_dim = base.input_dim
    if isinstance(full_ds, Subset):
        y = y[np.array(full_ds.indices, dtype=np.int64)]
        idx_train, idx_val, idx_test = stratified_train_val_test_indices(
            y, args.train_split, args.val_split, test_split, args.seed
        )
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_anchor_batch,
    )

    model = EHREncoderTransformer(
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_train, n_total = _count_params(model)
    print(f"  Parameters: trainable={n_train:,}  total={n_total:,}  lr={args.lr}  weight_decay={args.weight_decay}")
    _print_module_trainable(model)
    _anchor_label_stats(base, idx_val, "val")
    _anchor_label_stats(base, idx_train, "train")

    s2f_w = p2f_w = None
    if args.use_class_weights:
        s2f_w = _head_class_weights(
            base, idx_train, "anchor_has_s2f", "anchor_s2f_cls", args.num_classes, device, mode=cw_mode
        )
        p2f_w = _head_class_weights(
            base, idx_train, "anchor_has_p2f", "anchor_p2f_cls", args.num_classes, device, mode=cw_mode
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
    best_acc_score = -1.0
    best_acc_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    param_snap_start = _snapshot_trainable(model)
    w_before_step = None

    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        tr_s = tr_p = tr_s_uw = tr_p_uw = 0.0
        n_tr_batches = 0
        last_grad_norm = 0.0
        for batch_idx, batch in enumerate(train_loader):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, parts = forward_loss_parts(model, b, **loss_kw)
            opt.zero_grad()
            loss.backward()
            last_grad_norm = _grad_norm(model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if epoch == 0 and batch_idx == 0:
                w_before_step = _snapshot_trainable(model)
            opt.step()
            if epoch == 0 and batch_idx == 0:
                step_delta = _param_delta_l2(model, w_before_step)
                lens = b["ehr_mask"].long().sum(dim=1)
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
                print(
                    "  [train check epoch1 batch0] "
                    f"loss={float(loss):.4f}  loss_s2f={float(parts['loss_s2f']):.4f}  "
                    f"loss_p2f={float(parts['loss_p2f']):.4f}  grad_norm={last_grad_norm:.6f}  "
                    f"param_delta_after_1step={step_delta:.8f}"
                )
                print(
                    f"    batch: seq_len min/median/max="
                    f"{int(lens.min())}/{int(lens.median())}/{int(lens.max())}  "
                    f"n_s2f={int(s_ok.sum())}  n_p2f={int(p_ok.sum())}"
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
            tr_s_uw += float(parts["loss_s2f_uw"])
            tr_p_uw += float(parts["loss_p2f_uw"])
            n_tr_batches += 1
        tr /= max(n_tr_batches, 1)
        tr_s /= max(n_tr_batches, 1)
        tr_p /= max(n_tr_batches, 1)
        tr_s_uw /= max(n_tr_batches, 1)
        tr_p_uw /= max(n_tr_batches, 1)
        epoch_param_delta = _param_delta_l2(model, param_snap_start) if epoch == 0 else None
        st = eval_loader(model, val_loader, device, collect_pred_hist=True, **loss_kw)
        n_pred_s2f = int(np.count_nonzero(st["pred_hist_s2f"]))
        n_pred_p2f = int(np.count_nonzero(st["pred_hist_p2f"]))
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={tr:.4f}  "
            f"(s2f={tr_s:.4f} p2f={tr_p:.4f})  train_ce_uw_s2f={tr_s_uw:.4f}  "
            f"train_ce_uw_p2f={tr_p_uw:.4f}  val_loss={st['loss']:.4f}  "
            f"val_acc_s2f={st['acc_s2f']:.6f}  val_acc_p2f={st['acc_p2f']:.6f}"
        )
        print(
            f"  val pred_s2f: {_hist_str(st['pred_hist_s2f'])}  "
            f"val pred_p2f: {_hist_str(st['pred_hist_p2f'])}  "
            f"pred_diversity_s2f={n_pred_s2f}/3  pred_diversity_p2f={n_pred_p2f}/3"
        )
        print(
            f"  train diagnostics: last_batch_grad_norm={last_grad_norm:.6f}  "
            f"param_l2={_param_l2(model):.4f}"
            + (
                f"  epoch1_param_delta_vs_init={epoch_param_delta:.8f}"
                if epoch_param_delta is not None
                else ""
            )
        )
        improved = st["loss"] < best_val - args.early_stop_min_delta
        if improved:
            best_val = st["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_acc_s2f": st["acc_s2f"],
                    "val_acc_p2f": st["acc_p2f"],
                    "input_dim": input_dim,
                },
                out_dir / "best_loss.pt",
            )
        else:
            epochs_no_improve += 1

        acc_score = st["acc_s2f"] + st["acc_p2f"]
        if st["acc_s2f"] >= args.checkpoint_min_acc_s2f and acc_score > best_acc_score:
            best_acc_score = acc_score
            best_acc_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": st["loss"],
                    "val_acc_s2f": st["acc_s2f"],
                    "val_acc_p2f": st["acc_p2f"],
                    "input_dim": input_dim,
                },
                out_dir / "best_acc.pt",
            )

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}/{args.epochs} "
                f"(best epoch {best_epoch + 1}, best val_loss={best_val:.4f})"
            )
            stopped_early = True
            break

    torch.save(model.state_dict(), out_dir / "last.pt")
    ckpt_path = out_dir / "best_acc.pt"
    if not ckpt_path.is_file():
        fallback = out_dir / "best_loss.pt"
        if fallback.is_file():
            print("  No best_acc.pt found; falling back to best_loss.pt for evaluation")
            ckpt_path = fallback
    used_ckpt = "last.pt"
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        used_ckpt = ckpt_path.name
    test_st = eval_loader(model, test_loader, device, **loss_kw)
    ckpt_epoch = best_acc_epoch + 1 if best_acc_epoch >= 0 else (best_epoch + 1 if best_epoch >= 0 else None)
    print(
        f"\nTest (summary): loss={test_st['loss']:.4f}  "
        f"acc_s2f={test_st['acc_s2f']:.4f}  acc_p2f={test_st['acc_p2f']:.4f}"
    )
    print(f"  (checkpoint: {ckpt_path.name}  epoch {ckpt_epoch if ckpt_epoch else 'n/a'})")

    val_reports = evaluate_split_both_heads(
        model, val_loader, device, "Val (best checkpoint)", args.num_classes
    )
    test_reports = evaluate_split_both_heads(
        model, test_loader, device, "Test (best checkpoint)", args.num_classes
    )

    results = {
        "task": "ehrencoder_transformer_anchor_s2f_p2f",
        "lookback_hours": [args.lookback_max_hours, args.lookback_min_hours],
        "include_anchor_row": args.include_anchor_row,
        "p2f_loss_weight": args.p2f_loss_weight,
        "use_class_weights": args.use_class_weights,
        "class_weight_mode": cw_mode,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "anchor_pool": args.anchor_pool,
        "best_val_loss": best_val,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_acc_epoch": best_acc_epoch + 1 if best_acc_epoch >= 0 else None,
        "best_val_acc_score": best_acc_score if best_acc_epoch >= 0 else None,
        "checkpoint_used": used_ckpt,
        "stopped_early": stopped_early,
        "test_summary": test_st,
        "val": val_reports,
        "test": test_reports,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report_val.json", "w") as f:
        json.dump(val_reports, f, indent=2)
    with open(out_dir / "classification_report_test.json", "w") as f:
        json.dump(test_reports, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    p.add_argument("--anchor_pool", type=str, default=ANCHOR_POOL, choices=["last", "mean"])
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
    p.add_argument("--include_anchor_row", action=argparse.BooleanOptionalAction, default=INCLUDE_ANCHOR_ROW)
    p.add_argument("--p2f_loss_weight", type=float, default=P2F_LOSS_WEIGHT)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=USE_CLASS_WEIGHTS)
    p.add_argument(
        "--class_weight_mode",
        type=str,
        default=CLASS_WEIGHT_MODE,
        choices=["inverse_freq", "sqrt_inverse", "none"],
    )
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    p.add_argument("--checkpoint_min_acc_s2f", type=float, default=CHECKPOINT_MIN_ACC_S2F)
    a = p.parse_args()
    if not a.max_samples:
        a.max_samples = 0
    main(a)
