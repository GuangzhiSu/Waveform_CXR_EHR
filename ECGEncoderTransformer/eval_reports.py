"""Test/val classification reports for ECGEncoderTransformer (s2f / p2f heads)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _change_class_names(num_classes: int) -> list:
    return [f"change_{i}" for i in range(num_classes)]


def _label_count_list(y: np.ndarray, num_classes: int) -> list:
    return [int(x) for x in np.bincount(y, minlength=num_classes)]


@torch.no_grad()
def collect_head_preds_ecg(model, loader: DataLoader, device: torch.device, head: str) -> tuple:
    labels: list = []
    preds: list = []
    for batch in loader:
        if batch is None:
            continue
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
        if head == "s2f":
            ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
            logits, tgt = log_s, b["anchor_s2f"]
        else:
            ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
            logits, tgt = log_p, b["anchor_p2f"]
        if not ok.any():
            continue
        pred = logits[ok].argmax(1).cpu().numpy()
        y = tgt[ok].cpu().numpy()
        preds.extend(pred.tolist())
        labels.extend(y.tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def print_head_classification(split_name: str, head: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
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
    elif abs(acc - maj_acc) <= 0.002:
        print("  NOTE: accuracy ≈ majority baseline — model may be mostly guessing the majority class")
    print("  Classification report:")
    print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "n": n,
        "accuracy": acc,
        "majority_baseline": maj_acc,
        "true_counts": true_cnt,
        "pred_counts": pred_cnt,
        "n_unique_predictions": int(n_pred_classes),
        "collapsed_to_one_class": n_pred_classes == 1,
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_split_both_heads(model, loader: DataLoader, device: torch.device, split_name: str, num_classes: int) -> dict:
    model.eval()
    out = {}
    for head in ("s2f", "p2f"):
        y_true, y_pred = collect_head_preds_ecg(model, loader, device, head)
        out[head] = print_head_classification(split_name, head, y_true, y_pred, num_classes)
    return out
