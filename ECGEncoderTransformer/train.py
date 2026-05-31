"""End-to-end train: frozen baseline2 ECG encoder + causal transformer + dual s2f/p2f MLP heads."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
_EWT = PROJECT_ROOT / "EHRWindowTransformer"
_EXP_OLD = PROJECT_ROOT / "experiment1(old)"
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "EHRTrend",
    _EWT,
    _EXP,
    _EXP_OLD,
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from common import collate_ecg_window_batch, stratify_labels_from_anchor  # noqa: E402
from config import *  # noqa: F401,F403,E402
from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402
from eval_reports import evaluate_split_both_heads  # noqa: E402
from model import ECGEncoderTransformer  # noqa: E402


def masked_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[valid], y[valid])


def forward_loss_from_logits_parts(batch: dict, log_s: torch.Tensor, log_p: torch.Tensor) -> tuple:
    device = log_s.device
    s_tgt = batch["anchor_s2f"].to(device)
    p_tgt = batch["anchor_p2f"].to(device)
    s_ok = batch["anchor_has_s2f"].to(device) & (s_tgt >= 0)
    p_ok = batch["anchor_has_p2f"].to(device) & (p_tgt >= 0)
    loss_s = masked_ce(log_s, s_tgt, s_ok)
    loss_p = masked_ce(log_p, p_tgt, p_ok)
    return loss_s + loss_p, {"loss_s2f": loss_s, "loss_p2f": loss_p, "log_s2f": log_s, "log_p2f": log_p}


def _count_params(model: ECGEncoderTransformer) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _grad_norm(model: ECGEncoderTransformer) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        sq += float(p.grad.detach().pow(2).sum())
    return sq**0.5


def _param_l2(model: ECGEncoderTransformer) -> float:
    sq = sum(float(p.detach().pow(2).sum()) for p in model.parameters() if p.requires_grad)
    return sq**0.5


def _snapshot_trainable(model: ECGEncoderTransformer) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _param_delta_l2(model: ECGEncoderTransformer, before: dict) -> float:
    sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in before:
            continue
        sq += float((p.detach() - before[n]).pow(2).sum())
    return sq**0.5


def _print_module_trainable(model: ECGEncoderTransformer) -> None:
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


@torch.no_grad()
def eval_loader(model, loader, device, collect_pred_hist: bool = False) -> dict:
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
        log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
        loss, _ = forward_loss_from_logits_parts(b, log_s, log_p)
        tot += float(loss)
        n_batches += 1
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        if s_ok.any():
            ce_s_sum += float(F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok]))
            n_ce_s += 1
            pred = log_s[s_ok].argmax(1)
            acc_s_n += (pred == b["anchor_s2f"][s_ok]).float().sum().item()
            acc_s_d += int(s_ok.sum())
            if collect_pred_hist:
                for c in pred.cpu().numpy():
                    pred_s[int(c)] += 1
        if p_ok.any():
            ce_p_sum += float(F.cross_entropy(log_p[p_ok], b["anchor_p2f"][p_ok]))
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
    }
    if collect_pred_hist:
        out["pred_hist_s2f"] = pred_s
        out["pred_hist_p2f"] = pred_p
    return out


def _resolve_ecg_ckpt(path: str) -> Optional[str]:
    if path and os.path.isfile(path):
        return path
    if path:
        print(f"  WARNING: ECG checkpoint not found: {path}")
    return None


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
    print(f"ECGEncoderTransformer  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if not Path(args.ecg_labeled_csv).is_file():
        raise FileNotFoundError(f"Labeled ECG CSV not found: {args.ecg_labeled_csv}")

    print(f"  Dataset: ECGLabeledCatalogDataset  csv={args.ecg_labeled_csv}")
    full_ds = ECGLabeledCatalogDataset(
        labeled_csv=args.ecg_labeled_csv,
        ecg_target_len=args.ecg_target_len,
        normalize_ecg_per_lead_flag=not args.no_ecg_normalize,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        require_hours_in_window=not args.no_hours_filter,
    )
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
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ecg_window_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ecg_window_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ecg_window_batch,
    )

    ecg_ckpt = _resolve_ecg_ckpt(args.ecg_ckpt)
    model = ECGEncoderTransformer(
        ecg_dim=args.ecg_dim,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        anchor_pool=args.anchor_pool,
        ecg_ckpt_path=ecg_ckpt,
        input_channels=args.input_channels,
        sig_len=args.ecg_target_len,
        freeze_ecg=not args.unfreeze_ecg,
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_train, n_total = _count_params(model)
    print(f"  Parameters: trainable={n_train:,}  total={n_total:,}  lr={args.lr}  weight_decay={args.weight_decay}")
    print(f"  ECG ckpt: {ecg_ckpt or 'none (random init)'}  freeze_ecg={not args.unfreeze_ecg}")
    _print_module_trainable(model)
    _anchor_label_stats(base, idx_val, "val")
    _anchor_label_stats(base, idx_train, "train")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False
    param_snap_start = _snapshot_trainable(model)
    w_before_step = None

    for epoch in range(args.epochs):
        model.train()
        tr = tr_s = tr_p = 0.0
        n_tr_batches = 0
        last_grad_norm = 0.0
        for batch_idx, batch in enumerate(train_loader):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, log_p = model(b["ecg_seq"], b["ecg_mask"])
            loss, parts = forward_loss_from_logits_parts(b, log_s, log_p)
            opt.zero_grad()
            loss.backward()
            last_grad_norm = _grad_norm(model)
            if epoch == 0 and batch_idx == 0:
                w_before_step = _snapshot_trainable(model)
            opt.step()
            if epoch == 0 and batch_idx == 0:
                step_delta = _param_delta_l2(model, w_before_step)
                lens = b["ecg_mask"].long().sum(dim=1)
                valid_ecg = int(b["ecg_mask"].sum())
                s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
                p_ok = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
                print(
                    "  [train check epoch1 batch0] "
                    f"loss={float(loss):.4f}  loss_s2f={float(parts['loss_s2f']):.4f}  "
                    f"loss_p2f={float(parts['loss_p2f']):.4f}  grad_norm={last_grad_norm:.6f}  "
                    f"param_delta_after_1step={step_delta:.8f}"
                )
                print(
                    f"    batch: ecg_seq={tuple(b['ecg_seq'].shape)}  "
                    f"valid_ecg_slots={valid_ecg}/{b['ecg_mask'].numel()}  "
                    f"seq_len min/median/max={int(lens.min())}/{int(lens.median())}/{int(lens.max())}  "
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
            n_tr_batches += 1
        tr /= max(n_tr_batches, 1)
        tr_s /= max(n_tr_batches, 1)
        tr_p /= max(n_tr_batches, 1)
        epoch_param_delta = _param_delta_l2(model, param_snap_start) if epoch == 0 else None
        st = eval_loader(model, val_loader, device, collect_pred_hist=True)
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={tr:.4f}  "
            f"(s2f={tr_s:.4f} p2f={tr_p:.4f})  val_loss={st['loss']:.4f}  "
            f"val_acc_s2f={st['acc_s2f']:.6f}  val_acc_p2f={st['acc_p2f']:.6f}"
        )
        print(
            f"  val pred_s2f: {_hist_str(st['pred_hist_s2f'])}  "
            f"val pred_p2f: {_hist_str(st['pred_hist_p2f'])}"
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
    if (out_dir / "best.pt").is_file():
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    test_st = eval_loader(model, test_loader, device)
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
        "task": "ecg_encoder_transformer_labeled_catalog",
        "ecg_labeled_csv": args.ecg_labeled_csv,
        "ecg_ckpt": ecg_ckpt,
        "lookback_hours": [args.lookback_max_hours, args.lookback_min_hours],
        "anchor_pool": args.anchor_pool,
        "freeze_ecg": not args.unfreeze_ecg,
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
    p = argparse.ArgumentParser(
        description="ECGEncoderTransformer: frozen baseline2 ECG encoder + causal transformer + MLP heads"
    )
    p.add_argument("--ecg_labeled_csv", default=ECG_CATALOG_LABELED_CSV)
    p.add_argument("--no_hours_filter", action="store_true")
    p.add_argument("--ecg_ckpt", default=ECG_CKPT)
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--ecg_target_len", type=int, default=ECG_TARGET_LEN)
    p.add_argument("--input_channels", type=int, default=INPUT_CHANNELS)
    p.add_argument("--no_ecg_normalize", action="store_true")
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--ecg_dim", type=int, default=ECG_DIM)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--num_transformer_layers", type=int, default=NUM_TRANSFORMER_LAYERS)
    p.add_argument("--num_heads", type=int, default=NUM_HEADS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--anchor_pool", type=str, default=ANCHOR_POOL, choices=["last", "mean"])
    p.add_argument("--unfreeze_ecg", action="store_true")
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
    a = p.parse_args()
    if not a.max_samples:
        a.max_samples = 0
    main(a)
