"""
Train multimodal ECG + CXR ARDS severity classification (concat encoder embeddings + MLP head).
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _argv_value(flag: str):
    """Read ``--flag value`` from sys.argv (for MedTVT path hints before argparse)."""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


BE = Path(__file__).resolve().parents[1]
REPO = BE.parent
_EXP_OLD = REPO / "experiment1(old)"
sys.path.insert(0, str(_EXP_OLD))
sys.path.insert(0, str(BE))

# Must run before ``import MultimodalECGCXR.model`` (loads ``llama`` via SignalEncoder).
from medtvt_paths import ensure_medtvt_on_syspath  # noqa: E402

from MultimodalECGCXR.config import ECG_CKPT as _CFG_ECG_CKPT  # noqa: E402
from MultimodalECGCXR.config import VIT_PATH as _CFG_VIT_PATH  # noqa: E402

_early_vit = _argv_value("--vit_path")
_early_ecg = _argv_value("--ecg_ckpt")
_MM_MEDTVT_ROOT = ensure_medtvt_on_syspath(
    _early_vit,
    _early_ecg,
    _CFG_VIT_PATH,
    _CFG_ECG_CKPT,
)
print(f"  MedTVT-R1 (llama): sys.path[0]={_MM_MEDTVT_ROOT!r}  (required for ECG xresnet)")

sys.path.insert(0, str(BE / "CXRUni"))

from classification_utils import (
    compute_class_weights,
    print_multimodal_forward_spread,
    print_tensor_batch_diagnostics,
    print_trainable_param_counts,
    scan_cxr_train_files,
    scan_ecg_train_files,
    stratified_train_val_test_indices,
    total_grad_l2_norm,
)
from MultimodalECGCXR.config import *
from MultimodalECGCXR.dataset import MultimodalECGCXRDataset
from MultimodalECGCXR.model import MultimodalECGCXRBaseline

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collate_fn(batch):
    return {
        "signal": torch.stack([b["signal"] for b in batch]),
        "cxr": torch.stack([b["cxr"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def build_optimizer(model, args):
    """Frozen encoders: single group. Unfrozen: ViT + xresnet backbones vs proj+head."""
    if args.freeze_encoder:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    vit_params = [
        p for n, p in model.named_parameters()
        if n.startswith("cxr_encoder.vit") and p.requires_grad
    ]
    sig_enc_params = [
        p for n, p in model.named_parameters()
        if n.startswith("signal_encoder.encoder") and p.requires_grad
    ]
    rest = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not n.startswith("cxr_encoder.vit")
        and not n.startswith("signal_encoder.encoder")
    ]
    print(
        f"  Optimizer: backbone lr={args.backbone_lr}, wd={args.backbone_weight_decay} | "
        f"proj+head lr={args.lr}, wd={args.weight_decay}"
    )
    return torch.optim.AdamW(
        [
            {"params": vit_params, "lr": args.backbone_lr, "weight_decay": args.backbone_weight_decay},
            {"params": sig_enc_params, "lr": args.backbone_lr, "weight_decay": args.backbone_weight_decay},
            {"params": rest, "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multimodal ECG+CXR ARDS Classification. Device: {device}")
    print(
        f"  torch.cuda.is_available()={torch.cuda.is_available()}  "
        f"freeze_encoder (both)={args.freeze_encoder}"
    )
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    full_ds = MultimodalECGCXRDataset(
        csv_path=args.csv_path,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        split="train",
        imagenet_normalize=not args.no_cxr_normalize,
        normalize_ecg_per_lead=not args.no_ecg_normalize,
    )
    print(
        f"  CXR ImageNet normalize: {not args.no_cxr_normalize}  "
        f"(train=RandomCrop, val/test=CenterCrop)"
    )
    print(f"  ECG per-lead z-score: {not args.no_ecg_normalize}")

    test_split = 1.0 - args.train_split - args.val_split
    y = full_ds.df["p2f_class"].values
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    shared_kw = dict(
        df=full_ds.df,
        cxr_root=args.cxr_root,
        metadata_path=None,
        imagenet_normalize=not args.no_cxr_normalize,
        normalize_ecg_per_lead=not args.no_ecg_normalize,
    )
    train_ds = MultimodalECGCXRDataset(split="train", indices=idx_train, **shared_kw)
    val_ds = MultimodalECGCXRDataset(split="val", indices=idx_val, **shared_kw)
    test_ds = MultimodalECGCXRDataset(split="test", indices=idx_test, **shared_kw)
    n_train, n_val, n_test = len(idx_train), len(idx_val), len(idx_test)
    print(f"Split (stratified): train={n_train}, val={n_val}, test={n_test}")
    for name, yi in (
        ("train", y[idx_train]),
        ("val", y[idx_val]),
        ("test", y[idx_test]),
    ):
        c = np.bincount(yi.astype(int), minlength=args.num_classes)
        print(f"  {name} class counts [Severe, Moderate, Mild]: {c.tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    ecg_ckpt = args.ecg_ckpt if args.ecg_ckpt and os.path.exists(args.ecg_ckpt) else None
    if ecg_ckpt:
        print(f"  ECG xresnet checkpoint: {ecg_ckpt}")
    else:
        print("  ECG xresnet checkpoint: (none) — SignalEncoder backbone randomly initialized")
    model = MultimodalECGCXRBaseline(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        vit_path=args.vit_path,
        ecg_ckpt_path=ecg_ckpt,
        ecg_encoder_kind=args.ecg_encoder,
        freeze_encoder=args.freeze_encoder,
    )
    model = model.to(device)
    print_trainable_param_counts(model, "Multimodal ECG+CXR")

    if not args.skip_input_diag:
        print("\n=== Input diagnostics (train split) ===")
        scan_ecg_train_files(train_ds, max_scan=min(512, len(train_ds)), seed=args.seed)
        scan_cxr_train_files(train_ds, max_scan=min(512, len(train_ds)), seed=args.seed)
        diag_batch = next(iter(train_loader))
        print_tensor_batch_diagnostics("ecg", diag_batch["signal"])
        print_tensor_batch_diagnostics("cxr", diag_batch["cxr"])
        print_multimodal_forward_spread(model, train_loader, device, batch=diag_batch)
        print("=== End diagnostics ===\n")

    class_weights = compute_class_weights(y[idx_train], args.num_classes, device)
    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    criterion_eval = nn.CrossEntropyLoss()
    print(f"  class weights (train CE): {class_weights.cpu().numpy().round(4).tolist()}")
    print(f"  label_smoothing (train only): {args.label_smoothing}")
    optimizer = build_optimizer(model, args)

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Severe", "Moderate", "Mild"]

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            cxr = batch["cxr"].to(device)
            sig = batch["signal"].to(device)
            labels = batch["label"].to(device)
            pred = model(cxr, sig)
            loss = criterion_train(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.train_diag and epoch == 0 and step == 0:
                gnorm, gn = total_grad_l2_norm(model)
                print(
                    f"  [train_diag] epoch1 batch0: loss={loss.item():.6f}  "
                    f"||grad||_2={gnorm:.6f}  tensors_with_grad={gn}"
                )
                with torch.no_grad():
                    p0 = pred.argmax(dim=1)
                    pc = torch.bincount(p0, minlength=args.num_classes).cpu().tolist()
                    lc = torch.bincount(labels, minlength=args.num_classes).cpu().tolist()
                print(
                    f"  [train_diag] batch0 argmax [{', '.join(class_names)}]: pred={pc}  labels={lc}"
                )
                print(
                    f"  [train_diag] batch0 logits mean: "
                    f"{pred.detach().mean(0).cpu().numpy().round(4).tolist()}"
                )
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        val_pred_hist = torch.zeros(args.num_classes, dtype=torch.long)
        val_label_hist = torch.zeros(args.num_classes, dtype=torch.long)
        with torch.no_grad():
            for batch in val_loader:
                cxr = batch["cxr"].to(device)
                sig = batch["signal"].to(device)
                target = batch["label"].to(device)
                pred = model(cxr, sig)
                val_loss_sum += criterion_eval(pred, target).item()
                val_correct += (pred.argmax(1) == target).sum().item()
                val_total += target.size(0)
                if args.train_diag:
                    p = pred.argmax(1).cpu()
                    val_pred_hist += torch.bincount(p, minlength=args.num_classes)
                    val_label_hist += torch.bincount(target.cpu(), minlength=args.num_classes)
        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / len(val_loader) if val_loader else 0.0
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if args.train_diag:
            vp = val_pred_hist.tolist()
            vl = val_label_hist.tolist()
            n_used = sum(1 for c in vp if c > 0)
            max_pred = max(vp) if vp else 0
            frac_dom = max_pred / val_total if val_total else 0.0
            if frac_dom >= 0.99:
                collapse_note = "-> ~all val preds one class (collapsed)"
            elif n_used <= 2:
                collapse_note = "-> only 2/3 classes predicted (one class nearly absent)"
            else:
                collapse_note = "-> predictions spread across 3 classes"
            print(
                f"  [train_diag] val counts [{', '.join(class_names)}]: "
                f"pred={vp}  true={vl}  {collapse_note}"
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                out_dir / "best.pt",
            )
        if (epoch + 1) % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / f"checkpoint_{epoch+1}.pt")

    print(f"Best val_acc: {best_val_acc:.4f}")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            cxr = batch["cxr"].to(device)
            sig = batch["signal"].to(device)
            target = batch["label"].to(device)
            pred = model(cxr, sig)
            test_correct += (pred.argmax(1) == target).sum().item()
            test_total += target.size(0)
            all_preds.extend(pred.argmax(1).cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
    test_acc = test_correct / test_total if test_total else 0.0

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ["Severe", "Moderate", "Mild"]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== Test set ===")
    print(f"  test_acc: {test_acc:.4f}")
    print("  Classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("  Confusion matrix:")
    print(cm)

    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
        "modality": "ECG+CXR multimodal (concat)",
        "task": "ARDS_severity_classification",
        "imagenet_normalize": not args.no_cxr_normalize,
        "ecg_per_lead_normalize": not args.no_ecg_normalize,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "freeze_encoder": args.freeze_encoder,
        "backbone_lr": args.backbone_lr if not args.freeze_encoder else None,
        "backbone_weight_decay": args.backbone_weight_decay if not args.freeze_encoder else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=MULTIMODAL_CSV)
    parser.add_argument("--cxr_root", default=CXR_ROOT)
    parser.add_argument("--metadata_path", default=METADATA_PATH)
    parser.add_argument("--vit_path", default=VIT_PATH)
    parser.add_argument("--ecg_ckpt", default=ECG_CKPT, help="Path to ECG xresnet checkpoint (optional)")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument(
        "--ecg_encoder",
        type=str,
        default="cnn",
        choices=["cnn", "transformer"],
        help="ECG encoder implementation from models/encoders/ecg.py",
    )
    parser.add_argument("--freeze_encoder", action="store_true", default=FREEZE_ENCODER)
    parser.add_argument("--no_freeze", action="store_true")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--no_cxr_normalize", action="store_true", help="Skip ImageNet norm on CXR")
    parser.add_argument("--no_ecg_normalize", action="store_true", help="Skip per-lead z-score on ECG")
    parser.add_argument("--backbone_lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--backbone_weight_decay", type=float, default=BACKBONE_WEIGHT_DECAY)
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument(
        "--skip_input_diag",
        action="store_true",
        help="Skip file/tensor/forward diagnostics at startup",
    )
    parser.add_argument(
        "--train_diag",
        action="store_true",
        help="First-batch grad norm + per-epoch val pred vs label histograms (collapse debugging)",
    )
    args = parser.parse_args()
    if args.no_freeze:
        args.freeze_encoder = False
    main(args)
