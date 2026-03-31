"""
Train ECG ARDS severity classification baseline.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_utils import (
    compute_class_weights,
    make_subset,
    stratified_train_val_test_indices,
)
from ECGUni.dataset import ECGClassificationDataset
from ECGUni.model import ECGClassificationBaseline
from ECGUni.config import *

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collate_fn(batch):
    signal = torch.stack([b["signal"] for b in batch])
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"signal": signal, "label": label}


def build_optimizer(model, args, trainable):
    """AdamW: separate LR/WD for classification head vs LoRA when LoRA is on."""
    head_params, lora_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(p)
        else:
            head_params.append(p)
    if args.use_lora and lora_params:
        groups = [
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": args.lora_weight_decay},
        ]
        print(
            f"  Optimizer: head lr={args.lr}, wd={args.weight_decay} | "
            f"LoRA lr={args.lora_lr}, wd={args.lora_weight_decay}"
        )
        return torch.optim.AdamW(groups)
    return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ECG ARDS Classification. Device: {device}")

    full_ds = ECGClassificationDataset(
        csv_path=args.csv_path, split="train", normalize_per_lead=not args.no_normalize
    )
    print(f"  ECG per-lead z-score (time): {not args.no_normalize}")
    n = len(full_ds)
    test_split = 1.0 - args.train_split - args.val_split
    y = full_ds.df["p2f_class"].values
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    n_train, n_val, n_test = len(idx_train), len(idx_val), len(idx_test)
    print(f"Split (stratified): train={n_train}, val={n_val}, test={n_test}")
    for name, yi in (
        ("train", y[idx_train]),
        ("val", y[idx_val]),
        ("test", y[idx_test]),
    ):
        c = np.bincount(yi.astype(int), minlength=args.num_classes)
        print(f"  {name} class counts [Severe, Moderate, Mild]: {c.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    ecg_ckpt = args.ecg_ckpt if args.ecg_ckpt and os.path.exists(args.ecg_ckpt) else None
    if args.use_lora and not args.freeze_encoder:
        raise ValueError(
            "ECG LoRA: use frozen backbone with LoRA (default --freeze_encoder). "
            "Do not combine --use_lora with --no_freeze."
        )
    model = ECGClassificationBaseline(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        ecg_ckpt_path=ecg_ckpt,
        freeze_encoder=args.freeze_encoder,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable parameters: {n_trainable:,}  (LoRA={args.use_lora})")
    if args.use_lora:
        n_wrap = getattr(model, "_lora_encoder_layers", 0)
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, encoder layers wrapped={n_wrap}")
    class_weights = compute_class_weights(y[idx_train], args.num_classes, device)
    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    criterion_eval = nn.CrossEntropyLoss()
    print(f"  class weights (train CE): {class_weights.cpu().numpy().round(4).tolist()}")
    print(f"  label_smoothing (train only): {args.label_smoothing}")
    optimizer = build_optimizer(model, args, trainable)

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            pred = model(batch["signal"].to(device))
            loss = criterion_train(pred, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["signal"].to(device))
                target = batch["label"].to(device)
                val_loss_sum += criterion_eval(pred, target).item()
                val_correct += (pred.argmax(1) == target).sum().item()
                val_total += target.size(0)
        val_acc = val_correct / val_total if val_total else 0
        val_loss = val_loss_sum / len(val_loader) if val_loader else 0

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, out_dir / "best.pt")
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
            pred = model(batch["signal"].to(device))
            target = batch["label"].to(device)
            test_correct += (pred.argmax(1) == target).sum().item()
            test_total += target.size(0)
            all_preds.extend(pred.argmax(1).cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
    test_acc = test_correct / test_total if test_total else 0

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
        "modality": "ECG-only",
        "task": "ARDS_severity_classification",
        "normalize_per_lead": not args.no_normalize,
        "lr": args.lr,
        "lora_lr": args.lora_lr if args.use_lora else None,
        "weight_decay": args.weight_decay,
        "lora_weight_decay": args.lora_weight_decay if args.use_lora else None,
        "label_smoothing": args.label_smoothing,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DATA_CSV)
    parser.add_argument("--ecg_ckpt", default=ECG_CKPT)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--freeze_encoder", action="store_true", default=FREEZE_ENCODER)
    parser.add_argument("--no_freeze", action="store_true")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lora_lr", type=float, default=LORA_LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lora_weight_decay", type=float, default=LORA_WEIGHT_DECAY)
    parser.add_argument(
        "--label_smoothing", type=float, default=LABEL_SMOOTHING,
        help="CrossEntropyLoss label smoothing (train only; eval unsmoothed)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable per-lead z-score over time (use raw WFDB amplitudes after resample)",
    )
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--use_lora", action="store_true", default=USE_LORA)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (overrides config USE_LORA)")
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=float, default=LORA_ALPHA)
    args = parser.parse_args()
    if args.no_freeze:
        args.freeze_encoder = False
    if args.no_lora:
        args.use_lora = False
    main(args)
