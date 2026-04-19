"""Train ECG temporal ARDS severity classification baseline."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _argv_value(flag: str):
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


BE = Path(__file__).resolve().parents[1]
REPO = BE.parent
sys.path.insert(0, str(BE))
_EXP_OLD = REPO / "experiment1(old)"
if _EXP_OLD.is_dir():
    sys.path.insert(0, str(_EXP_OLD))

from medtvt_paths import ensure_medtvt_on_syspath  # noqa: E402
from ECGUni.config import ECG_CKPT as _CFG_ECG_CKPT  # noqa: E402

ensure_medtvt_on_syspath(_argv_value("--ecg_ckpt"), _CFG_ECG_CKPT)

from classification_utils import compute_class_weights, make_subset, stratified_train_val_test_indices
from ECGUni.config import *
from ECGUni.dataset import ECGTemporalClassificationDataset
from ECGUni.model import ECGTemporalClassificationBaseline


def collate_fn(batch):
    lens = [b["signal_seq"].shape[0] for b in batch]
    max_t = max(lens)
    bsz = len(batch)
    c, l = batch[0]["signal_seq"].shape[1:]
    seq = torch.zeros(bsz, max_t, c, l, dtype=torch.float32)
    mask = torch.zeros(bsz, max_t, dtype=torch.bool)
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["signal_seq"].shape[0]
        seq[i, :t] = b["signal_seq"]
        mask[i, :t] = True
    return {"signal_seq": seq, "signal_mask": mask, "label": label}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ECG ARDS Classification (temporal). Device: {device}")

    full_ds = ECGTemporalClassificationDataset(
        csv_path=args.csv_path,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        normalize_per_lead=not args.no_normalize,
    )
    print(f"  ECG per-lead z-score (time): {not args.no_normalize}")

    y = full_ds.labels
    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    print(f"Split (stratified): train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    for name, yi in (("train", y[idx_train]), ("val", y[idx_val]), ("test", y[idx_test])):
        c = np.bincount(yi.astype(int), minlength=args.num_classes)
        print(f"  {name} class counts [Severe, Moderate, Mild]: {c.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    ecg_ckpt = args.ecg_ckpt if args.ecg_ckpt and os.path.exists(args.ecg_ckpt) else None
    model = ECGTemporalClassificationBaseline(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        ecg_ckpt_path=ecg_ckpt,
        ecg_encoder_kind=args.ecg_encoder,
        freeze_encoder=args.freeze_encoder,
        pooling_stats=args.pooling_stats,
    ).to(device)

    class_weights = compute_class_weights(y[idx_train], args.num_classes, device)
    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    criterion_eval = nn.CrossEntropyLoss()
    print(f"  class weights (train CE): {class_weights.cpu().numpy().round(4).tolist()}")
    print(f"  label_smoothing (train only): {args.label_smoothing}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = model(batch["signal_seq"].to(device), batch["signal_mask"].to(device))
            labels = batch["label"].to(device)
            loss = criterion_train(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["signal_seq"].to(device), batch["signal_mask"].to(device))
                target = batch["label"].to(device)
                val_loss_sum += criterion_eval(pred, target).item()
                val_correct += (pred.argmax(1) == target).sum().item()
                val_total += target.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / len(val_loader) if val_loader else 0.0

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
            pred = model(batch["signal_seq"].to(device), batch["signal_mask"].to(device))
            target = batch["label"].to(device)
            test_correct += (pred.argmax(1) == target).sum().item()
            test_total += target.size(0)
            all_preds.extend(pred.argmax(1).cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
    test_acc = test_correct / test_total if test_total else 0.0

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ["Severe", "Moderate", "Mild"]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== Test set ===")
    print(f"  test_acc: {test_acc:.4f}")
    print("  Classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("  Confusion matrix:")
    print(cm)

    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
        "modality": "ECG-only temporal",
        "task": "ARDS_severity_classification",
        "normalize_per_lead": not args.no_normalize,
        "lookback_min_hours": args.lookback_min_hours,
        "lookback_max_hours": args.lookback_max_hours,
        "pooling_stats": list(args.pooling_stats),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DATA_CSV)
    parser.add_argument("--ecg_ckpt", default=ECG_CKPT)
    parser.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    parser.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
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
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--pooling_stats", nargs="+", default=list(POOLING_STATS))
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()
    if args.no_freeze:
        args.freeze_encoder = False
    main(args)
