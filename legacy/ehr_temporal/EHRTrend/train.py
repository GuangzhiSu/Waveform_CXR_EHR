"""Train EHR trend classification (decrease/remain/increase)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BaselineExperiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classification_utils import compute_class_weights, make_subset, stratified_train_val_test_indices
from config import *
from dataset import EHRTrendDataset
from model import EHRTrendBaseline


def collate_fn(batch):
    lengths = [b["ehr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    feat = batch[0]["ehr_seq"].shape[1]
    bsz = len(batch)

    seq = torch.zeros(bsz, max_len, feat, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        t = b["ehr_seq"].shape[0]
        seq[i, :t] = b["ehr_seq"]
        mask[i, :t] = True
    return {"ehr_seq": seq, "ehr_mask": mask, "label": labels}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EHR Trend Classification. Device: {device}")

    full_ds = EHRTrendDataset(
        anchor_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
    )
    input_dim = full_ds.input_dim
    print(f"EHR row input dim (percentile vector): {input_dim}")

    y = full_ds.anchor_labels
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
        print(f"  {name} trend counts [decrease, remain, increase]: {c.tolist()}")

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

    model = EHRTrendBaseline(
        input_dim=input_dim,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        pooling_stats=args.pooling_stats,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)

    class_weights = compute_class_weights(y[idx_train], args.num_classes, device)
    criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()
    print(f"  class weights (train CE): {class_weights.cpu().numpy().round(4).tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = model(batch["ehr_seq"].to(device), batch["ehr_mask"].to(device))
            loss = criterion_train(pred, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["ehr_seq"].to(device), batch["ehr_mask"].to(device))
                target = batch["label"].to(device)
                val_loss_sum += criterion_eval(pred, target).item()
                val_correct += (pred.argmax(1) == target).sum().item()
                val_total += target.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / len(val_loader) if len(val_loader) else 0.0

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
            pred = model(batch["ehr_seq"].to(device), batch["ehr_mask"].to(device))
            target = batch["label"].to(device)
            test_correct += (pred.argmax(1) == target).sum().item()
            test_total += target.size(0)
            all_preds.extend(pred.argmax(1).cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
    test_acc = test_correct / test_total if test_total else 0.0

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ["decrease", "remain", "increase"]
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
        "task": "EHR_trend_classification",
        "history_csv": args.history_csv,
        "anchor_csv": args.anchor_csv,
        "schema_csv": args.schema_csv,
        "lookback_min_hours": args.lookback_min_hours,
        "lookback_max_hours": args.lookback_max_hours,
        "pooling_stats": list(args.pooling_stats),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_csv", default=ANCHOR_CSV)
    parser.add_argument("--history_csv", default=SOURCE_CSV)
    parser.add_argument("--schema_csv", default=SCHEMA_CSV)
    parser.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    parser.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--pooling_stats", nargs="+", default=list(POOLING_STATS))
    parser.add_argument("--head_hidden_dim", type=int, default=HEAD_HIDDEN_DIM)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "output"))
    args = parser.parse_args()
    main(args)
