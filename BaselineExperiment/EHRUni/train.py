"""
Train EHR ARDS severity classification baseline.
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
from EHRUni.dataset import EHRClassificationDataset
from EHRUni.model import EHRClassificationBaseline
from EHRUni.config import *

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collate_fn(batch):
    ehr = torch.stack([b["ehr"] for b in batch])
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"ehr": ehr, "label": label}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EHR ARDS Classification. Device: {device}")

    full_ds = EHRClassificationDataset(
        csv_path=args.csv_path,
        feature_cols=EHR_FEATURE_COLS,
        scaler=None,
    )
    input_dim = full_ds.input_dim
    print(f"EHR input dim: {input_dim}")

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

    model = EHRClassificationBaseline(
        input_dim=input_dim,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    model = model.to(device)
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
        train_loss = 0
        for batch in train_loader:
            pred = model(batch["ehr"].to(device))
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
                pred = model(batch["ehr"].to(device))
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
            pred = model(batch["ehr"].to(device))
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
        "modality": "EHR-only",
        "task": "ARDS_severity_classification",
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DATA_CSV)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()
    main(args)
