"""
Train baseline: predict oxygenation from ECG + CXR.
EHR oxygenation (e.g. spo2) is ground truth; model uses only ECG + CXR.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import (
    DATA_CSV,
    CXR_ROOT,
    METADATA_PATH,
    ECG_CKPT,
    VIT_PATH,
    HIDDEN_DIM,
    FREEZE_ENCODERS,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    SEED,
    NUM_WORKERS,
    TARGET_COL,
)
from dataset import WaveformCXREHRDataset
from model import FusionBaseline


def collate_fn(batch):
    cxr = torch.stack([b["cxr"] for b in batch])
    signal = torch.stack([b["signal"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    return {"cxr": cxr, "signal": signal, "target": target}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    full_ds = WaveformCXREHRDataset(
        csv_path=args.csv_path,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        target_col=args.target_col,
        split="train",
    )
    n = len(full_ds)
    n_train = int(n * args.train_split)
    n_val = int(n * args.val_split)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

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

    # Model: ECG + CXR → oxygenation (EHR is ground truth only)
    model = FusionBaseline(
        hidden_dim=args.hidden_dim,
        vit_path=args.vit_path,
        ecg_ckpt_path=args.ecg_ckpt if os.path.exists(args.ecg_ckpt) else None,
        freeze_encoders=args.freeze_encoders,
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            cxr = batch["cxr"].to(device)
            signal = batch["signal"].to(device)
            target = batch["target"].to(device)

            pred = model(cxr, signal)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation: MSE, MAE, RMSE
        model.eval()
        val_loss = 0
        val_mae_sum = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                cxr = batch["cxr"].to(device)
                signal = batch["signal"].to(device)
                target = batch["target"].to(device)
                pred = model(cxr, signal)
                loss = criterion(pred, target)
                val_loss += loss.item()
                mae = (pred - target).abs().sum().item()
                val_mae_sum += mae
                val_n += target.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae = val_mae_sum / val_n if val_n > 0 else 0
        val_rmse = (val_loss ** 0.5)

        print(f"Epoch {epoch+1}/{args.epochs}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  val_mae={val_mae:.4f}  val_rmse={val_rmse:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_mae": val_mae, "val_rmse": val_rmse},
                out_dir / "best.pt",
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()},
                out_dir / f"checkpoint_{epoch+1}.pt",
            )

    print(f"Best val_mae: {best_val_mae:.4f}")

    # Final evaluation on held-out test set
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()
    test_loss = 0
    test_mae_sum = 0
    test_n = 0
    with torch.no_grad():
        for batch in test_loader:
            cxr = batch["cxr"].to(device)
            signal = batch["signal"].to(device)
            target = batch["target"].to(device)
            pred = model(cxr, signal)
            loss = criterion(pred, target)
            test_loss += loss.item()
            mae = (pred - target).abs().sum().item()
            test_mae_sum += mae
            test_n += target.size(0)
    test_mse = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    test_mae = test_mae_sum / test_n if test_n > 0 else 0
    test_rmse = (test_mse ** 0.5)
    print(f"\n=== Test set (held out) ===")
    print(f"  test_mse:  {test_mse:.4f}")
    print(f"  test_mae:  {test_mae:.4f}")
    print(f"  test_rmse: {test_rmse:.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "best_val_mae": best_val_mae,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mse": test_mse,
            "target_col": args.target_col,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DATA_CSV)
    parser.add_argument("--cxr_root", default=CXR_ROOT)
    parser.add_argument("--metadata_path", default=METADATA_PATH)
    parser.add_argument("--ecg_ckpt", default=ECG_CKPT)
    parser.add_argument("--vit_path", default=VIT_PATH)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--freeze_encoders", action="store_true", default=FREEZE_ENCODERS, help="Freeze pretrained encoders")
    parser.add_argument("--no_freeze", action="store_true", help="Don't freeze encoders (finetune all)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--test_split", type=float, default=TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--target_col", default=TARGET_COL)
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()
    if args.no_freeze:
        args.freeze_encoders = False
    main(args)
