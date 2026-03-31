"""
Train baseline3: CXR-only → predict oxygenation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.dataset import WaveformCXREHRDataset
from baseline3.model import CXROnlyBaseline
from baseline3.config import *

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


def collate_fn(batch):
    cxr = torch.stack([b["cxr"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    return {"cxr": cxr, "target": target}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline3: CXR-only. Device: {device}")

    full_ds = WaveformCXREHRDataset(
        csv_path=args.csv_path,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        target_col=args.target_col,
        split="train",
        load_cxr=True,
        load_signal=False,
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = CXROnlyBaseline(
        hidden_dim=args.hidden_dim,
        vit_path=args.vit_path,
        freeze_encoder=args.freeze_encoders,
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_mae = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            pred = model(batch["cxr"].to(device))
            loss = criterion(pred, batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss, val_mae_sum, val_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["cxr"].to(device))
                target = batch["target"].to(device)
                val_loss += criterion(pred, target).item()
                val_mae_sum += (pred - target).abs().sum().item()
                val_n += target.size(0)
        val_mae = val_mae_sum / val_n if val_n else 0
        val_rmse = (val_loss / len(val_loader)) ** 0.5 if val_loader else 0
        print(f"Epoch {epoch+1}/{args.epochs}  train_mse={train_loss:.4f}  val_mse={val_loss/len(val_loader):.4f}  val_mae={val_mae:.4f}  val_rmse={val_rmse:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mae": val_mae}, out_dir / "best.pt")
        if (epoch + 1) % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / f"checkpoint_{epoch+1}.pt")

    print(f"Best val_mae: {best_val_mae:.4f}")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_mae_sum, test_n = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["cxr"].to(device))
            test_mae_sum += (pred - batch["target"].to(device)).abs().sum().item()
            test_n += batch["target"].size(0)
    test_mae = test_mae_sum / test_n if test_n else 0
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            test_loss += criterion(model(batch["cxr"].to(device)), batch["target"].to(device)).item()
    test_mse = test_loss / len(test_loader) if test_loader else 0
    test_rmse = test_mse ** 0.5
    print(f"\n=== Test set ===")
    print(f"  test_mae: {test_mae:.4f}  test_rmse: {test_rmse:.4f}")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"best_val_mae": best_val_mae, "test_mae": test_mae, "test_rmse": test_rmse, "target_col": args.target_col, "modality": "CXR-only"}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DATA_CSV)
    parser.add_argument("--cxr_root", default=CXR_ROOT)
    parser.add_argument("--metadata_path", default=METADATA_PATH)
    parser.add_argument("--vit_path", default=VIT_PATH)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--freeze_encoders", action="store_true", default=FREEZE_ENCODERS)
    parser.add_argument("--no_freeze", action="store_true")
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
