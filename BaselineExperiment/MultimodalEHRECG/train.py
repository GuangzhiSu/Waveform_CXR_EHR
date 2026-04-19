"""Train Multimodal EHR+ECG: CLIP alignment loss + ARDS classification loss."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_utils import compute_class_weights, make_subset, stratified_train_val_test_indices

from MultimodalEHRECG.config import *
from MultimodalEHRECG.dataset import MultimodalEHRECGDataset
from MultimodalEHRECG.model import MultimodalEHRECGModel, clip_infonce_loss


def collate_fn(batch):
    max_e = max(b["ehr_seq"].shape[0] for b in batch)
    feat = batch[0]["ehr_seq"].shape[1]
    max_s = max(b["signal_seq"].shape[0] for b in batch)
    bsz = len(batch)

    ehr_seq = torch.zeros(bsz, max_e, feat, dtype=torch.float32)
    ehr_mask = torch.zeros(bsz, max_e, dtype=torch.bool)
    signal_seq = torch.zeros(bsz, max_s, 12, 1000, dtype=torch.float32)
    signal_mask = torch.zeros(bsz, max_s, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        te = b["ehr_seq"].shape[0]
        ehr_seq[i, :te] = b["ehr_seq"]
        ehr_mask[i, :te] = True
        ts = b["signal_seq"].shape[0]
        signal_seq[i, :ts] = b["signal_seq"]
        signal_mask[i, :ts] = True

    return {
        "ehr_seq": ehr_seq,
        "ehr_mask": ehr_mask,
        "signal_seq": signal_seq,
        "signal_mask": signal_mask,
        "label": labels,
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multimodal EHR+ECG (contrastive + classification). Device: {device}")

    full_ds = MultimodalEHRECGDataset(
        anchor_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        ecg_pool_csv=args.ecg_pool_csv,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
    )
    input_dim = full_ds.input_dim
    y = full_ds.labels
    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )

    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    print(
        f"Split (stratified): train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)} "
        f"(aligned anchors only)"
    )
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

    model = MultimodalEHRECGModel(
        input_dim=input_dim,
        num_classes=args.num_classes,
        ehr_embed_dim=args.ehr_embed_dim,
        ecg_hidden_dim=args.ecg_hidden_dim,
        contrast_dim=args.contrast_dim,
        fusion_hidden=args.fusion_hidden,
        pooling_stats=tuple(args.pooling_stats),
        ehr_encoder_kind=args.ehr_encoder,
        ecg_encoder_kind=args.ecg_encoder,
        ecg_ckpt_path=args.ecg_ckpt or None,
        freeze_ecg_encoder=not args.no_freeze_ecg,
        logit_scale_init=args.logit_scale_init,
    ).to(device)

    class_weights = compute_class_weights(y[idx_train], args.num_classes, device)
    criterion_task = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()
    print(f"  class weights (train CE): {class_weights.cpu().numpy().round(4).tolist()}")
    print(f"  lambda_contrast={args.lambda_contrast}, lambda_task={args.lambda_task}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = train_task_sum = train_con_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            ehr_seq = batch["ehr_seq"].to(device)
            ehr_mask = batch["ehr_mask"].to(device)
            signal_seq = batch["signal_seq"].to(device)
            signal_mask = batch["signal_mask"].to(device)
            target = batch["label"].to(device)

            logits, z_e, z_c, logit_scale = model(ehr_seq, ehr_mask, signal_seq, signal_mask)
            loss_task = criterion_task(logits, target)
            loss_con = clip_infonce_loss(z_e, z_c, logit_scale)
            loss = args.lambda_task * loss_task + args.lambda_contrast * loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_task_sum += loss_task.item()
            train_con_sum += loss_con.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_task_sum /= max(n_batches, 1)
        train_con_sum /= max(n_batches, 1)

        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ehr_seq = batch["ehr_seq"].to(device)
                ehr_mask = batch["ehr_mask"].to(device)
                signal_seq = batch["signal_seq"].to(device)
                signal_mask = batch["signal_mask"].to(device)
                target = batch["label"].to(device)
                logits, z_e, z_c, logit_scale = model(ehr_seq, ehr_mask, signal_seq, signal_mask)
                loss_task = criterion_eval(logits, target)
                loss_con = clip_infonce_loss(z_e, z_c, logit_scale)
                loss = args.lambda_task * loss_task + args.lambda_contrast * loss_con
                val_loss_sum += loss.item()
                val_correct += (logits.argmax(1) == target).sum().item()
                val_total += target.size(0)
        val_acc = val_correct / val_total if val_total else 0
        val_loss = val_loss_sum / len(val_loader) if val_loader else 0

        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} "
            f"(task={train_task_sum:.4f} clip={train_con_sum:.4f})  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
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
            ehr_seq = batch["ehr_seq"].to(device)
            ehr_mask = batch["ehr_mask"].to(device)
            signal_seq = batch["signal_seq"].to(device)
            signal_mask = batch["signal_mask"].to(device)
            target = batch["label"].to(device)
            logits, _, _, _ = model(ehr_seq, ehr_mask, signal_seq, signal_mask)
            test_correct += (logits.argmax(1) == target).sum().item()
            test_total += target.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
    test_acc = test_correct / test_total if test_total else 0

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ["Severe", "Moderate", "Mild"]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== Test set ===")
    print(f"  test_acc: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(cm)

    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
        "modality": "EHR+ECG",
        "task": "ARDS_severity_classification",
        "loss": "lambda_task * CE + lambda_contrast * CLIP_InfoNCE",
        "lambda_task": args.lambda_task,
        "lambda_contrast": args.lambda_contrast,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_csv", default=ANCHOR_CSV)
    p.add_argument("--history_csv", default=HISTORY_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--ecg_pool_csv", default=ECG_POOL_CSV)
    p.add_argument("--ecg_ckpt", default=ECG_CKPT)
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--ehr_embed_dim", type=int, default=EHR_EMBED_DIM)
    p.add_argument("--ecg_hidden_dim", type=int, default=ECG_HIDDEN_DIM)
    p.add_argument("--contrast_dim", type=int, default=CONTRAST_DIM)
    p.add_argument("--fusion_hidden", type=int, default=FUSION_HIDDEN)
    p.add_argument(
        "--ehr_encoder",
        type=str,
        default="mlp",
        choices=["mlp", "transformer", "contrastive"],
        help="EHR encoder implementation from models/encoders/ehr.py",
    )
    p.add_argument(
        "--ecg_encoder",
        type=str,
        default="cnn",
        choices=["cnn", "transformer"],
        help="ECG encoder implementation from models/encoders/ecg.py",
    )
    p.add_argument("--pooling_stats", nargs="+", default=list(POOLING_STATS))
    p.add_argument(
        "--no_freeze_ecg",
        action="store_true",
        help="Train ECG SignalEncoder weights (default: frozen MedTVT checkpoint).",
    )
    p.add_argument("--logit_scale_init", type=float, default=LOGIT_SCALE_INIT)
    p.add_argument("--lambda_contrast", type=float, default=LAMBDA_CONTRAST)
    p.add_argument("--lambda_task", type=float, default=LAMBDA_TASK)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "output"))
    args = p.parse_args()
    main(args)
