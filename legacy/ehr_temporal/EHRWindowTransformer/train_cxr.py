"""Train CXRWindowTransformer on [t-24h, t-12h] CXR images only; supervise anchor s2f/p2f change."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EHRTREND = PROJECT_ROOT / "EHRTrend"
_exp_old = PROJECT_ROOT / "experiment1(old)"
for _p in (PROJECT_ROOT, PROJECT_ROOT / "BaselineExperiment", _exp_old, EHRTREND):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import (  # noqa: E402
    BATCH_SIZE,
    CXR_DIM_DEFAULT,
    CXR_ROOT_DEFAULT,
    EPOCHS,
    LR,
    METADATA_PATH_DEFAULT,
    NEXTSTEP_D_MODEL,
    NEXTSTEP_DROPOUT,
    NEXTSTEP_EARLY_STOP_MIN_DELTA,
    NEXTSTEP_EARLY_STOP_PATIENCE,
    NEXTSTEP_ENRICHED_CSV,
    NEXTSTEP_NUM_HEADS,
    NEXTSTEP_NUM_TRANSFORMER_LAYERS,
    NUM_CLASSES,
    NUM_WORKERS,
    P2F_OR_S2F_CSV,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    VIT_PATH_DEFAULT,
    WEIGHT_DECAY,
)
from cxr_window_dataset import CXRWindowDataset  # noqa: E402

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX))
from common import (  # noqa: E402
    collate_cxr_window_batch,
    eval_loader,
    forward_loss_from_logits,
    stratify_labels_from_anchor,
)
from model import CXRWindowTransformer  # noqa: E402


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CXRWindowTransformer  device={device}")

    full_ds = CXRWindowDataset(
        anchor_source_csv=args.anchor_csv,
        history_csv=args.history_csv,
        label_lookup_csv=args.label_lookup_csv,
        enriched_csv=args.enriched_csv,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        cxr_split=args.cxr_split,
        imagenet_normalize=not args.no_imagenet_normalize,
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
        collate_fn=collate_cxr_window_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_window_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_window_batch,
    )

    model = CXRWindowTransformer(
        cxr_dim=args.cxr_dim,
        d_model=args.d_model,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
        vit_path=args.vit_path,
        freeze_cxr=not args.unfreeze_cxr,
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        for batch in train_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            log_s, log_p = model(b["cxr_seq"], b["cxr_mask"])
            loss = forward_loss_from_logits(b, log_s, log_p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss)
        tr /= max(len(train_loader), 1)
        st = eval_loader(model, val_loader, device, "cxr_seq", "cxr_mask")
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={tr:.4f}  val_loss={st['loss']:.4f}  "
            f"val_acc_s2f={st['acc_s2f']:.4f}  val_acc_p2f={st['acc_p2f']:.4f}"
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
    test_st = eval_loader(model, test_loader, device, "cxr_seq", "cxr_mask")
    print(
        f"Test: loss={test_st['loss']:.4f}  acc_s2f={test_st['acc_s2f']:.4f}  acc_p2f={test_st['acc_p2f']:.4f}"
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "task": "cxr_window_transformer_anchor_s2f_p2f",
                "lookback_hours": [args.lookback_max_hours, args.lookback_min_hours],
                "best_val_loss": best_val,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "stopped_early": stopped_early,
                "test": test_st,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--history_csv", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--label_lookup_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--enriched_csv", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--cxr_root", default=CXR_ROOT_DEFAULT)
    p.add_argument("--metadata_path", default=METADATA_PATH_DEFAULT)
    p.add_argument("--lookback_min_hours", type=int, default=12)
    p.add_argument("--lookback_max_hours", type=int, default=24)
    p.add_argument("--cxr_split", default="train")
    p.add_argument("--no_imagenet_normalize", action="store_true")
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--cxr_dim", type=int, default=CXR_DIM_DEFAULT)
    p.add_argument("--d_model", type=int, default=NEXTSTEP_D_MODEL)
    p.add_argument("--num_heads", type=int, default=NEXTSTEP_NUM_HEADS)
    p.add_argument("--num_layers", type=int, default=NEXTSTEP_NUM_TRANSFORMER_LAYERS)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=NEXTSTEP_DROPOUT)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--vit_path", default=VIT_PATH_DEFAULT)
    p.add_argument("--unfreeze_cxr", action="store_true")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "output_cxr_window"),
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=NEXTSTEP_EARLY_STOP_PATIENCE)
    p.add_argument("--early_stop_min_delta", type=float, default=NEXTSTEP_EARLY_STOP_MIN_DELTA)
    a = p.parse_args()
    if not a.max_samples:
        a.max_samples = 0
    main(a)
