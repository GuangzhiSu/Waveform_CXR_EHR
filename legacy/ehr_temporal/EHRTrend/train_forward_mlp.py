"""Train row-level EHR encoder + MLP heads to predict forward [t+12h,t+24h] severity changes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BaselineExperiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classification_utils import make_subset, stratified_train_val_test_indices
from config import (
    BATCH_SIZE,
    EPOCHS,
    FORWARD_EARLY_STOP_MIN_DELTA,
    FORWARD_EARLY_STOP_PATIENCE,
    FORWARD_MLP_OUTPUT_DIR,
    LR,
    NUM_CLASSES,
    P2F_OR_S2F_CSV,
    NEXTSTEP_ENRICHED_CSV,
    SCHEMA_CSV,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from forward_mlp_dataset import EHRForwardChangeDataset
from forward_mlp_model import ForwardChangeRowModel


def masked_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only on rows where ``valid``; 0 if none (avoids NaN when all ignored)."""
    if not valid.any():
        return logits.new_tensor(0.0)
    y_m = y.clone()
    y_m[~valid] = -100
    return F.cross_entropy(logits, y_m, ignore_index=-100)


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "s2f_y": torch.tensor([b["s2f_y"] for b in batch], dtype=torch.long),
        "p2f_y": torch.tensor([b["p2f_y"] for b in batch], dtype=torch.long),
        "s2f_valid": torch.tensor([b["s2f_valid"] for b in batch], dtype=torch.bool),
        "p2f_valid": torch.tensor([b["p2f_valid"] for b in batch], dtype=torch.bool),
    }


def stratify_labels(s_forward: np.ndarray, p_forward: np.ndarray) -> np.ndarray:
    ys = np.where(s_forward >= 0, s_forward, 5)
    yp = np.where(p_forward >= 0, p_forward, 5)
    return ys * 6 + yp


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Forward-change row MLP. Device: {device}")

    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  No enriched join (missing file): {enr!r}")
        enr = None

    full_ds = EHRForwardChangeDataset(
        source_csv=args.source_csv,
        schema_csv=args.schema_csv,
        enriched_csv=enr,
        forward_min_hours=args.forward_min_hours,
        forward_max_hours=args.forward_max_hours,
    )
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        idxs = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, idxs.tolist())
        print(f"  Subset max_samples={args.max_samples}")

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    s_lab = np.asarray(base.s_forward)
    p_lab = np.asarray(base.p_forward)
    if isinstance(full_ds, Subset):
        idx = np.array(full_ds.indices, dtype=np.int64)
        s_lab = s_lab[idx]
        p_lab = p_lab[idx]

    y_strat = stratify_labels(s_lab, p_lab)
    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y_strat, args.train_split, args.val_split, test_split, args.seed
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
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = ForwardChangeRowModel(
        input_dim=base.input_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0
    stopped_early = False
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        for batch in train_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ls, lp = model(b["x"])
            loss = masked_ce(ls, b["s2f_y"], b["s2f_valid"]) + masked_ce(lp, b["p2f_y"], b["p2f_valid"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss)
        tr /= max(len(train_loader), 1)

        model.eval()
        vs, vp, ns, np_ = 0.0, 0.0, 0, 0
        for batch in val_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ls, lp = model(b["x"])
            vs += float(masked_ce(ls, b["s2f_y"], b["s2f_valid"]))
            vp += float(masked_ce(lp, b["p2f_y"], b["p2f_valid"]))
            ns += 1
            np_ += 1
        vs /= max(ns, 1)
        vp /= max(np_, 1)
        vtot = vs + vp

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={tr:.4f}  val_ce_s2f={vs:.4f}  val_ce_p2f={vp:.4f}")

        improved = vtot < best_val - args.early_stop_min_delta
        if improved:
            best_val = vtot
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")
        else:
            epochs_no_improve += 1

        if (
            args.early_stop_patience > 0
            and epochs_no_improve >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}/{args.epochs} "
                f"(no improvement on val_ce_s2f+val_ce_p2f for {args.early_stop_patience} epochs; "
                f"best epoch {best_epoch + 1}, best sum={best_val:.4f})"
            )
            stopped_early = True
            break

    torch.save(model.state_dict(), out_dir / "last.pt")
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    def run_eval(loader, name):
        tot_s, tot_p, n = 0.0, 0.0, 0
        acc_s_n, acc_s_d, acc_p_n, acc_p_d = 0.0, 0, 0.0, 0
        for batch in loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ls, lp = model(b["x"])
            tot_s += float(masked_ce(ls, b["s2f_y"], b["s2f_valid"]))
            tot_p += float(masked_ce(lp, b["p2f_y"], b["p2f_valid"]))
            n += 1
            m = b["s2f_valid"]
            if m.any():
                acc_s_n += (ls[m].argmax(1) == b["s2f_y"][m].to(device)).float().sum().item()
                acc_s_d += int(m.sum())
            m = b["p2f_valid"]
            if m.any():
                acc_p_n += (lp[m].argmax(1) == b["p2f_y"][m].to(device)).float().sum().item()
                acc_p_d += int(m.sum())
        tot_s /= max(n, 1)
        tot_p /= max(n, 1)
        print(
            f"{name}: ce_s2f={tot_s:.4f} ce_p2f={tot_p:.4f} "
            f"acc_s2f={acc_s_n/max(acc_s_d,1):.4f} acc_p2f={acc_p_n/max(acc_p_d,1):.4f}"
        )
        return {
            "ce_s2f": tot_s,
            "ce_p2f": tot_p,
            "acc_s2f": acc_s_n / max(acc_s_d, 1),
            "acc_p2f": acc_p_n / max(acc_p_d, 1),
        }

    st = run_eval(test_loader, "Test")
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_ce_sum": best_val,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "stopped_early": stopped_early,
                "early_stop_patience": args.early_stop_patience,
                "test": st,
                "task": "forward_change_row_mlp",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--enriched_csv", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true")
    p.add_argument("--forward_min_hours", type=int, default=12)
    p.add_argument("--forward_max_hours", type=int, default=24)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", default=FORWARD_MLP_OUTPUT_DIR)
    p.add_argument("--max_samples", type=int, default=0, help="If >0, random subset for dev")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=FORWARD_EARLY_STOP_PATIENCE,
        help="Stop if val_ce_s2f+val_ce_p2f does not improve for this many epochs; 0 disables",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=FORWARD_EARLY_STOP_MIN_DELTA,
        help="Minimum decrease in val sum to count as improvement",
    )
    a = p.parse_args()
    if a.max_samples is None:
        a.max_samples = 0
    main(a)
