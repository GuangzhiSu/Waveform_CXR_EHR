"""Train DirectWindowTransformer on [t-24h, t-12h] EHR only; supervise anchor s2f/p2f change vs ground truth."""
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
EHRTREND = PROJECT_ROOT / "EHRTrend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BaselineExperiment"))
sys.path.insert(0, str(EHRTREND))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LR,
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
    SCHEMA_CSV,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from ehr_nextstep_dataset import EHRNextStepDataset  # noqa: E402

_EX = Path(__file__).resolve().parent
# Must precede EHRTrend so ``import model`` resolves to this folder (EHRTrend also has model.py).
sys.path.insert(0, str(_EX))
from model import DirectWindowTransformer  # noqa: E402


def masked_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.new_tensor(0.0)
    y_m = y.clone()
    y_m[~valid] = -100
    return F.cross_entropy(logits, y_m, ignore_index=-100)


def collate_window_batch(batch):
    lengths = [b["ehr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    feat = batch[0]["ehr_seq"].shape[1]
    bsz = len(batch)
    seq = torch.zeros(bsz, max_len, feat, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    anchor_s2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_p2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_has_s2f = torch.zeros(bsz, dtype=torch.bool)
    anchor_has_p2f = torch.zeros(bsz, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["ehr_seq"].shape[0]
        seq[i, :t] = b["ehr_seq"]
        mask[i, :t] = True
        c = b["anchor_s2f_cls"]
        anchor_s2f[i] = c if c >= 0 else -1
        c2 = b["anchor_p2f_cls"]
        anchor_p2f[i] = c2 if c2 >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])
    return {
        "ehr_seq": seq,
        "ehr_mask": mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
    }


def _stratify_labels_from_dataset(ds: EHRNextStepDataset) -> np.ndarray:
    n = len(ds)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ds.anchor_has_p2f[i] and ds.anchor_p2f_cls[i] >= 0:
            y[i] = int(ds.anchor_p2f_cls[i])
        elif ds.anchor_has_s2f[i] and ds.anchor_s2f_cls[i] >= 0:
            y[i] = 3 + int(ds.anchor_s2f_cls[i])
        else:
            y[i] = 0
    return y


def forward_loss(model: DirectWindowTransformer, batch: dict) -> torch.Tensor:
    log_s, log_p = model(batch["ehr_seq"], batch["ehr_mask"])
    device = log_s.device
    s_tgt = batch["anchor_s2f"].to(device)
    p_tgt = batch["anchor_p2f"].to(device)
    s_ok = batch["anchor_has_s2f"].to(device) & (s_tgt >= 0)
    p_ok = batch["anchor_has_p2f"].to(device) & (p_tgt >= 0)
    return masked_ce(log_s, s_tgt, s_ok) + masked_ce(log_p, p_tgt, p_ok)


@torch.no_grad()
def eval_loader(model, loader, device) -> dict:
    model.eval()
    tot = 0.0
    n_batches = 0
    acc_s_n = acc_s_d = acc_p_n = acc_p_d = 0.0
    ce_s_sum = ce_p_sum = 0.0
    n_ce_s = n_ce_p = 0
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b["ehr_seq"], b["ehr_mask"])
        ls = masked_ce(log_s, b["anchor_s2f"], b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0))
        lp = masked_ce(log_p, b["anchor_p2f"], b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0))
        tot += float(ls + lp)
        n_batches += 1
        m = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        if m.any():
            ce_s_sum += float(F.cross_entropy(log_s[m], b["anchor_s2f"][m]))
            n_ce_s += 1
            acc_s_n += (log_s[m].argmax(1) == b["anchor_s2f"][m]).float().sum().item()
            acc_s_d += int(m.sum())
        m = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        if m.any():
            ce_p_sum += float(F.cross_entropy(log_p[m], b["anchor_p2f"][m]))
            n_ce_p += 1
            acc_p_n += (log_p[m].argmax(1) == b["anchor_p2f"][m]).float().sum().item()
            acc_p_d += int(m.sum())
    return {
        "loss": tot / max(n_batches, 1),
        "ce_s2f": ce_s_sum / max(n_ce_s, 1),
        "ce_p2f": ce_p_sum / max(n_ce_p, 1),
        "acc_s2f": acc_s_n / max(acc_s_d, 1),
        "acc_p2f": acc_p_n / max(acc_p_d, 1),
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EHRWindowTransformer (direct)  device={device}")

    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  No enriched join (missing file): {enr!r}")
        enr = None

    full_ds = EHRNextStepDataset(
        anchor_source_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        enriched_csv=enr,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
    )
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        rng = np.random.RandomState(args.seed)
        idxs = rng.choice(n_all, size=args.max_samples, replace=False)
        full_ds = Subset(full_ds, idxs.tolist())
        print(f"  Subset max_samples={args.max_samples}")

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    input_dim = base.input_dim
    y = _stratify_labels_from_dataset(base)
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
        collate_fn=collate_window_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_window_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_window_batch,
    )

    model = DirectWindowTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            loss = forward_loss(model, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss)
        tr /= max(len(train_loader), 1)
        st = eval_loader(model, val_loader, device)
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
                {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val, "input_dim": input_dim},
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
    test_st = eval_loader(model, test_loader, device)
    print(
        f"Test: loss={test_st['loss']:.4f}  acc_s2f={test_st['acc_s2f']:.4f}  acc_p2f={test_st['acc_p2f']:.4f}"
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "task": "ehr_window_direct_transformer_anchor_s2f_p2f",
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
    p.add_argument("--history_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--enriched_csv", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true")
    p.add_argument("--lookback_min_hours", type=int, default=12)
    p.add_argument("--lookback_max_hours", type=int, default=24)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--d_model", type=int, default=NEXTSTEP_D_MODEL)
    p.add_argument("--num_heads", type=int, default=NEXTSTEP_NUM_HEADS)
    p.add_argument("--num_layers", type=int, default=NEXTSTEP_NUM_TRANSFORMER_LAYERS)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=NEXTSTEP_DROPOUT)
    p.add_argument("--max_seq_len", type=int, default=8192)
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
        default=str(Path(__file__).resolve().parent / "output_direct_window"),
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=NEXTSTEP_EARLY_STOP_PATIENCE)
    p.add_argument("--early_stop_min_delta", type=float, default=NEXTSTEP_EARLY_STOP_MIN_DELTA)
    a = p.parse_args()
    if not a.max_samples:
        a.max_samples = 0
    main(a)
