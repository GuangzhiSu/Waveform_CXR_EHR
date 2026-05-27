"""Train frozen-style multimodal discriminative MLP (3 encoders -> concat -> StepDisc heads)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_exp_old = PROJECT_ROOT / "experiment1(old)"
for _p in (PROJECT_ROOT, PROJECT_ROOT / "BaselineExperiment", _exp_old, Path(__file__).resolve().parent):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices
from config import (
    CXR_DIM_DEFAULT,
    CXR_ROOT_DEFAULT,
    ECG_CKPT_DEFAULT,
    ECG_DIM_DEFAULT,
    ECG_TARGET_LEN_DEFAULT,
    EMBED_DIM,
    EPOCHS,
    FORWARD_EARLY_STOP_MIN_DELTA,
    FORWARD_EARLY_STOP_PATIENCE,
    FUSE_DIM_DEFAULT,
    LR,
    METADATA_PATH_DEFAULT,
    MM_FORWARD_MLP_OUTPUT_DIR,
    MULTIMODAL_HISTORY_CSV,
    NEXTSTEP_DROPOUT,
    NEXTSTEP_ENRICHED_CSV,
    NUM_CLASSES,
    NUM_WORKERS,
    P2F_OR_S2F_CSV,
    SCHEMA_CSV,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    VIT_PATH_DEFAULT,
    WEIGHT_DECAY,
)
from multimodal_forward_mlp_dataset import MultimodalForwardMLPDataset
from multimodal_forward_mlp_model import MultimodalForwardMLPModel

_AGENT_DEBUG_LOG = Path("/work/gs285/Waveform_CXR_EHR/.cursor/debug-bb9b63.log")


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "train") -> None:
    payload = {
        "sessionId": "bb9b63",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _AGENT_DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def masked_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        # Preserve autograd graph: new_tensor(0) is detached and breaks loss.backward() when both task masks are empty.
        return logits.sum() * 0.0
    y_m = y.clone()
    y_m[~valid] = -100
    return F.cross_entropy(logits, y_m, ignore_index=-100)


def collate_mm(batch):
    return {
        "ehr": torch.stack([b["ehr"] for b in batch]),
        "cxr": torch.stack([b["cxr"] for b in batch]),
        "ecg": torch.stack([b["ecg"] for b in batch]),
        "cxr_valid": torch.tensor([b["cxr_valid"] for b in batch], dtype=torch.bool),
        "ecg_valid": torch.tensor([b["ecg_valid"] for b in batch], dtype=torch.bool),
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
    print(f"Multimodal forward-change MLP. Device: {device}")

    enr = None if args.no_enriched else args.enriched_csv_for_group
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  enriched_csv_for_group missing: {enr!r} — using None")
        enr = None

    full_ds = MultimodalForwardMLPDataset(
        anchor_csv=args.anchor_csv,
        enriched_csv=args.history_csv,
        schema_csv=args.schema_csv,
        enriched_csv_for_group=enr,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path or None,
        forward_min_hours=args.forward_min_hours,
        forward_max_hours=args.forward_max_hours,
        ecg_target_len=args.ecg_target_len,
        cxr_split=args.cxr_split,
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

    ecg_ckpt = args.ecg_ckpt if args.ecg_ckpt and Path(args.ecg_ckpt).is_file() else None
    model = MultimodalForwardMLPModel(
        input_dim=base.input_dim,
        ehr_embed_dim=args.embed_dim,
        cxr_dim=args.cxr_dim,
        ecg_dim=args.ecg_dim,
        fuse_dim=args.fuse_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        vit_path=args.vit_path,
        freeze_cxr=args.freeze_cxr,
        ecg_ckpt_path=ecg_ckpt,
        freeze_ecg=args.freeze_ecg,
        ecg_sig_len=args.ecg_target_len,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mm,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_mm)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_mm)

    best_val = float("inf")
    epochs_no_improve = 0
    stopped_early = False
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        for batch in train_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ls, lp = model(b["ehr"], b["cxr"], b["ecg"], b["cxr_valid"], b["ecg_valid"])
            l_s = masked_ce(ls, b["s2f_y"], b["s2f_valid"])
            l_p = masked_ce(lp, b["p2f_y"], b["p2f_valid"])
            loss = l_s + l_p
            # #region agent log
            if not loss.requires_grad:
                _agent_debug_log(
                    "D",
                    "train_multimodal_forward_mlp.py:train_loop",
                    "loss has no grad before backward",
                    {
                        "epoch": epoch,
                        "s2f_valid_any": bool(b["s2f_valid"].any().item()),
                        "p2f_valid_any": bool(b["p2f_valid"].any().item()),
                        "l_s_requires_grad": bool(l_s.requires_grad),
                        "l_p_requires_grad": bool(l_p.requires_grad),
                        "ls_requires_grad": bool(ls.requires_grad),
                        "lp_requires_grad": bool(lp.requires_grad),
                    },
                    run_id="post-fix",
                )
            # #endregion
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss)
        tr /= max(len(train_loader), 1)

        model.eval()
        vs, vp, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ls, lp = model(b["ehr"], b["cxr"], b["ecg"], b["cxr_valid"], b["ecg_valid"])
                vs += float(masked_ce(ls, b["s2f_y"], b["s2f_valid"]))
                vp += float(masked_ce(lp, b["p2f_y"], b["p2f_valid"]))
                n += 1
        vs /= max(n, 1)
        vp /= max(n, 1)
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

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (best sum={best_val:.4f}, best_epoch={best_epoch+1})")
            stopped_early = True
            break

    torch.save(model.state_dict(), out_dir / "last.pt")
    if (out_dir / "best.pt").is_file():
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()

    def run_eval(loader, name):
        tot_s, tot_p, n = 0.0, 0.0, 0
        acc_s_n, acc_s_d, acc_p_n, acc_p_d = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ls, lp = model(b["ehr"], b["cxr"], b["ecg"], b["cxr_valid"], b["ecg_valid"])
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
                "early_stop_min_delta": args.early_stop_min_delta,
                "test": st,
                "task": "multimodal_forward_change_mlp",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--history_csv", default=MULTIMODAL_HISTORY_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--enriched_csv_for_group", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true", help="Do not use enriched for subject_id group_id")
    p.add_argument("--cxr_root", default=CXR_ROOT_DEFAULT)
    p.add_argument("--metadata_path", default=METADATA_PATH_DEFAULT)
    p.add_argument("--ecg_ckpt", default=ECG_CKPT_DEFAULT)
    p.add_argument("--vit_path", default=VIT_PATH_DEFAULT)
    p.add_argument("--forward_min_hours", type=int, default=12)
    p.add_argument("--forward_max_hours", type=int, default=24)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--cxr_dim", type=int, default=CXR_DIM_DEFAULT)
    p.add_argument("--ecg_dim", type=int, default=ECG_DIM_DEFAULT)
    p.add_argument("--fuse_dim", type=int, default=FUSE_DIM_DEFAULT)
    p.add_argument("--dropout", type=float, default=NEXTSTEP_DROPOUT)
    p.add_argument("--ecg_target_len", type=int, default=ECG_TARGET_LEN_DEFAULT)
    p.add_argument("--cxr_split", type=str, default="train")
    p.add_argument("--freeze_cxr", action="store_true", default=True)
    p.add_argument("--no_freeze_cxr", action="store_false", dest="freeze_cxr")
    p.add_argument("--freeze_ecg", action="store_true", default=True)
    p.add_argument("--no_freeze_ecg", action="store_false", dest="freeze_ecg")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers (0 is safest on NFS)")
    p.add_argument("--output_dir", default=MM_FORWARD_MLP_OUTPUT_DIR)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=FORWARD_EARLY_STOP_PATIENCE)
    p.add_argument("--early_stop_min_delta", type=float, default=FORWARD_EARLY_STOP_MIN_DELTA)
    a = p.parse_args()
    if a.no_enriched:
        a.enriched_csv_for_group = ""
    if not a.max_samples:
        a.max_samples = 0
    main(a)
