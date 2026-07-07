"""Train the ECG-CXR patient-temporal contrastive baseline.

  q_{t1->t2} = f(CXR_t1, ECG_interval)  pulled toward the true CXR_t2 embedding.
  loss = w_cross * cross_patient_loss + w_temporal * temporal_loss

Ablation via --loss_mode {cross, temporal, combined}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from dataset import PatientTemporalData, PatientTemporalDataset, collate_fn  # noqa: E402
from losses import total_loss  # noqa: E402
from metrics import evaluate_retrieval  # noqa: E402
from model import PatientTemporalModel  # noqa: E402
from runtime import get_device, set_seed  # noqa: E402
from sampler import NPatientsKIntervalsSampler  # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=C.PAIRS_JSON)
    ap.add_argument("--cxr_emb", default=C.CXR_EMB_NPY)
    ap.add_argument("--cxr_ids", default=C.CXR_IDS_JSON)
    ap.add_argument("--ecg_emb", default=C.ECG_EMB_NPY)
    ap.add_argument("--ecg_ids", default=C.ECG_IDS_JSON)
    ap.add_argument("--output_dir", default=C.OUTPUT_DIR)
    ap.add_argument("--loss_mode", default=C.LOSS_MODE, choices=["cross", "temporal", "combined"])
    ap.add_argument("--lambda_temporal", type=float, default=C.LAMBDA_TEMPORAL)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--learnable_temperature", action="store_true",
                    default=C.LEARNABLE_TEMPERATURE)
    ap.add_argument("--proj_dim", type=int, default=C.PROJ_DIM)
    ap.add_argument("--d_model", type=int, default=C.D_MODEL)
    ap.add_argument("--ecg_tx_layers", type=int, default=C.ECG_TX_LAYERS)
    ap.add_argument("--ecg_pool", default=C.ECG_POOL, choices=["mean", "cls"])
    ap.add_argument("--n_patients", type=int, default=C.N_PATIENTS)
    ap.add_argument("--k_intervals", type=int, default=C.K_INTERVALS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--steps_per_epoch", type=int, default=C.STEPS_PER_EPOCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--max_grad_norm", type=float, default=C.MAX_GRAD_NORM)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--early_stop_patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None, help="Optional run name (subdir of output_dir).")
    return ap.parse_args()


def run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm):
    model.train()
    agg = {"loss": 0.0, "cross_patient_loss": 0.0, "temporal_loss": 0.0,
           "n_temporal_rows": 0, "steps": 0, "skipped": 0}
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        loss, logs = total_loss(out["logits"], b["patient_id"], w_cross, w_temporal,
                                c2_rows=b.get("c2_row"))
        if not torch.isfinite(loss):
            agg["skipped"] += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()
        for k in ("loss", "cross_patient_loss", "temporal_loss"):
            agg[k] += logs[k]
        agg["n_temporal_rows"] += logs["n_temporal_rows"]
        agg["steps"] += 1
    s = max(agg["steps"], 1)
    return {"loss": agg["loss"] / s, "cross_patient_loss": agg["cross_patient_loss"] / s,
            "temporal_loss": agg["temporal_loss"] / s, "steps": agg["steps"],
            "skipped": agg["skipped"], "avg_temporal_rows": agg["n_temporal_rows"] / s}


def monitor_value(eval_res: dict, loss_mode: str) -> float:
    if not eval_res:
        return float("nan")
    if loss_mode == "temporal":
        v = eval_res.get("temporal", {}).get("temporal_recall@1", float("nan"))
        return v if v == v else 0.0  # nan-safe
    return eval_res.get("cross_patient", {}).get("recall@1", 0.0)


def main():
    args = build_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.output_dir) / args.tag if args.tag else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Patient-temporal contrastive training (loss_mode={args.loss_mode}, device={device}) ===")
    w_cross, w_temporal = C.loss_weights(args.loss_mode, args.lambda_temporal)
    print(f"  loss weights: w_cross={w_cross}  w_temporal={w_temporal}")

    data = PatientTemporalData(
        pairs_json=args.pairs, cxr_emb_npy=args.cxr_emb, cxr_ids_json=args.cxr_ids,
        ecg_emb_npy=args.ecg_emb, ecg_ids_json=args.ecg_ids, seed=args.seed,
        train_split=C.TRAIN_SPLIT, val_split=C.VAL_SPLIT, test_split=C.TEST_SPLIT,
    )
    train_ds = PatientTemporalDataset(data, data.split_indices["train"])
    val_ds = PatientTemporalDataset(data, data.split_indices["val"])
    test_ds = PatientTemporalDataset(data, data.split_indices["test"])

    sampler = NPatientsKIntervalsSampler(
        train_ds.patient_ids(), args.n_patients, args.k_intervals,
        num_batches=args.steps_per_epoch, seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    cxr_dim = data.cxr_emb.shape[1]
    ecg_dim = data.ecg_emb.shape[1]
    model = PatientTemporalModel(
        cxr_dim=cxr_dim, ecg_dim=ecg_dim, proj_dim=args.proj_dim,
        cxr_proj_hidden=C.CXR_PROJ_HIDDEN, d_model=args.d_model,
        ecg_tx_layers=args.ecg_tx_layers, ecg_tx_heads=C.ECG_TX_HEADS,
        ecg_tx_mlp_ratio=C.ECG_TX_MLP_RATIO, fusion_hidden=C.FUSION_HIDDEN,
        dropout=C.DROPOUT, ecg_pool=args.ecg_pool, temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}  (cxr_dim={cxr_dim}, ecg_dim={ecg_dim})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_monitor = -1.0
    best_epoch = -1
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        tr = run_epoch(model, train_loader, optimizer, device, w_cross, w_temporal, args.max_grad_norm)
        val_res = evaluate_retrieval(model, val_ds, data.cxr_emb, device, args.eval_batch_size)
        mon = monitor_value(val_res, args.loss_mode)
        history.append({"epoch": epoch, "train": tr, "val": val_res, "monitor": mon,
                        "temperature": model.temperature_value()})
        vc = val_res.get("cross_patient", {})
        vt = val_res.get("temporal", {})
        print(f"  [E{epoch:03d}] train_loss={tr['loss']:.4f} "
              f"(cross={tr['cross_patient_loss']:.4f} temp={tr['temporal_loss']:.4f}) "
              f"| val R@1={vc.get('recall@1', float('nan')):.4f} R@5={vc.get('recall@5', float('nan')):.4f} "
              f"MRR={vc.get('mrr', float('nan')):.4f} | val T-R@1={vt.get('temporal_recall@1', float('nan')):.4f} "
              f"T-MRR={vt.get('temporal_mrr', float('nan')):.4f} | temp={model.temperature_value():.4f}"
              + (f"  [skipped {tr['skipped']}]" if tr["skipped"] else ""))

        if mon > best_monitor + C.EARLY_STOP_MIN_DELTA:
            best_monitor = mon
            best_epoch = epoch
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, monitor={best_monitor:.4f})")
                break

    # Final test evaluation with best checkpoint.
    best_path = out_dir / "best.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_res = evaluate_retrieval(model, test_ds, data.cxr_emb, device, args.eval_batch_size)
    print("=== TEST ===")
    print(json.dumps(test_res, indent=2))

    results = {
        "loss_mode": args.loss_mode, "w_cross": w_cross, "w_temporal": w_temporal,
        "best_epoch": best_epoch, "best_val_monitor": best_monitor,
        "test": test_res, "history": history, "args": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
