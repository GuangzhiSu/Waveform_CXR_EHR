"""Generic train + eval loop shared by all staged experiments.

Given an :class:`~experiments.ExperimentSpec` and a loaded :class:`StagedData`,
``fit`` trains a :class:`StagedModel` with the N-patients x K-intervals sampler
and the two contrastive losses, doing patient-split retrieval evaluation each
epoch with early stopping, then a final test evaluation with the best weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from losses import total_loss
from metrics import evaluate_retrieval
from sampler import NPatientsKIntervalsSampler
from staged_dataset import StagedData, StagedDataset, collate_fn
from staged_model import StagedModel


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm):
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
            [p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        for k in ("loss", "cross_patient_loss", "temporal_loss"):
            agg[k] += logs[k]
        agg["n_temporal_rows"] += logs["n_temporal_rows"]
        agg["steps"] += 1
    s = max(agg["steps"], 1)
    return {"loss": agg["loss"] / s, "cross_patient_loss": agg["cross_patient_loss"] / s,
            "temporal_loss": agg["temporal_loss"] / s, "steps": agg["steps"],
            "skipped": agg["skipped"], "avg_temporal_rows": agg["n_temporal_rows"] / s}


def _monitor(eval_res: dict, loss_mode: str) -> float:
    if not eval_res:
        return float("nan")
    if loss_mode == "temporal":
        v = eval_res.get("temporal", {}).get("temporal_recall@1", float("nan"))
        return v if v == v else 0.0
    return eval_res.get("cross_patient", {}).get("recall@1", 0.0)


def _pairs_file(spec, args) -> str:
    return {
        "single": args.single_pairs,
        "seq_target": args.seq_target_pairs,
        "seq_t1": args.pairs,
    }[spec.pairs_kind]


def load_staged_data(spec, args) -> StagedData:
    return StagedData(
        pairs_json=_pairs_file(spec, args), kind=spec.data_kind(),
        cxr_emb_npy=args.cxr_emb, cxr_ids_json=args.cxr_ids,
        ecg_emb_npy=args.ecg_emb, ecg_ids_json=args.ecg_ids, seed=args.seed,
        train_split=C.TRAIN_SPLIT, val_split=C.VAL_SPLIT, test_split=C.TEST_SPLIT,
    )


def fit(spec, args, data: StagedData | None = None, device=None, verbose: bool = True) -> dict:
    """Train one experiment; return a results dict (test metrics + history)."""
    set_seed(args.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data is None:
        data = load_staged_data(spec, args)

    perturb = spec.ecg_perturb
    train_ds = StagedDataset(data, data.split_indices["train"], ecg_perturb=perturb, seed=args.seed)
    val_ds = StagedDataset(data, data.split_indices["val"], ecg_perturb=perturb, seed=args.seed + 1)
    test_ds = StagedDataset(data, data.split_indices["test"], ecg_perturb=perturb, seed=args.seed + 2)

    sampler = NPatientsKIntervalsSampler(
        train_ds.patient_ids(), args.n_patients, args.k_intervals,
        num_batches=args.steps_per_epoch, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    model = StagedModel(
        spec, cxr_dim=data.cxr_emb.shape[1], ecg_dim=data.ecg_emb.shape[1],
        proj_dim=args.proj_dim, cxr_proj_hidden=C.CXR_PROJ_HIDDEN, d_model=args.d_model,
        ecg_tx_layers=args.ecg_tx_layers, ecg_tx_heads=C.ECG_TX_HEADS,
        ecg_tx_mlp_ratio=C.ECG_TX_MLP_RATIO, fusion_hidden=C.FUSION_HIDDEN,
        time_emb_dim=C.TIME_EMB_DIM, dropout=C.DROPOUT, temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    w_cross, w_temporal = C.loss_weights(spec.loss_mode, spec.lambda_temporal)
    if verbose:
        print(f"  [{spec.name}] kind={spec.pairs_kind} trainable={n_train:,} "
              f"w_cross={w_cross} w_temporal={w_temporal} perturb={perturb}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_monitor, best_epoch, patience = -1.0, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        tr = _run_epoch(model, train_loader, optimizer, device, w_cross, w_temporal,
                        args.max_grad_norm)
        val_res = evaluate_retrieval(model, val_ds, data.cxr_emb, device,
                                     args.eval_batch_size, collate_fn=collate_fn)
        mon = _monitor(val_res, spec.loss_mode)
        history.append({"epoch": epoch, "train": tr, "val": val_res, "monitor": mon,
                        "temperature": model.temperature_value()})
        if verbose:
            vc, vt = val_res.get("cross_patient", {}), val_res.get("temporal", {})
            print(f"    [E{epoch:03d}] loss={tr['loss']:.4f} "
                  f"(x={tr['cross_patient_loss']:.3f} t={tr['temporal_loss']:.3f}) "
                  f"| R@1={vc.get('recall@1', float('nan')):.4f} "
                  f"R@5={vc.get('recall@5', float('nan')):.4f} "
                  f"MRR={vc.get('mrr', float('nan')):.4f} "
                  f"| T-R@1={vt.get('temporal_recall@1', float('nan')):.4f}")

        if mon > best_monitor + C.EARLY_STOP_MIN_DELTA:
            best_monitor, best_epoch, patience = mon, epoch, 0
            torch.save({"model": model.state_dict(), "spec": spec.asdict(), "epoch": epoch},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                if verbose:
                    print(f"    Early stop @E{epoch} (best E{best_epoch}, mon={best_monitor:.4f})")
                break

    best_path = out_dir / "best.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_res = evaluate_retrieval(model, test_ds, data.cxr_emb, device,
                                  args.eval_batch_size, collate_fn=collate_fn)

    results = {
        "spec": spec.asdict(), "loss_mode": spec.loss_mode,
        "w_cross": w_cross, "w_temporal": w_temporal,
        "best_epoch": best_epoch, "best_val_monitor": best_monitor,
        "n_trainable_params": n_train, "test": test_res, "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  [{spec.name}] TEST: {json.dumps(test_res)}")
    return results
