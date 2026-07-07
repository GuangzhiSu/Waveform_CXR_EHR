"""Generic train + eval loop shared by all staged experiments.

Given an :class:`~experiments.ExperimentSpec` and a loaded :class:`StagedData`,
``fit`` trains a :class:`StagedModel` with the N-patients x K-intervals sampler
and the two contrastive losses, doing patient-split retrieval evaluation each
epoch with early stopping, then a final test evaluation with the best weights.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from losses import batch_retrieval_metrics, total_loss
from metrics import evaluate_retrieval
from runtime import set_seed
from sampler import NPatientsKIntervalsSampler
from staged_dataset import StagedData, StagedDataset, collate_fn
from staged_model import StagedModel


def _temporal_negative_stats(dataset: StagedDataset) -> dict:
    by_patient: dict[int, set[int]] = {}
    for idx in dataset.indices:
        pair = dataset.data.pairs[idx]
        by_patient.setdefault(int(pair["patient_id"]), set()).add(int(pair["c2"]))

    counts: dict[int, int] = {}
    for idx in dataset.indices:
        pair = dataset.data.pairs[idx]
        n_neg = len(by_patient[int(pair["patient_id"])] - {int(pair["c2"])})
        counts[n_neg] = counts.get(n_neg, 0) + 1

    n_queries = len(dataset.indices)
    n_with_neg = sum(v for k, v in counts.items() if k >= 1)
    return {
        "n_queries": int(n_queries),
        "n_patients": int(len(by_patient)),
        "negative_count_distribution": {str(k): int(counts[k]) for k in sorted(counts)},
        "queries_with_temporal_negative": int(n_with_neg),
        "fraction_with_temporal_negative": (n_with_neg / n_queries) if n_queries else float("nan"),
    }


def _safe_div(num: int | float, den: int | float) -> float:
    return (num / den) if den else float("nan")


def _run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm,
               temporal_min_horizon_hours=None, temporal_max_horizon_hours=None,
               epoch: int = 0, global_step_start: int = 0,
               iteration_records: list | None = None, dynamics_log_every: int = 1):
    model.train()
    agg = {"loss": 0.0, "cross_patient_loss": 0.0, "temporal_loss": 0.0,
           "n_temporal_rows": 0, "steps": 0, "skipped": 0,
           "cross_patient_top1_correct": 0, "cross_patient_top5_correct": 0,
           "cross_patient_rows": 0, "temporal_top1_correct": 0,
           "temporal_top5_correct": 0, "temporal_metric_rows": 0}
    global_step = int(global_step_start)
    dynamics_log_every = max(1, int(dynamics_log_every or 1))
    for iter_in_epoch, batch in enumerate(loader, start=1):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        loss, logs = total_loss(out["logits"], b["patient_id"], w_cross, w_temporal,
                                c2_rows=b.get("c2_row"),
                                c2_times_h=b.get("c2_time_h"),
                                ecg_times_h=b.get("ecg_times_h"),
                                ecg_mask=b.get("ecg_mask"),
                                temporal_min_horizon_hours=temporal_min_horizon_hours,
                                temporal_max_horizon_hours=temporal_max_horizon_hours)
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
        batch_metrics = batch_retrieval_metrics(
            out["logits"].detach(),
            b["patient_id"],
            c2_rows=b.get("c2_row"),
            c2_times_h=b.get("c2_time_h"),
            ecg_times_h=b.get("ecg_times_h"),
            ecg_mask=b.get("ecg_mask"),
            temporal_min_horizon_hours=temporal_min_horizon_hours,
            temporal_max_horizon_hours=temporal_max_horizon_hours,
        )
        agg["cross_patient_top1_correct"] += batch_metrics["cross_patient_top1_correct"]
        agg["cross_patient_top5_correct"] += batch_metrics["cross_patient_top5_correct"]
        agg["cross_patient_rows"] += batch_metrics["cross_patient_rows"]
        agg["temporal_top1_correct"] += batch_metrics["temporal_top1_correct"]
        agg["temporal_top5_correct"] += batch_metrics["temporal_top5_correct"]
        agg["temporal_metric_rows"] += batch_metrics["temporal_rows"]
        agg["steps"] += 1
        global_step += 1

        if iteration_records is not None and (global_step % dynamics_log_every == 0):
            x_rows = batch_metrics["cross_patient_rows"]
            t_rows = batch_metrics["temporal_rows"]
            iteration_records.append({
                "global_step": global_step,
                "epoch": epoch,
                "iter_in_epoch": iter_in_epoch,
                "loss": logs["loss"],
                "cross_patient_loss": logs["cross_patient_loss"],
                "temporal_loss": logs["temporal_loss"],
                "train_cross_patient_R@1": _safe_div(
                    batch_metrics["cross_patient_top1_correct"], x_rows),
                "train_cross_patient_R@5": _safe_div(
                    batch_metrics["cross_patient_top5_correct"], x_rows),
                "train_cross_patient_rows": x_rows,
                "train_temporal_R@1": _safe_div(
                    batch_metrics["temporal_top1_correct"], t_rows),
                "train_temporal_R@5": _safe_div(
                    batch_metrics["temporal_top5_correct"], t_rows),
                "train_temporal_rows": t_rows,
            })
    s = max(agg["steps"], 1)

    def _acc(correct, rows):
        return (correct / rows) if rows else float("nan")

    return {"loss": agg["loss"] / s, "cross_patient_loss": agg["cross_patient_loss"] / s,
            "temporal_loss": agg["temporal_loss"] / s, "steps": agg["steps"],
            "skipped": agg["skipped"], "avg_temporal_rows": agg["n_temporal_rows"] / s,
            "cross_patient_batch_top1": _acc(agg["cross_patient_top1_correct"],
                                             agg["cross_patient_rows"]),
            "cross_patient_batch_top5": _acc(agg["cross_patient_top5_correct"],
                                             agg["cross_patient_rows"]),
            "temporal_batch_top1": _acc(agg["temporal_top1_correct"],
                                        agg["temporal_metric_rows"]),
            "temporal_batch_top5": _acc(agg["temporal_top5_correct"],
                                        agg["temporal_metric_rows"]),
            "cross_patient_batch_rows": agg["cross_patient_rows"],
            "temporal_batch_rows": agg["temporal_metric_rows"],
            "last_global_step": global_step}


def _monitor(eval_res: dict, loss_mode: str) -> float:
    if not eval_res:
        return float("nan")
    if loss_mode == "temporal":
        v = eval_res.get("temporal", {}).get("temporal_mrr", float("nan"))
        return v if v == v else 0.0
    return eval_res.get("cross_patient", {}).get("mrr", 0.0)


def _pairs_file(spec, args) -> str:
    return {
        "single": args.single_pairs,
        "seq_target": args.seq_target_pairs,
        "seq_t1": args.pairs,
    }[spec.pairs_kind]


def _write_train_dynamics(records: list[dict], out_dir: Path):
    if not records:
        return {}

    json_path = out_dir / "train_dynamics.json"
    csv_path = out_dir / "train_dynamics.csv"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(records[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

    plot_path = out_dir / "train_dynamics_r_at_1_5.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.asarray([r["global_step"] for r in records], dtype=float)

        def values(key: str):
            return np.asarray([r.get(key, float("nan")) for r in records], dtype=float)

        def rolling(y: np.ndarray, window: int = 100):
            out = np.full_like(y, np.nan, dtype=float)
            for i in range(len(y)):
                s = max(0, i - window + 1)
                chunk = y[s:i + 1]
                if np.isfinite(chunk).any():
                    out[i] = np.nanmean(chunk)
            return out

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        series = [
            (axes[0], "train_cross_patient_R@1", "Cross-patient R@1", "#1f77b4"),
            (axes[0], "train_cross_patient_R@5", "Cross-patient R@5", "#ff7f0e"),
            (axes[1], "train_temporal_R@1", "Temporal R@1", "#2ca02c"),
            (axes[1], "train_temporal_R@5", "Temporal R@5", "#d62728"),
        ]
        for ax, key, label, color in series:
            y = values(key)
            ax.plot(steps, y, color=color, alpha=0.18, linewidth=0.6)
            ax.plot(steps, rolling(y), color=color, linewidth=1.8, label=f"{label} (rolling 100)")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.legend(loc="best")
        axes[0].set_ylabel("Batch recall")
        axes[1].set_ylabel("Batch recall")
        axes[1].set_xlabel("Optimizer step")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: failed to plot train dynamics: {e}")
        plot_path = None

    return {
        "train_dynamics_json": str(json_path),
        "train_dynamics_csv": str(csv_path),
        "train_dynamics_plot": str(plot_path) if plot_path is not None else None,
    }


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
        num_batches=args.steps_per_epoch, seed=args.seed,
        target_rows=train_ds.target_rows(),
        min_targets_per_patient=getattr(args, "min_train_targets_per_patient", 1),
        sample_unique_targets=getattr(args, "sample_unique_targets", False))
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    model = StagedModel(
        spec, cxr_dim=data.cxr_emb.shape[1], ecg_dim=data.ecg_emb.shape[1],
        proj_dim=args.proj_dim, cxr_proj_hidden=C.CXR_PROJ_HIDDEN, d_model=args.d_model,
        ecg_tx_layers=args.ecg_tx_layers, ecg_tx_heads=C.ECG_TX_HEADS,
        ecg_tx_mlp_ratio=C.ECG_TX_MLP_RATIO, fusion_hidden=C.FUSION_HIDDEN,
        time_emb_dim=C.TIME_EMB_DIM, dropout=C.DROPOUT, temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
    ).to(device)

    init_from = getattr(args, "init_from", None)
    init_report = None
    if init_from:
        ckpt = torch.load(init_from, map_location=device)
        state = ckpt.get("model", ckpt)
        current = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in current and tuple(current[k].shape) == tuple(v.shape)
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        init_report = {
            "path": str(init_from),
            "loaded_keys": len(compatible),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        if verbose:
            print(f"  warm-start: loaded {len(compatible)} compatible tensors from {init_from}")

    if getattr(args, "freeze_cxr_base", False):
        for p in model.cxr_proj.parameters():
            p.requires_grad = False
        if getattr(model, "g", None) is not None:
            for p in model.g.parameters():
                p.requires_grad = False
        if verbose:
            print("  freeze_cxr_base: froze cxr_proj and g parameters")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_config = {
        "cxr_dim": int(data.cxr_emb.shape[1]),
        "ecg_dim": int(data.ecg_emb.shape[1]),
        "proj_dim": int(args.proj_dim),
        "cxr_proj_hidden": int(C.CXR_PROJ_HIDDEN),
        "d_model": int(args.d_model),
        "ecg_tx_layers": int(args.ecg_tx_layers),
        "ecg_tx_heads": int(C.ECG_TX_HEADS),
        "ecg_tx_mlp_ratio": float(C.ECG_TX_MLP_RATIO),
        "fusion_hidden": int(C.FUSION_HIDDEN),
        "time_emb_dim": int(C.TIME_EMB_DIM),
        "dropout": float(C.DROPOUT),
        "temperature": float(args.temperature),
        "learnable_temperature": bool(args.learnable_temperature),
    }

    effective_loss_mode = getattr(args, "loss_mode_override", None) or spec.loss_mode
    effective_lambda_temporal = (
        spec.lambda_temporal if getattr(args, "lambda_temporal_override", None) is None
        else float(args.lambda_temporal_override)
    )
    w_cross, w_temporal = C.loss_weights(effective_loss_mode, effective_lambda_temporal)
    if verbose:
        print(f"  [{spec.name}] kind={spec.pairs_kind} trainable={n_train:,} "
              f"w_cross={w_cross} w_temporal={w_temporal} perturb={perturb}")
        print(f"  sampler: N={args.n_patients} K={args.k_intervals} "
              f"eligible_patients={len(sampler.eligible):,} "
              f"min_targets={getattr(args, 'min_train_targets_per_patient', 1)} "
              f"unique_targets={getattr(args, 'sample_unique_targets', False)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_monitor, best_epoch, patience = -1.0, -1, 0
    history = []
    train_dynamics = [] if not getattr(args, "no_train_dynamics", False) else None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        tr = _run_epoch(model, train_loader, optimizer, device, w_cross, w_temporal,
                        args.max_grad_norm,
                        temporal_min_horizon_hours=spec.temporal_min_horizon_hours,
                        temporal_max_horizon_hours=spec.temporal_max_horizon_hours,
                        epoch=epoch, global_step_start=global_step,
                        iteration_records=train_dynamics,
                        dynamics_log_every=getattr(args, "dynamics_log_every", 1))
        global_step = tr.pop("last_global_step")
        val_res = evaluate_retrieval(model, val_ds, data.cxr_emb, device,
                                     args.eval_batch_size, collate_fn=collate_fn)
        mon = _monitor(val_res, effective_loss_mode)
        history.append({"epoch": epoch, "train": tr, "val": val_res, "monitor": mon,
                        "temperature": model.temperature_value()})
        if verbose:
            vc, vt = val_res.get("cross_patient", {}), val_res.get("temporal", {})
            print(f"    [E{epoch:03d}] loss={tr['loss']:.4f} "
                  f"(x={tr['cross_patient_loss']:.3f} t={tr['temporal_loss']:.3f}) "
                  f"| B@1 x={tr['cross_patient_batch_top1']:.3f} "
                  f"t={tr['temporal_batch_top1']:.3f} "
                  f"| R@1={vc.get('recall@1', float('nan')):.4f} "
                  f"R@5={vc.get('recall@5', float('nan')):.4f} "
                  f"MRR={vc.get('mrr', float('nan')):.4f} "
                  f"| T-R@1={vt.get('temporal_recall@1', float('nan')):.4f}")

        if mon > best_monitor + C.EARLY_STOP_MIN_DELTA:
            best_monitor, best_epoch, patience = mon, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "spec": spec.asdict(),
                "model_config": model_config,
                "epoch": epoch,
            }, out_dir / "best.pt")
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
        "spec": spec.asdict(), "loss_mode": effective_loss_mode,
        "w_cross": w_cross, "w_temporal": w_temporal,
        "effective_lambda_temporal": effective_lambda_temporal,
        "best_epoch": best_epoch, "best_val_monitor": best_monitor,
        "n_trainable_params": n_train, "model_config": model_config,
        "init_from": init_report,
        "sampler_config": {
            "n_patients": int(args.n_patients),
            "k_intervals": int(args.k_intervals),
            "min_train_targets_per_patient": int(getattr(args, "min_train_targets_per_patient", 1)),
            "sample_unique_targets": bool(getattr(args, "sample_unique_targets", False)),
            "eligible_train_patients": int(len(sampler.eligible)),
        },
        "split_temporal_negative_stats": {
            "train": _temporal_negative_stats(train_ds),
            "val": _temporal_negative_stats(val_ds),
            "test": _temporal_negative_stats(test_ds),
        },
        "test": test_res, "history": history,
    }
    results.update(_write_train_dynamics(train_dynamics or [], out_dir))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  [{spec.name}] TEST: {json.dumps(test_res)}")
    return results
