"""Diagnose in-batch retrieval accuracy for staged contrastive checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from experiments import REGISTRY  # noqa: E402
from losses import batch_retrieval_metrics  # noqa: E402
from run_experiments import resolve_specs  # noqa: E402
from sampler import NPatientsKIntervalsSampler  # noqa: E402
from staged_dataset import StagedData, StagedDataset, collate_fn  # noqa: E402
from staged_model import StagedModel  # noqa: E402


SUMMARY_COLUMNS = [
    "experiment",
    "split",
    "n_batches",
    "cross_top1_weighted",
    "cross_top1_batch_mean",
    "cross_top1_batch_p10",
    "cross_top1_batch_p50",
    "cross_top1_batch_p90",
    "cross_top5_weighted",
    "temporal_top1_weighted",
    "temporal_top1_batch_mean",
    "temporal_top1_batch_p10",
    "temporal_top1_batch_p50",
    "temporal_top1_batch_p90",
    "temporal_top5_weighted",
    "cross_rows",
    "temporal_rows",
]


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _percentiles(values: list[float]) -> dict:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(vals.mean()),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _weighted(correct: int, rows: int) -> float | None:
    return (correct / rows) if rows else None


def _load_data(spec, args, cache: dict[str, StagedData]) -> StagedData:
    if spec.pairs_kind not in cache:
        pairs_json = {
            "single": args.single_pairs,
            "seq_target": args.seq_target_pairs,
            "seq_t1": args.pairs,
        }[spec.pairs_kind]
        cache[spec.pairs_kind] = StagedData(
            pairs_json=pairs_json,
            kind=spec.data_kind(),
            cxr_emb_npy=args.cxr_emb,
            cxr_ids_json=args.cxr_ids,
            ecg_emb_npy=args.ecg_emb,
            ecg_ids_json=args.ecg_ids,
            seed=args.seed,
            train_split=C.TRAIN_SPLIT,
            val_split=C.VAL_SPLIT,
            test_split=C.TEST_SPLIT,
        )
    return cache[spec.pairs_kind]


def _build_model(spec, data: StagedData, args, device):
    model = StagedModel(
        spec,
        cxr_dim=data.cxr_emb.shape[1],
        ecg_dim=data.ecg_emb.shape[1],
        proj_dim=args.proj_dim,
        cxr_proj_hidden=C.CXR_PROJ_HIDDEN,
        d_model=args.d_model,
        ecg_tx_layers=args.ecg_tx_layers,
        ecg_tx_heads=C.ECG_TX_HEADS,
        ecg_tx_mlp_ratio=C.ECG_TX_MLP_RATIO,
        fusion_hidden=C.FUSION_HIDDEN,
        time_emb_dim=C.TIME_EMB_DIM,
        dropout=C.DROPOUT,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
    ).to(device)
    return model


def _dataset_for_split(spec, data: StagedData, split: str, seed: int) -> StagedDataset:
    split_offset = {"train": 0, "val": 1, "test": 2}[split]
    return StagedDataset(
        data,
        data.split_indices[split],
        ecg_perturb=spec.ecg_perturb,
        seed=seed + split_offset,
    )


@torch.no_grad()
def _evaluate_split(model, spec, dataset: StagedDataset, split: str, args, device):
    if len(dataset) == 0:
        return {"summary": {"experiment": spec.name, "split": split, "n_batches": 0}, "batches": []}
    max_batches = None if args.max_batches <= 0 else args.max_batches
    sampler = NPatientsKIntervalsSampler(
        dataset.patient_ids(),
        args.n_patients,
        args.k_intervals,
        num_batches=max_batches,
        seed=args.seed + {"train": 0, "val": 10_000, "test": 20_000}[split],
    )
    sampler.set_epoch(args.sampler_epoch)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    totals = {
        "cross_top1_correct": 0,
        "cross_top5_correct": 0,
        "cross_rows": 0,
        "temporal_top1_correct": 0,
        "temporal_top5_correct": 0,
        "temporal_rows": 0,
    }
    records = []
    model.eval()
    for batch_idx, batch in enumerate(loader):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        m = batch_retrieval_metrics(
            out["logits"],
            b["patient_id"],
            c2_rows=b.get("c2_row"),
            c2_times_h=b.get("c2_time_h"),
            ecg_times_h=b.get("ecg_times_h"),
            ecg_mask=b.get("ecg_mask"),
            temporal_min_horizon_hours=spec.temporal_min_horizon_hours,
            temporal_max_horizon_hours=spec.temporal_max_horizon_hours,
        )
        cross_rows = m["cross_patient_rows"]
        temporal_rows = m["temporal_rows"]
        rec = {
            "batch_idx": batch_idx,
            "batch_size": int(b["patient_id"].numel()),
            "cross_top1": _weighted(m["cross_patient_top1_correct"], cross_rows),
            "cross_top5": _weighted(m["cross_patient_top5_correct"], cross_rows),
            "cross_rows": cross_rows,
            "temporal_top1": _weighted(m["temporal_top1_correct"], temporal_rows),
            "temporal_top5": _weighted(m["temporal_top5_correct"], temporal_rows),
            "temporal_rows": temporal_rows,
        }
        records.append(rec)
        totals["cross_top1_correct"] += m["cross_patient_top1_correct"]
        totals["cross_top5_correct"] += m["cross_patient_top5_correct"]
        totals["cross_rows"] += cross_rows
        totals["temporal_top1_correct"] += m["temporal_top1_correct"]
        totals["temporal_top5_correct"] += m["temporal_top5_correct"]
        totals["temporal_rows"] += temporal_rows

    cross_top1 = _percentiles([r["cross_top1"] for r in records if r["cross_top1"] is not None])
    temporal_top1 = _percentiles(
        [r["temporal_top1"] for r in records if r["temporal_top1"] is not None]
    )
    summary = {
        "experiment": spec.name,
        "split": split,
        "n_batches": len(records),
        "cross_top1_weighted": _weighted(totals["cross_top1_correct"], totals["cross_rows"]),
        "cross_top1_batch_mean": cross_top1["mean"],
        "cross_top1_batch_p10": cross_top1["p10"],
        "cross_top1_batch_p50": cross_top1["p50"],
        "cross_top1_batch_p90": cross_top1["p90"],
        "cross_top5_weighted": _weighted(totals["cross_top5_correct"], totals["cross_rows"]),
        "temporal_top1_weighted": _weighted(totals["temporal_top1_correct"], totals["temporal_rows"]),
        "temporal_top1_batch_mean": temporal_top1["mean"],
        "temporal_top1_batch_p10": temporal_top1["p10"],
        "temporal_top1_batch_p50": temporal_top1["p50"],
        "temporal_top1_batch_p90": temporal_top1["p90"],
        "temporal_top5_weighted": _weighted(totals["temporal_top5_correct"], totals["temporal_rows"]),
        "cross_rows": totals["cross_rows"],
        "temporal_rows": totals["temporal_rows"],
    }
    return {"summary": summary, "batches": records}


def _write_outputs(out_dir: Path, summaries: list[dict], details: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "batch_accuracy_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for row in summaries:
            w.writerow({k: _jsonable(row.get(k)) for k in SUMMARY_COLUMNS})

    json_path = out_dir / "batch_accuracy_diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2, default=_jsonable)
    return csv_path, json_path


def _print_summary(summaries: list[dict]):
    print("experiment                         split  xB@1   tB@1   xB@5   tB@5   batches")
    print("-" * 82)
    for r in summaries:
        def fmt(v):
            return "  -" if v is None else f"{v:5.3f}"

        print(
            f"{r['experiment']:<34} {r['split']:<5} "
            f"{fmt(r.get('cross_top1_weighted'))}  "
            f"{fmt(r.get('temporal_top1_weighted'))}  "
            f"{fmt(r.get('cross_top5_weighted'))}  "
            f"{fmt(r.get('temporal_top5_weighted'))}  "
            f"{r.get('n_batches', 0):>7}"
        )


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset: step1/step2/step3/step4 or specific experiment names.")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--output_dir", default=C.STAGED_OUTPUT_DIR)
    ap.add_argument("--pairs", default=C.PAIRS_JSON)
    ap.add_argument("--seq_target_pairs", default=C.SEQ_TARGET_PAIRS_JSON)
    ap.add_argument("--single_pairs", default=C.SINGLE_ECG_PAIRS_JSON)
    ap.add_argument("--cxr_emb", default=C.CXR_EMB_NPY)
    ap.add_argument("--cxr_ids", default=C.CXR_IDS_JSON)
    ap.add_argument("--ecg_emb", default=C.ECG_EMB_NPY)
    ap.add_argument("--ecg_ids", default=C.ECG_IDS_JSON)
    ap.add_argument("--proj_dim", type=int, default=C.PROJ_DIM)
    ap.add_argument("--d_model", type=int, default=C.D_MODEL)
    ap.add_argument("--ecg_tx_layers", type=int, default=C.ECG_TX_LAYERS)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--learnable_temperature", action="store_true",
                    default=C.LEARNABLE_TEMPERATURE)
    ap.add_argument("--n_patients", type=int, default=C.N_PATIENTS)
    ap.add_argument("--k_intervals", type=int, default=C.K_INTERVALS)
    ap.add_argument("--max_batches", type=int, default=100,
                    help="Batches per split; use 0 for all batches.")
    ap.add_argument("--sampler_epoch", type=int, default=999)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main():
    args = build_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    specs = resolve_specs(args.only)
    output_dir = Path(args.output_dir)
    data_cache: dict[str, StagedData] = {}
    summaries = []
    details = {"output_dir": str(output_dir), "splits": args.splits, "experiments": {}}

    print(f"=== Batch accuracy diagnostics (device={device}, max_batches={args.max_batches}) ===")
    for spec in specs:
        if spec.name not in REGISTRY:
            raise SystemExit(f"Unknown experiment: {spec.name}")
        ckpt_path = output_dir / spec.name / "best.pt"
        if not ckpt_path.is_file():
            print(f"  !! skip {spec.name}: missing {ckpt_path}")
            continue
        data = _load_data(spec, args, data_cache)
        model = _build_model(spec, data, args, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        details["experiments"][spec.name] = {"checkpoint": str(ckpt_path), "splits": {}}
        for split in args.splits:
            ds = _dataset_for_split(spec, data, split, args.seed)
            res = _evaluate_split(model, spec, ds, split, args, device)
            details["experiments"][spec.name]["splits"][split] = res
            summaries.append(res["summary"])

    csv_path, json_path = _write_outputs(output_dir, summaries, details)
    _print_summary(summaries)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
