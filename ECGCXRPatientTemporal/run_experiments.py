"""Run the staged ECG->CXR experiments and emit one unified results table.

Recommended order (Step 1 .. Step 6):

    Step 1  exp1a_single_ecg_cross        single ECG -> future CXR, cross only
            exp1b_single_ecg_combined     single ECG -> future CXR, cross + 0.2 temporal
    Step 2  exp2_single_ecg_predictor     single ECG + predictor g(., dt)
    Step 3  exp3a_seq_ecg_meanpool        ECG sequence -> mean pool
            exp3b_seq_ecg_future_query    ECG sequence + learnable future query
    Step 4  exp4c_fusion_cxr1_ecgseq      CXR_t1 + ECG sequence  (target model)
            exp4a_ecg_only                shortcut control A
            exp4b_cxr_only                shortcut control B
            exp4d_fusion_shuffled_ecg     shortcut control D
            exp4e_fusion_zeroed_ecg       shortcut control E

Examples:
    python run_experiments.py                       # all, in order
    python run_experiments.py --only step1 step2
    python run_experiments.py --only exp4c_fusion_cxr1_ecgseq exp4b_cxr_only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from engine import fit, load_staged_data  # noqa: E402
from experiments import ALL_IN_ORDER, REGISTRY, STEP_GROUPS  # noqa: E402

TABLE_COLUMNS = [
    "experiment_name", "input_type", "target_window",
    "uses_cxr_t1", "uses_single_ecg", "uses_ecg_sequence",
    "uses_predictor_g", "uses_transformer", "uses_future_query",
    "ecg_perturb", "fusion_mode", "loss_type", "lambda_temporal",
    "cross_patient_recall@1", "cross_patient_recall@5", "cross_patient_recall@10",
    "cross_patient_mrr", "cross_patient_median_rank",
    "within_patient_temporal_recall@1", "within_patient_temporal_recall@5",
    "within_patient_temporal_mrr",
]


def resolve_specs(only) -> list:
    if not only:
        names = list(ALL_IN_ORDER)
    else:
        names = []
        for token in only:
            if token in STEP_GROUPS:
                names.extend(STEP_GROUPS[token])
            elif token in REGISTRY:
                names.append(token)
            else:
                raise SystemExit(f"Unknown experiment/step: {token!r}. "
                                 f"Choices: {list(STEP_GROUPS)} or {list(REGISTRY)}")
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return [REGISTRY[n] for n in ordered]


def results_to_row(spec, results: dict) -> dict:
    row = spec.table_row_meta()
    test = results.get("test", {})
    cp = test.get("cross_patient", {})
    tp = test.get("temporal", {})
    row.update({
        "cross_patient_recall@1": cp.get("recall@1"),
        "cross_patient_recall@5": cp.get("recall@5"),
        "cross_patient_recall@10": cp.get("recall@10"),
        "cross_patient_mrr": cp.get("mrr"),
        "cross_patient_median_rank": cp.get("median_rank"),
        "within_patient_temporal_recall@1": tp.get("temporal_recall@1"),
        "within_patient_temporal_recall@5": tp.get("temporal_recall@5"),
        "within_patient_temporal_mrr": tp.get("temporal_mrr"),
    })
    return row


def write_table(rows: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in TABLE_COLUMNS})
    json_path = out_dir / "results_table.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n=== Unified results table -> {csv_path} ===")
    _pretty_print(rows)
    return csv_path


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Y" if v else "."
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _pretty_print(rows: list):
    cols = ["experiment_name", "input_type", "uses_cxr_t1", "ecg_perturb", "fusion_mode",
            "cross_patient_recall@1", "cross_patient_recall@5", "cross_patient_mrr",
            "within_patient_temporal_recall@1", "within_patient_temporal_mrr"]
    short = {"experiment_name": "experiment", "input_type": "input",
             "uses_cxr_t1": "cxr1", "ecg_perturb": "perturb",
             "fusion_mode": "fusion",
             "cross_patient_recall@1": "xR@1", "cross_patient_recall@5": "xR@5",
             "cross_patient_mrr": "xMRR", "within_patient_temporal_recall@1": "tR@1",
             "within_patient_temporal_mrr": "tMRR"}
    widths = {c: max(len(short[c]), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(short[c].ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset: step1/step2/step3/step4 or specific experiment names.")
    ap.add_argument("--pairs", default=C.PAIRS_JSON,
                    help="Exp 4 sequence pairs (with CXR_t1).")
    ap.add_argument("--seq_target_pairs", default=C.SEQ_TARGET_PAIRS_JSON,
                    help="Exp 3 sequence pairs (no CXR_t1).")
    ap.add_argument("--single_pairs", default=C.SINGLE_ECG_PAIRS_JSON,
                    help="Single-ECG pairs (Exp 1/2).")
    ap.add_argument("--cxr_emb", default=C.CXR_EMB_NPY)
    ap.add_argument("--cxr_ids", default=C.CXR_IDS_JSON)
    ap.add_argument("--ecg_emb", default=C.ECG_EMB_NPY)
    ap.add_argument("--ecg_ids", default=C.ECG_IDS_JSON)
    ap.add_argument("--output_dir", default=C.STAGED_OUTPUT_DIR)
    ap.add_argument("--proj_dim", type=int, default=C.PROJ_DIM)
    ap.add_argument("--d_model", type=int, default=C.D_MODEL)
    ap.add_argument("--ecg_tx_layers", type=int, default=C.ECG_TX_LAYERS)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--learnable_temperature", action="store_true",
                    default=C.LEARNABLE_TEMPERATURE)
    ap.add_argument("--n_patients", type=int, default=C.N_PATIENTS)
    ap.add_argument("--k_intervals", type=int, default=C.K_INTERVALS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--steps_per_epoch", type=int, default=C.STEPS_PER_EPOCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--max_grad_norm", type=float, default=C.MAX_GRAD_NORM)
    ap.add_argument("--loss_mode_override", default=None, choices=["cross", "temporal", "combined"],
                    help="Override each selected spec's loss_mode.")
    ap.add_argument("--lambda_temporal_override", type=float, default=None,
                    help="Override each selected spec's temporal loss weight.")
    ap.add_argument("--init_from", default=None,
                    help="Optional checkpoint for compatible-key warm-start.")
    ap.add_argument("--freeze_cxr_base", action="store_true",
                    help="Freeze cxr_proj and g after optional warm-start.")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--early_stop_patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dynamics_log_every", type=int, default=1,
                    help="Record batch R@1/R@5 every N optimizer steps.")
    ap.add_argument("--no_train_dynamics", action="store_true",
                    help="Disable per-iteration train dynamics CSV/JSON/PNG outputs.")
    return ap.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main():
    args = build_args()
    device = get_device(args.device)
    specs = resolve_specs(args.only)
    print(f"=== Staged ECG->CXR experiments (device={device}) ===")
    print("  Running: " + ", ".join(s.name for s in specs))

    # Cache loaded data per pairs-kind (single / sequence) to avoid reloading.
    data_cache: dict[str, object] = {}
    rows = []
    for spec in specs:
        if spec.pairs_kind not in data_cache:
            print(f"\n--- Loading {spec.pairs_kind} data ---")
            data_cache[spec.pairs_kind] = load_staged_data(spec, args)
        print(f"\n########## {spec.name} ##########")
        print(f"  {spec.description}")
        try:
            results = fit(spec, args, data=data_cache[spec.pairs_kind], device=device)
            rows.append(results_to_row(spec, results))
        except Exception as e:  # keep going so one failure does not lose the table
            import traceback
            traceback.print_exc()
            print(f"  !! {spec.name} FAILED: {e}")
            row = spec.table_row_meta()
            row["experiment_name"] = spec.name + " (FAILED)"
            rows.append(row)

    write_table(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
