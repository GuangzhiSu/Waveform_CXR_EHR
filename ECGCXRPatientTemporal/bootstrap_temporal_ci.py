"""Bootstrap confidence intervals for within-patient temporal retrieval.

This script reloads trained staged-model checkpoints, recomputes per-query
within-patient temporal ranks on the test split, and writes method-level and
paired-delta bootstrap CIs. It is intentionally separate from ``metrics.py`` so
the main training/eval path stays compact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from runtime import get_device  # noqa: E402
from staged_dataset import StagedData, StagedDataset, collate_fn  # noqa: E402
from staged_model import StagedModel  # noqa: E402


def _data_kind(spec: dict) -> str:
    return "single" if spec.get("pairs_kind") == "single" else "sequence"


def _pairs_file(spec: dict, args) -> str:
    return {
        "single": args.single_pairs,
        "seq_target": args.seq_target_pairs,
        "seq_t1": args.pairs,
    }[spec["pairs_kind"]]


@torch.no_grad()
def _collect(model, dataset, device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    qs, pids, c2_rows = [], [], []
    model.eval()
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q, _, _ = model.encode(b)
        qs.append(q.float().cpu())
        pids.append(batch["patient_id"])
        c2_rows.append(batch["c2_row"])
    return torch.cat(qs), torch.cat(pids).numpy(), torch.cat(c2_rows).numpy()


@torch.no_grad()
def _gallery(model, cxr_emb: np.ndarray, gallery_rows: np.ndarray, device, batch_size: int):
    vecs = []
    for s in range(0, len(gallery_rows), batch_size):
        rows = gallery_rows[s:s + batch_size]
        x = torch.from_numpy(cxr_emb[rows].astype(np.float32)).to(device)
        vecs.append(model.cxr_proj(x).float().cpu())
    return torch.cat(vecs)


def _per_query_temporal(model, dataset, data, device, batch_size: int) -> dict:
    q, pids, c2_rows = _collect(model, dataset, device, batch_size)
    gallery_rows, inv = np.unique(c2_rows, return_inverse=True)
    target_gidx = inv
    gallery = _gallery(model, data.cxr_emb, gallery_rows, device, batch_size)

    gallery_patient = np.empty(len(gallery_rows), dtype=np.int64)
    gallery_patient[target_gidx] = pids
    sims = (q @ gallery.t()).numpy()

    pat_to_gidx: dict[int, list[int]] = {}
    for g, pid in enumerate(gallery_patient):
        pat_to_gidx.setdefault(int(pid), []).append(g)

    rows = []
    for i in range(sims.shape[0]):
        cand = pat_to_gidx.get(int(pids[i]), [])
        if len(cand) < 2:
            continue
        cand = np.asarray(cand)
        tpos = int(np.where(cand == target_gidx[i])[0][0])
        sub = sims[i, cand]
        rank = int((sub > sub[tpos]).sum()) + 1
        rows.append({
            "query_index": int(i),
            "patient_id": int(pids[i]),
            "target_c2_row": int(c2_rows[i]),
            "rank": rank,
            "recall1": float(rank <= 1),
            "reciprocal_rank": float(1.0 / rank),
            "n_candidates": int(len(cand)),
        })
    return {
        "rows": rows,
        "recall1": np.asarray([r["recall1"] for r in rows], dtype=float),
        "mrr": np.asarray([r["reciprocal_rank"] for r in rows], dtype=float),
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _load_run(run_dir: Path, args, device):
    results = json.load(open(run_dir / "results.json"))
    spec = results["spec"]
    model_config = results["model_config"]
    data = StagedData(
        pairs_json=_pairs_file(spec, args),
        kind=_data_kind(spec),
        cxr_emb_npy=args.cxr_emb,
        cxr_ids_json=args.cxr_ids,
        ecg_emb_npy=args.ecg_emb,
        ecg_ids_json=args.ecg_ids,
        seed=args.seed,
        train_split=C.TRAIN_SPLIT,
        val_split=C.VAL_SPLIT,
        test_split=C.TEST_SPLIT,
    )
    ds = StagedDataset(
        data,
        data.split_indices["test"],
        ecg_perturb=spec.get("ecg_perturb", "none"),
        seed=args.seed + 2,
    )
    model = StagedModel(
        SimpleNamespace(**spec),
        cxr_dim=data.cxr_emb.shape[1],
        ecg_dim=data.ecg_emb.shape[1],
        proj_dim=model_config["proj_dim"],
        cxr_proj_hidden=model_config["cxr_proj_hidden"],
        d_model=model_config["d_model"],
        ecg_tx_layers=model_config["ecg_tx_layers"],
        ecg_tx_heads=model_config["ecg_tx_heads"],
        ecg_tx_mlp_ratio=model_config["ecg_tx_mlp_ratio"],
        fusion_hidden=model_config["fusion_hidden"],
        time_emb_dim=model_config["time_emb_dim"],
        dropout=model_config["dropout"],
        temperature=model_config["temperature"],
        learnable_temperature=model_config["learnable_temperature"],
    ).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    return spec, _per_query_temporal(model, ds, data, device, args.eval_batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--reference_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pairs", default=C.PAIRS_JSON)
    ap.add_argument("--seq_target_pairs", default=C.SEQ_TARGET_PAIRS_JSON)
    ap.add_argument("--single_pairs", default=C.SINGLE_ECG_PAIRS_JSON)
    ap.add_argument("--cxr_emb", default=C.CXR_EMB_NPY)
    ap.add_argument("--cxr_ids", default=C.CXR_IDS_JSON)
    ap.add_argument("--ecg_emb", default=C.ECG_EMB_NPY)
    ap.add_argument("--ecg_ids", default=C.ECG_IDS_JSON)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if len(args.run_dirs) != len(args.labels):
        raise SystemExit("--run_dirs and --labels must have the same length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)

    per_method = {}
    summaries = []
    for label, run_dir in zip(args.labels, args.run_dirs):
        _, vals = _load_run(Path(run_dir), args, device)
        per_method[label] = vals
        r_mean, r_lo, r_hi = _bootstrap_ci(vals["recall1"], args.n_boot, rng)
        m_mean, m_lo, m_hi = _bootstrap_ci(vals["mrr"], args.n_boot, rng)
        summaries.append({
            "label": label,
            "n_queries": int(vals["recall1"].size),
            "temporal_recall@1": r_mean,
            "temporal_recall@1_ci95_low": r_lo,
            "temporal_recall@1_ci95_high": r_hi,
            "temporal_mrr": m_mean,
            "temporal_mrr_ci95_low": m_lo,
            "temporal_mrr_ci95_high": m_hi,
        })

    ref = per_method[args.reference_label]
    deltas = []
    for label, vals in per_method.items():
        if label == args.reference_label:
            continue
        if vals["recall1"].shape != ref["recall1"].shape:
            raise RuntimeError(f"Cannot paired-bootstrap {label}: query shape differs")
        dr = ref["recall1"] - vals["recall1"]
        dm = ref["mrr"] - vals["mrr"]
        r_mean, r_lo, r_hi = _bootstrap_ci(dr, args.n_boot, rng)
        m_mean, m_lo, m_hi = _bootstrap_ci(dm, args.n_boot, rng)
        deltas.append({
            "reference_label": args.reference_label,
            "comparison_label": label,
            "delta_temporal_recall@1": r_mean,
            "delta_temporal_recall@1_ci95_low": r_lo,
            "delta_temporal_recall@1_ci95_high": r_hi,
            "delta_temporal_mrr": m_mean,
            "delta_temporal_mrr_ci95_low": m_lo,
            "delta_temporal_mrr_ci95_high": m_hi,
        })

    with open(out_dir / "method_bootstrap_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    with open(out_dir / "paired_delta_bootstrap_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0]))
        w.writeheader()
        w.writerows(deltas)
    with open(out_dir / "bootstrap_ci.json", "w") as f:
        json.dump({"methods": summaries, "paired_deltas": deltas}, f, indent=2)

    print(json.dumps({"methods": summaries, "paired_deltas": deltas}, indent=2))


if __name__ == "__main__":
    main()
