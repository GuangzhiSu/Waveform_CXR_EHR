#!/usr/bin/env python3
"""
Sanity-check ECG/CXR: file-path coverage, tensor stats, and forward-pass softmax predictions.

Usage (from repo root or BaselineExperiment):
  python verify_inputs_and_predictions.py --modality both
  python verify_inputs_and_predictions.py --modality ecg --ecg_csv /path/to/p2f_ecg_all_classified.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "ECGUni"))
sys.path.insert(0, str(_BASE / "CXRUni"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classification_utils import stratified_train_val_test_indices

from cxr_classification.dataset import CXRClassificationDataset


def _ecg_stats(df: pd.DataFrame, sample_max: int | None) -> dict:
    from ECGUni.dataset import load_ecg, normalize_ecg_per_lead

    rows = range(len(df)) if sample_max is None else range(min(sample_max, len(df)))
    n = 0
    path_ok = 0
    nonzero_after_load = 0
    for i in rows:
        n += 1
        row = df.iloc[i]
        wf = row.get("wf_File_Path")
        if pd.isna(wf) or not str(wf).strip():
            continue
        p = str(wf).strip()
        if os.path.exists(p) or os.path.exists(p + ".hea"):
            path_ok += 1
        sig = load_ecg(p)
        if float(sig.abs().mean()) > 1e-8:
            nonzero_after_load += 1
    # Normalized sample (first row with nonzero load)
    norm_mean = norm_std = None
    for i in rows:
        wf = df.iloc[i].get("wf_File_Path")
        if pd.isna(wf) or not str(wf).strip():
            continue
        sig = load_ecg(str(wf).strip())
        if float(sig.abs().mean()) <= 1e-8:
            continue
        z = normalize_ecg_per_lead(sig)
        norm_mean = float(z.mean())
        norm_std = float(z.std())
        break
    return {
        "rows_scanned": n,
        "path_exists_or_hea": path_ok,
        "nonzero_signal_after_load": nonzero_after_load,
        "sample_zscore_mean": norm_mean,
        "sample_zscore_std": norm_std,
    }


def _cxr_stats(full_df: pd.DataFrame, cxr_root: str, sample_max: int | None) -> dict:
    from cxr_classification.dataset import get_cxr_path

    # ``full_df`` must already include ``study_id`` (metadata merge), same as training.
    rows = range(len(full_df)) if sample_max is None else range(min(sample_max, len(full_df)))
    n = 0
    path_ok = 0
    empty_path = 0
    for i in rows:
        n += 1
        row = full_df.iloc[i]
        dicom_id = row["dicom_id"]
        subject_id = row["subject_id"]
        study_id = row.get("study_id", row.get("wf_Study_ID", ""))
        p = get_cxr_path(dicom_id, subject_id, study_id, cxr_root)
        if not p:
            empty_path += 1
            continue
        if os.path.isfile(p):
            path_ok += 1
    return {
        "rows_scanned": n,
        "jpg_found": path_ok,
        "empty_constructed_path": empty_path,
    }


def _run_ecg_predictions(
    csv_path: str,
    ecg_ckpt: str | None,
    model_ckpt: Path | None,
    device: torch.device,
    seed: int,
) -> None:
    from ECGUni.config import DATA_CSV, ECG_CKPT, TRAIN_SPLIT, VAL_SPLIT, NUM_CLASSES, HIDDEN_DIM
    from ECGUni.dataset import ECGClassificationDataset
    from ECGUni.model import ECGClassificationBaseline
    from ECGUni.config import FREEZE_ENCODER, USE_LORA, LORA_R, LORA_ALPHA

    csv_path = csv_path or DATA_CSV
    if not csv_path or not os.path.isfile(csv_path):
        print("  SKIP: no valid ECG CSV for forward pass.")
        return
    ckpt_enc = ecg_ckpt or ECG_CKPT
    ckpt_enc = ckpt_enc if ckpt_enc and os.path.isfile(ckpt_enc) else None

    ds = ECGClassificationDataset(csv_path=csv_path, normalize_per_lead=True)
    y = ds.df["p2f_class"].values
    ts = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, TRAIN_SPLIT, VAL_SPLIT, ts, seed
    )
    from torch.utils.data import Subset

    val_ds = Subset(ds, idx_val[: min(32, len(idx_val))].tolist())
    print("  First 3 val samples (path → exists):")
    for j in range(min(3, len(val_ds))):
        item = val_ds[j]
        p = item.get("wf_File_Path")
        ok = bool(p) and not pd.isna(p) and os.path.exists(str(p).strip())
        print(f"    [{j}] exists={ok}  {str(p)[:100]}...")
    loader = DataLoader(
        val_ds,
        batch_size=min(8, len(val_ds)),
        shuffle=False,
        collate_fn=lambda b: {
            "signal": torch.stack([x["signal"] for x in b]),
            "label": torch.tensor([x["label"] for x in b], dtype=torch.long),
        },
    )

    model = ECGClassificationBaseline(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        ecg_ckpt_path=ckpt_enc,
        freeze_encoder=FREEZE_ENCODER,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    ).to(device)
    model.eval()

    if model_ckpt and model_ckpt.is_file():
        sd = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        print(f"  Loaded trained weights: {model_ckpt}")
    else:
        print("  No ECG best.pt — forward pass uses random head (+ frozen encoder + LoRA init).")

    batch = next(iter(loader))
    x = batch["signal"].to(device)
    labels = batch["label"].to(device)
    # Per-sample std (z-scored ECG should be ~1 scale; ~0 ⇒ missing file → zeros)
    st = x.std(dim=(1, 2)).cpu().tolist()
    print(f"  Per-sample signal std (first val batch): {[round(e, 4) for e in st]}")
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

    if x.size(0) > 1:
        spread = (logits - logits[0:1]).abs().max().item()
        print(f"  Logits max |Δ| vs batch[0] (if >0, model responds to differing inputs): {spread:.6f}")

    names = ["Severe", "Moderate", "Mild"]
    print(f"  Batch shape: {tuple(x.shape)}  labels: {labels.cpu().tolist()}")
    for i in range(min(4, x.size(0))):
        p = prob[i].cpu().numpy().tolist()
        print(
            f"    [{i}] true={names[labels[i].item()]}  pred={names[pred[i].item()]}  "
            f"softmax={[round(t, 4) for t in p]}"
        )


def _run_cxr_predictions(
    csv_path: str,
    cxr_root: str,
    metadata_path: str,
    vit_path: str,
    model_ckpt: Path | None,
    device: torch.device,
    seed: int,
) -> None:
    from cxr_classification.config import (
        DATA_CSV,
        CXR_ROOT,
        METADATA_PATH,
        VIT_PATH,
        TRAIN_SPLIT,
        VAL_SPLIT,
        NUM_CLASSES,
        HIDDEN_DIM,
        FREEZE_ENCODER,
    )
    from cxr_classification.dataset import CXRClassificationDataset
    from cxr_classification.model import CXRClassificationBaseline

    csv_path = csv_path or DATA_CSV
    if not csv_path or not os.path.isfile(csv_path):
        print("  SKIP: no valid CXR CSV for forward pass.")
        return
    cxr_root = cxr_root or CXR_ROOT
    meta = metadata_path or METADATA_PATH
    vit_path = vit_path or VIT_PATH

    full_ds = CXRClassificationDataset(
        csv_path=csv_path,
        cxr_root=cxr_root,
        metadata_path=meta if os.path.isfile(meta) else None,
        split="train",
        imagenet_normalize=True,
    )
    y = full_ds.df["p2f_class"].values
    ts = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, TRAIN_SPLIT, VAL_SPLIT, ts, seed
    )
    shared_kw = dict(
        df=full_ds.df,
        cxr_root=cxr_root,
        metadata_path=None,
        imagenet_normalize=True,
    )
    idx_v = idx_val[: min(32, len(idx_val))]
    val_ds = CXRClassificationDataset(split="val", indices=idx_v, **shared_kw)
    loader = DataLoader(
        val_ds,
        batch_size=min(8, len(val_ds)),
        shuffle=False,
        collate_fn=lambda b: {
            "cxr": torch.stack([x["cxr"] for x in b]),
            "label": torch.tensor([x["label"] for x in b], dtype=torch.long),
        },
    )

    model = CXRClassificationBaseline(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        vit_path=vit_path,
        freeze_encoder=FREEZE_ENCODER,
    ).to(device)
    model.eval()

    if model_ckpt and model_ckpt.is_file():
        sd = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        print(f"  Loaded trained weights: {model_ckpt}")
    else:
        print("  No CXR best.pt — forward pass uses random head (+ frozen ViT).")

    batch = next(iter(loader))
    x = batch["cxr"].to(device)
    labels = batch["label"].to(device)
    energy = x.abs().mean(dim=(1, 2, 3)).cpu().tolist()
    print(f"  Per-sample |cxr| mean (first batch): {[round(e, 6) for e in energy]}")
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

    if x.size(0) > 1:
        spread = (logits - logits[0:1]).abs().max().item()
        print(f"  Logits max |Δ| vs batch[0]: {spread:.6f}")

    names = ["Severe", "Moderate", "Mild"]
    print(f"  Batch shape: {tuple(x.shape)}  tensor min/max: {x.min().item():.4f} / {x.max().item():.4f}")
    print(f"  labels: {labels.cpu().tolist()}")
    for i in range(min(4, x.size(0))):
        p = prob[i].cpu().numpy().tolist()
        print(
            f"    [{i}] true={names[labels[i].item()]}  pred={names[pred[i].item()]}  "
            f"softmax={[round(t, 4) for t in p]}"
        )


def main():
    ap = argparse.ArgumentParser(description="Verify ECG/CXR input loading and predictions.")
    ap.add_argument("--modality", choices=["ecg", "cxr", "both"], default="both")
    ap.add_argument("--ecg_csv", default=None, help="Default: ECGUni config DATA_CSV")
    ap.add_argument("--cxr_csv", default=None, help="Default: CXR config DATA_CSV")
    ap.add_argument("--cxr_root", default=None)
    ap.add_argument("--metadata_path", default=None)
    ap.add_argument("--ecg_ckpt", default=None, help="Encoder checkpoint for SignalEncoder")
    ap.add_argument("--vit_path", default=None)
    ap.add_argument("--ecg_model_ckpt", default=None, help="Full model best.pt from training")
    ap.add_argument("--cxr_model_ckpt", default=None)
    ap.add_argument("--sample_max", type=int, default=None, help="Cap rows for path scan (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    exp = _BASE
    default_ecg_pt = exp / "ECGUni" / "output" / "best.pt"
    default_cxr_pt = exp / "CXRUni" / "cxr_classification" / "output" / "best.pt"

    if args.modality in ("ecg", "both"):
        from ECGUni.config import DATA_CSV as DEF_ECG

        csv_e = args.ecg_csv or DEF_ECG
        print("=== ECG: path / load stats ===")
        if not os.path.isfile(csv_e):
            print(f"  SKIP: CSV not found: {csv_e}")
        else:
            df = pd.read_csv(csv_e, low_memory=False)
            df = df[df["p2f_class"].notna()].copy()
            st = _ecg_stats(df, args.sample_max)
            print(f"  CSV rows (with label): {len(df)}")
            print(f"  Rows scanned: {st['rows_scanned']}")
            print(f"  wf path exists (.hea ok): {st['path_exists_or_hea']} / {st['rows_scanned']}")
            print(f"  Nonzero waveform after load_ecg: {st['nonzero_signal_after_load']} / {st['rows_scanned']}")
            print(f"  Example per-lead z-score (first good row): mean={st['sample_zscore_mean']}, std={st['sample_zscore_std']}")

        print("\n=== ECG: forward pass (softmax) ===")
        if not os.path.isfile(csv_e):
            print("  SKIP forward: CSV missing.")
        else:
            try:
                _run_ecg_predictions(
                    csv_e,
                    args.ecg_ckpt,
                    Path(args.ecg_model_ckpt) if args.ecg_model_ckpt else default_ecg_pt,
                    device,
                    args.seed,
                )
            except Exception as e:
                print(f"  ECG forward failed: {e}")
                raise

    if args.modality in ("cxr", "both"):
        from cxr_classification.config import DATA_CSV as DEF_CXR, CXR_ROOT, METADATA_PATH

        csv_c = args.cxr_csv or DEF_CXR
        root = args.cxr_root or CXR_ROOT
        meta = args.metadata_path or METADATA_PATH
        print("\n=== CXR: path stats ===")
        if not os.path.isfile(csv_c):
            print(f"  SKIP: CSV not found: {csv_c}")
        else:
            full_ds = CXRClassificationDataset(
                csv_path=csv_c,
                cxr_root=root,
                metadata_path=meta if os.path.isfile(meta) else None,
                split="train",
            )
            st = _cxr_stats(full_ds.df, root, args.sample_max)
            print(f"  CSV rows (with label): {len(full_ds.df)}")
            print(f"  Rows scanned: {st['rows_scanned']}")
            print(f"  JPG found at constructed path: {st['jpg_found']} / {st['rows_scanned']}")
            print(f"  Empty path (missing subj/study/dicom): {st['empty_constructed_path']}")

        print("\n=== CXR: forward pass (softmax) ===")
        if not os.path.isfile(csv_c):
            print("  SKIP forward: CSV missing.")
        else:
            try:
                _run_cxr_predictions(
                    csv_c,
                    root,
                    meta,
                    args.vit_path,
                    Path(args.cxr_model_ckpt) if args.cxr_model_ckpt else default_cxr_pt,
                    device,
                    args.seed,
                )
            except Exception as e:
                print(f"  CXR forward failed: {e}")
                raise

    print("\nDone.")


if __name__ == "__main__":
    main()
