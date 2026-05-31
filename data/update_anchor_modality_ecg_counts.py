#!/usr/bin/env python
"""
Refresh ECG_window_count and ECG_signal on p2f_or_s2f_anchor_modality_window.csv
from p2f_or_s2f_ecg_catalog_labeled.csv (chunked; low memory).

Usage:
  python data/update_anchor_modality_ecg_counts.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = BASE / "data" / "p2f_or_s2f_anchor_modality_window.csv"
DEFAULT_LABELED = BASE / "data" / "p2f_or_s2f_ecg_catalog_labeled.csv"
LOOKBACK_MIN_H = 12
LOOKBACK_MAX_H = 24
CHUNK = 50_000


def _counts_from_labeled(labeled_path: Path, lb_min: float, lb_max: float) -> dict[str, int]:
    usecols = ["anchor_index", "anchor_hadm_id", "hours_ecg_to_anchor", "wf_File_Name"]
    ecg_ids: dict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(labeled_path, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk["anchor_index"] = chunk["anchor_index"].astype(str).str.strip()
        chunk = chunk[chunk["anchor_index"] != ""].copy()
        hrs = pd.to_numeric(chunk["hours_ecg_to_anchor"], errors="coerce")
        chunk = chunk[(hrs > lb_min) & (hrs <= lb_max)].copy()
        if chunk.empty:
            continue
        chunk["_mk"] = chunk["anchor_hadm_id"].astype(str) + "|" + chunk["anchor_index"]
        for mk, fn in zip(chunk["_mk"], chunk["wf_File_Name"].astype(str)):
            ecg_ids[mk].add(fn)
    return {k: len(v) for k, v in ecg_ids.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-csv", default=str(DEFAULT_ANCHOR))
    ap.add_argument("--from-labeled", default=str(DEFAULT_LABELED))
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--lookback-min-hours", type=float, default=LOOKBACK_MIN_H)
    ap.add_argument("--lookback-max-hours", type=float, default=LOOKBACK_MAX_H)
    args = ap.parse_args()

    anchor_path = Path(args.anchor_csv)
    labeled_path = Path(args.from_labeled)
    out_path = Path(args.output) if args.output else anchor_path
    tmp_path = out_path.with_suffix(".ecg_counts_tmp.csv")

    if not anchor_path.is_file():
        print(f"Anchor CSV not found: {anchor_path}")
        return 1
    if not labeled_path.is_file():
        print(f"Labeled ECG CSV not found: {labeled_path}")
        return 1

    print(f"Aggregating ECG counts from: {labeled_path}", flush=True)
    count_map = _counts_from_labeled(labeled_path, args.lookback_min_hours, args.lookback_max_hours)
    print(f"  Anchors with ECG in window: {len(count_map):,}", flush=True)

    print(f"Updating anchor table (chunked): {anchor_path}", flush=True)
    n_sig = 0
    n_rows = 0
    first = True
    if tmp_path.exists():
        tmp_path.unlink()

    for chunk in pd.read_csv(anchor_path, chunksize=CHUNK, low_memory=False):
        n_rows += len(chunk)
        chunk["_mk"] = chunk["hadm_id"].astype(str) + "|" + chunk["index"].astype(str)
        chunk["ECG_window_count"] = chunk["_mk"].map(count_map).fillna(0).astype(np.int32)
        chunk["ECG_signal"] = (chunk["ECG_window_count"] > 0).astype(np.int8)
        n_sig += int((chunk["ECG_signal"] == 1).sum())
        chunk = chunk.drop(columns=["_mk"])
        chunk.to_csv(tmp_path, mode="w" if first else "a", header=first, index=False)
        first = False

    tmp_path.replace(out_path)
    print(f"\nSaved → {out_path}")
    print(f"  Rows: {n_rows:,}")
    print(f"  ECG_signal=1: {n_sig:,} ({100 * n_sig / max(n_rows, 1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
