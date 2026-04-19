#!/usr/bin/env python
"""
Build multimodal rows: same supertable row (``index``) with both CXR (dicom) and ECG (wf_*).

Inner-joins ``p2f_cxr_classified.csv`` with ``p2f_ecg_all_classified.csv`` on ``index``.
ECG-side ``wf_*`` columns are taken from the ECG extract; CXR-side rows drop overlapping ``wf_*``
so each merged row has a single ``wf_File_Path`` for waveform loading.

Output: ``p2f_ecg_cxr_multimodal.csv`` (same schema width as CXR row + wf columns from ECG).
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr_csv", default="p2f_cxr_classified.csv", help="CXR classified CSV (under data/ or absolute)")
    parser.add_argument("--ecg_csv", default="p2f_ecg_all_classified.csv", help="ECG all-window classified CSV")
    parser.add_argument("--output", default="p2f_ecg_cxr_multimodal.csv", help="Output multimodal CSV")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    cxr_path = Path(args.cxr_csv) if Path(args.cxr_csv).is_absolute() else data_dir / args.cxr_csv
    ecg_path = Path(args.ecg_csv) if Path(args.ecg_csv).is_absolute() else data_dir / args.ecg_csv
    out_path = Path(args.output) if Path(args.output).is_absolute() else data_dir / args.output

    if not cxr_path.exists():
        raise FileNotFoundError(f"CXR CSV not found: {cxr_path}")
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG CSV not found: {ecg_path}")

    cxr = pd.read_csv(cxr_path, low_memory=False)
    ecg = pd.read_csv(ecg_path, low_memory=False)
    print(f"Loaded CXR rows: {len(cxr):,}  ECG rows: {len(ecg):,}")

    wf_cols = [c for c in ecg.columns if c.startswith("wf_")]
    if not wf_cols:
        raise ValueError("ECG CSV has no wf_* columns.")

    cxr_drop = cxr.drop(columns=[c for c in wf_cols if c in cxr.columns], errors="ignore")
    merged = cxr_drop.merge(ecg[["index"] + wf_cols], on="index", how="inner")
    print(f"Inner join on index: {len(merged):,} multimodal rows")

    if "p2f_class" in merged.columns:
        merged = merged[merged["p2f_class"].notna()].copy()
        merged["p2f_class"] = merged["p2f_class"].astype(int)

    if "wf_File_Path" in merged.columns:
        n_wf = merged["wf_File_Path"].notna() & (merged["wf_File_Path"].astype(str).str.strip() != "")
        print(f"  Non-empty wf_File_Path: {int(n_wf.sum()):,} / {len(merged):,}")

    for c, name in [(0, "Severe"), (1, "Moderate"), (2, "Mild")]:
        if "p2f_class" in merged.columns:
            n = (merged["p2f_class"] == c).sum()
            print(f"  {name}: {n:,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
