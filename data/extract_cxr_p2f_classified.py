#!/usr/bin/env python
"""
Extract rows with CXR_signal=1 from p2f_vent_fio2_enriched.csv and add ARDS severity class.

ARDS classification (PaO2/FiO2 ratio):
  - Mild:     200 <= p2f <= 300
  - Moderate: 100 <= p2f < 200
  - Severe:   p2f < 100

Rows with p2f > 300 or NaN are excluded (not in ARDS severity range).
Output: p2f_cxr_classified.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def classify_p2f(value):
    """Map p2f_vent_fio2 to ARDS class: 0=Severe, 1=Moderate, 2=Mild."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(v) or not np.isfinite(v):
        return np.nan
    if v < 100:
        return 0  # Severe
    if v < 200:
        return 1  # Moderate
    if v <= 300:
        return 2  # Mild
    return np.nan  # > 300: exclude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="p2f_vent_fio2_enriched.csv",
                        help="Input CSV path (relative to data/ or absolute)")
    parser.add_argument("--output", default="p2f_cxr_classified.csv",
                        help="Output CSV path (relative to data/ or absolute)")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else data_dir / args.input
    output_path = Path(args.output) if Path(args.output).is_absolute() else data_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {input_path.name}")

    # Filter: CXR_signal == 1
    df = df[df["CXR_signal"] == 1].copy()
    print(f"After CXR_signal=1 filter: {len(df):,} rows")

    # Ensure p2f is numeric
    df["p2f_vent_fio2"] = pd.to_numeric(df["p2f_vent_fio2"], errors="coerce")

    # Add classification
    df["p2f_class"] = df["p2f_vent_fio2"].apply(classify_p2f)

    # Keep only rows with valid class (exclude > 300 and NaN)
    before = len(df)
    df = df[df["p2f_class"].notna()].copy()
    df["p2f_class"] = df["p2f_class"].astype(int)
    excluded = before - len(df)
    print(f"Excluded {excluded:,} rows (p2f>300 or NaN). Remaining: {len(df):,}")

    # Ensure dicom_id present for CXR path
    df = df[df["dicom_id"].notna()].copy()
    df = df[df["dicom_id"].astype(str).str.strip() != ""].copy()
    print(f"After dicom_id filter: {len(df):,} rows")

    # Class distribution
    for c, name in [(0, "Severe"), (1, "Moderate"), (2, "Mild")]:
        n = (df["p2f_class"] == c).sum()
        print(f"  {name}: {n:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
