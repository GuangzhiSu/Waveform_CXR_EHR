#!/usr/bin/env python
"""
Extract EHR rows with valid p2f in ARDS range and add p2f_class.
EHR is available for all rows; no CXR/ECG filter needed.
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
        return 0
    if v < 200:
        return 1
    if v <= 300:
        return 2
    return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="p2f_vent_fio2_enriched.csv", help="Input CSV")
    parser.add_argument("--output", default="p2f_ehr_classified.csv", help="Output CSV")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else data_dir / args.input
    output_path = Path(args.output) if Path(args.output).is_absolute() else data_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    df["p2f_vent_fio2"] = pd.to_numeric(df["p2f_vent_fio2"], errors="coerce")
    df["p2f_class"] = df["p2f_vent_fio2"].apply(classify_p2f)
    df = df[df["p2f_class"].notna()].copy()
    df["p2f_class"] = df["p2f_class"].astype(int)
    print(f"After p2f_class filter (ARDS range): {len(df):,}")

    for c, name in [(0, "Severe"), (1, "Moderate"), (2, "Mild")]:
        n = (df["p2f_class"] == c).sum()
        print(f"  {name}: {n:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
