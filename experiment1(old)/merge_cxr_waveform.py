#!/usr/bin/env python
"""
Merge CXR+supertable data with ECG waveform data.
Match on: same patient (subject_id) and same time (year, month, day, hour).
Add waveform columns to matched rows; save to new CSV and PKL.
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

CXR_CSV = "/work/gs285/Waveform_CXR_EHR/cxr_supertable_matched.csv"
WAVEFORM_CSV = "/hpc/group/kamaleswaranlab/Waveform/MIMIC_waveform/MatchedFilePath/MIMIC4MathedPath.csv"
OUTPUT_CSV = "/work/gs285/Waveform_CXR_EHR/cxr_supertable_waveform_matched.csv"
OUTPUT_PKL = "/work/gs285/Waveform_CXR_EHR/cxr_supertable_waveform_matched.pkl"

# Waveform columns to add (with wf_ prefix to avoid conflicts)
WF_COLS = [
    "Study_ID", "File_Name", "Base_Time", "End_Time", "DurationMin", "sigLen",
    "ECG_Time", "stayHours", "File_Path"
]
WF_PREFIX = "wf_"


def _datetime_to_date_int_hour(series):
    """Same as run_full_match.py: _date_int=YYYYMMDD, _hour=hour."""
    dt = pd.to_datetime(series, errors="coerce")
    date_int = dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day
    hour = dt.dt.hour
    return date_int, hour


def main():
    print("Loading CXR + supertable data...", flush=True)
    cxr = pd.read_csv(CXR_CSV, low_memory=False)
    print(f"  CXR rows: {len(cxr):,}")

    print("Loading waveform data...", flush=True)
    wf = pd.read_csv(WAVEFORM_CSV)
    print(f"  Waveform rows: {len(wf):,}")

    # CXR: subject_id + supertable_datetime (same format as run_full_match.py)
    cxr_date_int, cxr_hour = _datetime_to_date_int_hour(cxr["supertable_datetime"])
    sid = pd.to_numeric(cxr["subject_id"], errors="coerce").fillna(-999999).astype("int64")
    cxr = cxr.assign(_date_int=cxr_date_int, _hour=cxr_hour, _sid=sid)

    # Waveform: Subject_ID + Base_Time
    wf_date_int, wf_hour = _datetime_to_date_int_hour(wf["Base_Time"])
    wf = wf.assign(_date_int=wf_date_int, _hour=wf_hour, _sid=wf["Subject_ID"].astype(int))

    # Keep all waveforms per (subject, date_int, hour) - no dedup
    # (one CXR can match multiple waveforms in the same hour)
    wf_rename = {c: WF_PREFIX + c for c in WF_COLS if c in wf.columns}
    wf = wf.rename(columns=wf_rename)

    merge_cols = ["_sid", "_date_int", "_hour"]
    wf_for_merge = wf[[c for c in merge_cols + list(wf_rename.values()) if c in wf.columns]]

    print("Merging on (subject_id, _date_int, _hour) [same as run_full_match.py]...", flush=True)
    merged = cxr.merge(
        wf_for_merge,
        on=merge_cols,
        how="inner",  # keep only rows with waveform match
        suffixes=("", "_wf")
    )

    # Drop helper columns
    for c in merge_cols + ["_sid"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    n_with_wf = len(merged)
    print(f"  Total rows (CXR-waveform pairs): {n_with_wf:,}")

    print(f"Saving CSV: {OUTPUT_CSV}", flush=True)
    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"Saving PKL: {OUTPUT_PKL}", flush=True)
    merged.to_pickle(OUTPUT_PKL)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
