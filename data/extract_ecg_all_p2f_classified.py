#!/usr/bin/env python
"""
Extract ALL ECG samples in the lookback window for rows with ECG_signal=1.

Original enrich logic picks only the most recent ECG. This script expands to one row per ECG
in the window [row_time-18h, row_time-6h] within admission.

Output: p2f_ecg_all_classified.csv with p2f_class (0=Severe, 1=Moderate, 2=Mild).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

ADMISSIONS_PATH = "/hpc/group/kamaleswaranlab/mimic_iv/mimic-iv-3.1-decompress/hosp/admissions.csv"
WAVEFORM_CSV = "/hpc/group/kamaleswaranlab/Waveform/MIMIC_waveform/MatchedFilePath/MIMIC4MathedPath.csv"
LOOKBACK_MIN_H = 6
LOOKBACK_MAX_H = 18
ECG_MATCH_COLS = [
    "wf_Study_ID", "wf_File_Name", "wf_Base_Time", "wf_End_Time",
    "wf_DurationMin", "wf_sigLen", "wf_ECG_Time", "wf_stayHours", "wf_File_Path"
]
WF_COLS_RAW = ["Study_ID", "File_Name", "Base_Time", "End_Time", "DurationMin", "sigLen", "ECG_Time", "stayHours", "File_Path"]


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


def _find_datetime_col(df):
    for name in ["index", "recorded_time", "supertable_datetime"]:
        if name in df.columns:
            valid = pd.to_datetime(df[name], errors="coerce").notna().sum()
            if valid > len(df) * 0.5:
                return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="p2f_vent_fio2_enriched.csv", help="Input CSV (enriched)")
    parser.add_argument("--output", default="p2f_ecg_all_classified.csv", help="Output CSV")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else data_dir / args.input
    output_path = Path(args.output) if Path(args.output).is_absolute() else data_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print("Loading enriched CSV...", flush=True)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    df = df[df["ECG_signal"] == 1].copy()
    print(f"  ECG_signal=1: {len(df):,}")

    dt_col = _find_datetime_col(df)
    if not dt_col:
        raise ValueError("Could not find datetime column")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df[df[dt_col].notna()].copy()

    df["p2f_vent_fio2"] = pd.to_numeric(df["p2f_vent_fio2"], errors="coerce")
    df["p2f_class"] = df["p2f_vent_fio2"].apply(classify_p2f)
    df = df[df["p2f_class"].notna()].copy()
    df["p2f_class"] = df["p2f_class"].astype(int)
    print(f"  After p2f_class filter (valid ARDS range): {len(df):,}")

    print("Loading admissions...", flush=True)
    admissions = pd.read_csv(ADMISSIONS_PATH, parse_dates=["admittime", "dischtime"],
                             usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    adm_detail = admissions.drop_duplicates(subset=["hadm_id"], keep="first").set_index("hadm_id")

    print("Loading waveform...", flush=True)
    wf = pd.read_csv(WAVEFORM_CSV)
    wf["wf_Base_Time"] = pd.to_datetime(wf["Base_Time"], errors="coerce")
    wf = wf[wf["wf_Base_Time"].notna()].copy()
    wf["Subject_ID"] = pd.to_numeric(wf["Subject_ID"], errors="coerce")
    wf = wf[wf["Subject_ID"].notna()]
    wf["Subject_ID"] = wf["Subject_ID"].astype(int)
    wf_rename = {c: "wf_" + c for c in WF_COLS_RAW if c in wf.columns and c != "Base_Time"}
    wf = wf.rename(columns=wf_rename)
    wf_by_subject = wf.groupby("Subject_ID")

    def get_all_ecgs(hadm_id, row_time):
        """All ECGs in [row_time-18h, row_time-6h], same subject, within admission."""
        if pd.isna(row_time):
            return []
        hid = int(hadm_id) if pd.notna(hadm_id) else None
        if hid is None or hid not in adm_detail.index:
            return []
        subj_row = adm_detail.loc[hid]
        sid = subj_row["subject_id"]
        adm_start = subj_row["admittime"]
        adm_end = subj_row["dischtime"]
        lo = row_time - pd.Timedelta(hours=LOOKBACK_MAX_H)
        hi = row_time - pd.Timedelta(hours=LOOKBACK_MIN_H)
        try:
            sub = wf_by_subject.get_group(int(sid)).copy()
        except KeyError:
            return []
        bt = sub["wf_Base_Time"]
        if bt.dtype == object or bt.dtype.kind == "O":
            bt = pd.to_datetime(bt, errors="coerce")
        mask = bt.notna() & (bt >= lo) & (bt <= hi) & (bt >= adm_start) & (bt <= adm_end)
        cand = sub.loc[mask]
        return [cand.iloc[i].to_dict() for i in range(len(cand))]

    base_cols = [c for c in df.columns if c not in ECG_MATCH_COLS]
    rows_out = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Expanding"):
        ecgs = get_all_ecgs(row["hadm_id"], row[dt_col])
        if not ecgs:
            continue
        for ecg in ecgs:
            out = {c: row.get(c) for c in base_cols}
            for c in ECG_MATCH_COLS:
                out[c] = ecg.get(c, row.get(c))
            rows_out.append(out)

    out_df = pd.DataFrame(rows_out)
    if len(out_df) == 0:
        raise RuntimeError("No output rows. Check hadm_id and waveform data.")

    print(f"\nOutput: {len(out_df):,} rows (one per ECG in window)")
    for c, name in [(0, "Severe"), (1, "Moderate"), (2, "Mild")]:
        n = (out_df["p2f_class"] == c).sum()
        print(f"  {name}: {n:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
