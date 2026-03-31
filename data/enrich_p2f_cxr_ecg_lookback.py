#!/usr/bin/env python
"""
Enrich input CSV with CXR and ECG matches from 6–12 hours before each row.

Uses raw sources only (no pre-built CSVs):
  - METADATA_PATH: MIMIC-CXR (subject_id, StudyDate, StudyTime, dicom_id)
  - ADMISSIONS_PATH: subject_id ↔ hadm_id, admittime, dischtime
  - WAVEFORM_CSV: Subject_ID, Base_Time, File_Path, etc.

Patient identity: hadm_id → subject_id via admissions. Same logic as run_full_match:
  meta.merge(admissions, on="subject_id") + admittime <= CXR_time <= dischtime → hadm_id per CXR.

Match logic (lookback [ref-18h, ref-6h]):
  - CXR: from metadata+admissions, same hadm_id, CXR time in window
  - ECG: from waveform, same subject_id (from admissions), Base_Time in window and within admission

Usage:
  python enrich_p2f_cxr_ecg_lookback.py [--input INPUT_CSV] [--output OUTPUT_CSV]
"""
import argparse
import csv
import pandas as pd
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# Same as experiment1/run_full_match.py
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
ADMISSIONS_PATH = "/hpc/group/kamaleswaranlab/mimic_iv/mimic-iv-3.1-decompress/hosp/admissions.csv"
WAVEFORM_CSV = "/hpc/group/kamaleswaranlab/Waveform/MIMIC_waveform/MatchedFilePath/MIMIC4MathedPath.csv"

BASE = "/work/gs285/Waveform_CXR_EHR"
DEFAULT_INPUT = f"{BASE}/p2f_vent_fio2_valid_rows.csv"
DEFAULT_OUTPUT = f"{BASE}/p2f_vent_fio2_enriched.csv"

# Output column names
CXR_MATCH_COLS = ["dicom_id", "subject_id", "hadm_id", "supertable_datetime"]
ECG_MATCH_COLS = [
    "wf_Study_ID", "wf_File_Name", "wf_Base_Time", "wf_End_Time",
    "wf_DurationMin", "wf_sigLen", "wf_ECG_Time", "wf_stayHours", "wf_File_Path"
]
WF_COLS_RAW = ["Study_ID", "File_Name", "Base_Time", "End_Time", "DurationMin", "sigLen", "ECG_Time", "stayHours", "File_Path"]
LOOKBACK_MIN_H = 6   # 上界：ref 前 6 小时
LOOKBACK_MAX_H = 18  # 下界：ref 前 18 小时 → 窗口 [ref-18h, ref-6h]


def _find_datetime_col(df):
    """Find column to use as row reference time."""
    for name in ["index", "recorded_time", "supertable_datetime"]:
        if name in df.columns:
            valid = pd.to_datetime(df[name], errors="coerce").notna().sum()
            if valid > len(df) * 0.5:
                return name
    for c in df.columns:
        if "time" in c.lower() or "datetime" in c.lower() or "date" in c.lower():
            if pd.to_datetime(df[c], errors="coerce").notna().sum() > len(df) * 0.5:
                return c
    return None


def _cxr_datetime(row):
    """Convert StudyDate, StudyTime to datetime (same as run_full_match)."""
    sd, st = int(row["StudyDate"]), row["StudyTime"]
    y, m, d = sd // 10000, (sd % 10000) // 100, sd % 100
    h = int(st // 10000)
    return pd.Timestamp(year=y, month=m, day=d, hour=h, minute=0, second=0)


def _csv_val(v):
    """Convert value for CSV write: NaN/None->'', else str."""
    if v is None or (hasattr(v, "__float__") and pd.isna(v)):
        return ""
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input CSV with EHR rows")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output enriched CSV")
    ap.add_argument("--datetime-col", default=None, help="Column name for row reference time")
    ap.add_argument("--hadm-col", default="hadm_id", help="Column for hospital admission ID")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        print("Run extract_p2f_rows.py first to generate p2f_vent_fio2_valid_rows.csv")
        return 1

    print("Loading input CSV...", flush=True)
    df = pd.read_csv(input_path, low_memory=False)
    n_in = len(df)
    print(f"  Rows: {n_in:,}")

    dt_col = args.datetime_col or _find_datetime_col(df)
    if not dt_col or dt_col not in df.columns:
        print(f"Could not find datetime column. Available: {list(df.columns[:20])}...")
        return 1
    print(f"  Datetime column: {dt_col}")

    if args.hadm_col not in df.columns:
        print(f"Column {args.hadm_col} not found.")
        return 1

    print("Loading admissions (hadm_id ↔ subject_id, admittime, dischtime)...", flush=True)
    admissions = pd.read_csv(ADMISSIONS_PATH, parse_dates=["admittime", "dischtime"],
                             usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
    hadm_to_subject = admissions.drop_duplicates(subset=["hadm_id"], keep="first").set_index("hadm_id")["subject_id"].to_dict()
    adm_detail = admissions.drop_duplicates(subset=["hadm_id"], keep="first").set_index("hadm_id")
    print(f"  Admissions: {len(adm_detail):,}")

    print("Loading CXR metadata + admissions (same logic as run_full_match)...", flush=True)
    meta = pd.read_csv(METADATA_PATH, usecols=["subject_id", "StudyDate", "StudyTime", "dicom_id"])
    meta["_cxr_dt"] = meta.apply(_cxr_datetime, axis=1)
    meta_adm = meta.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime"]], on="subject_id")
    meta_adm = meta_adm[(meta_adm["admittime"] <= meta_adm["_cxr_dt"]) & (meta_adm["dischtime"] >= meta_adm["_cxr_dt"])]
    cxr = meta_adm.drop_duplicates(subset=["dicom_id"], keep="first")[
        ["subject_id", "dicom_id", "hadm_id", "_cxr_dt"]
    ].rename(columns={"_cxr_dt": "supertable_datetime"})
    cxr["hadm_id"] = cxr["hadm_id"].astype(int)
    cxr_by_hadm = cxr.groupby("hadm_id")
    print(f"  CXR rows (in admission): {len(cxr):,}")

    print("Loading waveform (Subject_ID, Base_Time, File_Path, ...)...", flush=True)
    wf = pd.read_csv(WAVEFORM_CSV)
    wf["wf_Base_Time"] = pd.to_datetime(wf["Base_Time"], errors="coerce")
    wf = wf[wf["wf_Base_Time"].notna()].copy()
    wf["Subject_ID"] = pd.to_numeric(wf["Subject_ID"], errors="coerce")
    wf = wf[wf["Subject_ID"].notna()]
    wf["Subject_ID"] = wf["Subject_ID"].astype(int)
    wf_rename = {c: "wf_" + c for c in WF_COLS_RAW if c in wf.columns and c != "Base_Time"}
    wf = wf.rename(columns=wf_rename)
    wf_by_subject = wf.groupby("Subject_ID")
    print(f"  Waveform rows: {len(wf):,}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    invalid_dt = df[dt_col].isna().sum()
    if invalid_dt > 0:
        print(f"  Warning: {invalid_dt} rows with invalid datetime", flush=True)

    def get_best_cxr(hadm_id, row_time):
        """Most recent CXR in [row_time-18h, row_time-6h], same hadm_id."""
        if pd.isna(row_time):
            return None
        hid = int(hadm_id) if pd.notna(hadm_id) else None
        if hid is None:
            return None
        lo = row_time - pd.Timedelta(hours=LOOKBACK_MAX_H)
        hi = row_time - pd.Timedelta(hours=LOOKBACK_MIN_H)
        try:
            sub = cxr_by_hadm.get_group(hid)
        except KeyError:
            return None
        mask = (sub["supertable_datetime"] >= lo) & (sub["supertable_datetime"] <= hi)
        cand = sub.loc[mask]
        if len(cand) == 0:
            return None
        best = cand.loc[cand["supertable_datetime"].idxmax()]
        return {c: best.get(c) for c in CXR_MATCH_COLS}

    def get_best_ecg(hadm_id, row_time):
        """Most recent ECG in [row_time-18h, row_time-6h], same subject, within admission."""
        if pd.isna(row_time):
            return None
        hid = int(hadm_id) if pd.notna(hadm_id) else None
        if hid is None or hid not in adm_detail.index:
            return None
        subj_row = adm_detail.loc[hid]
        sid = subj_row["subject_id"]
        adm_start = subj_row["admittime"]
        adm_end = subj_row["dischtime"]
        lo = row_time - pd.Timedelta(hours=LOOKBACK_MAX_H)
        hi = row_time - pd.Timedelta(hours=LOOKBACK_MIN_H)
        try:
            sub = wf_by_subject.get_group(int(sid)).copy()
        except KeyError:
            return None
        bt = sub["wf_Base_Time"]
        if bt.dtype == object or bt.dtype.kind == "O":
            bt = pd.to_datetime(bt, errors="coerce")
        mask = bt.notna() & (bt >= lo) & (bt <= hi) & (bt >= adm_start) & (bt <= adm_end)
        cand = sub.loc[mask]
        if len(cand) == 0:
            return None
        best = cand.loc[cand["wf_Base_Time"].idxmax()]
        return {c: best.get(c) for c in ECG_MATCH_COLS}

    out_cols = list(df.columns)
    for c in CXR_MATCH_COLS + ECG_MATCH_COLS + ["CXR_signal", "ECG_signal"]:
        if c not in out_cols:
            out_cols.append(c)

    cxr_count = 0
    ecg_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore", restval="")
        writer.writeheader()
        for idx, row in tqdm(df.iterrows(), total=n_in, desc="Enriching"):
            out = {k: row.get(k) for k in out_cols}
            hadm = row.get(args.hadm_col)
            row_time = row.get(dt_col)
            if hasattr(row_time, "to_pydatetime"):
                row_time = pd.Timestamp(row_time)

            if pd.isna(hadm):
                for c in CXR_MATCH_COLS:
                    out[c] = None
                for c in ECG_MATCH_COLS:
                    out[c] = None
                out["CXR_signal"] = 0
                out["ECG_signal"] = 0
                writer.writerow({k: _csv_val(out[k]) for k in out_cols})
                continue

            cbest = get_best_cxr(hadm, row_time)
            if cbest is not None:
                out["CXR_signal"] = 1
                cxr_count += 1
                for c in CXR_MATCH_COLS:
                    out[c] = cbest.get(c)
            else:
                out["CXR_signal"] = 0
                for c in CXR_MATCH_COLS:
                    out[c] = None
                hid = int(hadm) if pd.notna(hadm) else None
                out["subject_id"] = hadm_to_subject.get(hid) if hid else None

            ebest = get_best_ecg(hadm, row_time)
            if ebest is not None:
                out["ECG_signal"] = 1
                ecg_count += 1
                for c in ECG_MATCH_COLS:
                    out[c] = ebest.get(c)
            else:
                out["ECG_signal"] = 0
                for c in ECG_MATCH_COLS:
                    out[c] = None

            writer.writerow({k: _csv_val(out[k]) for k in out_cols})

    print(f"\nSaved {n_in:,} rows to {output_path}")
    print(f"  CXR_signal=1: {cxr_count:,}")
    print(f"  ECG_signal=1: {ecg_count:,}")
    return 0


if __name__ == "__main__":
    exit(main())
