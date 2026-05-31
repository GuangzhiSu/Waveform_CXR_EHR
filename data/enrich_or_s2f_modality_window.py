#!/usr/bin/env python
"""
Build CXR/ECG modality catalogs for the p2f_or_s2f cohort and match [t-24h, t-12h] windows.

Uses the same raw sources as enrich_p2f_cxr_ecg_lookback.py and the same acquisition-time
indexing as EHRWindowTransformer CXR/ECG window datasets:
  - CXR: MIMIC-CXR metadata + admissions → hadm_id, dicom_id, supertable_datetime
  - ECG: MIMIC waveform CSV → subject_id, wf_Base_Time (within admission)

Outputs:
  1. p2f_or_s2f_cxr_catalog.csv       — all in-admission CXRs for cohort hadm_ids
  2. p2f_or_s2f_ecg_catalog.csv       — all in-admission ECGs for cohort subject_ids
  3. p2f_or_s2f_anchor_modality_window.csv — anchor rows + window counts / signal flags
  4. (optional --write-matches) long-format CXR/ECG match tables per anchor

Usage:
  python data/enrich_or_s2f_modality_window.py
  python data/enrich_or_s2f_modality_window.py --max-rows 5000   # smoke test
  sbatch data/run_enrich_or_s2f_modality_window.sh
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


BASE = Path("/work/gs285/Waveform_CXR_EHR")
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
ADMISSIONS_PATH = "/hpc/group/kamaleswaranlab/mimic_iv/mimic-iv-3.1-decompress/hosp/admissions.csv"
WAVEFORM_CSV = "/hpc/group/kamaleswaranlab/Waveform/MIMIC_waveform/MatchedFilePath/MIMIC4MathedPath.csv"

DEFAULT_INPUT = BASE / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv"
DEFAULT_CXR_CATALOG = BASE / "data" / "p2f_or_s2f_cxr_catalog.csv"
DEFAULT_ECG_CATALOG = BASE / "data" / "p2f_or_s2f_ecg_catalog.csv"
DEFAULT_ANCHOR_OUT = BASE / "data" / "p2f_or_s2f_anchor_modality_window.csv"
DEFAULT_CXR_MATCHES = BASE / "data" / "p2f_or_s2f_cxr_window_matches.csv"
DEFAULT_ECG_MATCHES = BASE / "data" / "p2f_or_s2f_ecg_window_matches.csv"
CHECKPOINT_FILE = BASE / "data" / "enrich_or_s2f_modality_window_checkpoint.json"
PARTIAL_COUNTS_FILE = BASE / "data" / "enrich_or_s2f_modality_window_partial.npz"

LOOKBACK_MIN_H = 12
LOOKBACK_MAX_H = 24

CXR_CATALOG_COLS = ["hadm_id", "subject_id", "dicom_id", "supertable_datetime"]
ECG_CATALOG_COLS = [
    "subject_id",
    "hadm_id",
    "wf_Study_ID",
    "wf_File_Name",
    "wf_Base_Time",
    "wf_End_Time",
    "wf_DurationMin",
    "wf_sigLen",
    "wf_ECG_Time",
    "wf_stayHours",
    "wf_File_Path",
]
WF_COLS_RAW = [
    "Study_ID",
    "File_Name",
    "Base_Time",
    "End_Time",
    "DurationMin",
    "sigLen",
    "ECG_Time",
    "stayHours",
    "File_Path",
]


def _read_csv_filtered(
    path: str,
    subject_col: str,
    allowed_ids: set[int],
    parse_dates: list[str] | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Stream-read a large CSV and keep rows whose subject_col is in allowed_ids."""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=parse_dates or []):
        chunk[subject_col] = pd.to_numeric(chunk[subject_col], errors="coerce")
        chunk = chunk[chunk[subject_col].isin(allowed_ids)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _cxr_datetime(row) -> pd.Timestamp:
    sd, st = int(row["StudyDate"]), row["StudyTime"]
    y, m, d = sd // 10000, (sd % 10000) // 100, sd % 100
    h = int(st // 10000)
    return pd.Timestamp(year=y, month=m, day=d, hour=h, minute=0, second=0)


def _csv_val(v):
    if v is None or (hasattr(v, "__float__") and pd.isna(v)):
        return ""
    return v


def _load_checkpoint() -> dict:
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=0)


def _build_cxr_catalog(hadm_ids: set[int]) -> pd.DataFrame:
    print("Loading admissions...", flush=True)
    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        parse_dates=["admittime", "dischtime"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
    admissions["hadm_id"] = admissions["hadm_id"].astype(np.int64)

    cohort_hadm = hadm_ids.intersection(set(admissions["hadm_id"].unique()))
    print(f"  Cohort hadm_ids: {len(hadm_ids):,} (in admissions: {len(cohort_hadm):,})", flush=True)

    cohort_subjects = set(
        admissions.loc[admissions["hadm_id"].isin(cohort_hadm), "subject_id"].astype(np.int64).unique()
    )

    print("Loading CXR metadata (filtered by cohort subjects)...", flush=True)
    meta = _read_csv_filtered(METADATA_PATH, "subject_id", cohort_subjects)
    if meta.empty:
        return pd.DataFrame(columns=CXR_CATALOG_COLS)
    meta["_cxr_dt"] = meta.apply(_cxr_datetime, axis=1)
    meta_adm = meta.merge(admissions, on="subject_id")
    meta_adm = meta_adm[
        (meta_adm["admittime"] <= meta_adm["_cxr_dt"]) & (meta_adm["dischtime"] >= meta_adm["_cxr_dt"])
    ]
    meta_adm = meta_adm[meta_adm["hadm_id"].isin(cohort_hadm)]
    cxr = (
        meta_adm.drop_duplicates(subset=["dicom_id"], keep="first")[
            ["subject_id", "dicom_id", "hadm_id", "_cxr_dt"]
        ]
        .rename(columns={"_cxr_dt": "supertable_datetime"})
        .sort_values(["hadm_id", "supertable_datetime"])
        .reset_index(drop=True)
    )
    cxr["hadm_id"] = cxr["hadm_id"].astype(np.int64)
    cxr["subject_id"] = cxr["subject_id"].astype(np.int64)
    print(f"  CXR catalog rows: {len(cxr):,}, hadms={cxr['hadm_id'].nunique():,}", flush=True)
    return cxr


def _build_ecg_catalog(subject_ids: set[int], hadm_ids: set[int]) -> pd.DataFrame:
    print("Loading admissions for ECG admission bounds...", flush=True)
    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        parse_dates=["admittime", "dischtime"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
    admissions["hadm_id"] = admissions["hadm_id"].astype(np.int64)
    admissions["subject_id"] = admissions["subject_id"].astype(np.int64)
    adm = admissions[admissions["hadm_id"].isin(hadm_ids) | admissions["subject_id"].isin(subject_ids)].copy()

    print("Loading waveform (filtered by cohort subjects)...", flush=True)
    wf = _read_csv_filtered(WAVEFORM_CSV, "Subject_ID", subject_ids)
    if wf.empty:
        return pd.DataFrame(columns=ECG_CATALOG_COLS)
    wf["wf_Base_Time"] = pd.to_datetime(wf["Base_Time"], errors="coerce")
    wf = wf[wf["wf_Base_Time"].notna()].copy()
    wf["Subject_ID"] = pd.to_numeric(wf["Subject_ID"], errors="coerce")
    wf = wf.dropna(subset=["Subject_ID"])
    wf["Subject_ID"] = wf["Subject_ID"].astype(np.int64)
    wf = wf[wf["Subject_ID"].isin(subject_ids)]
    wf_rename = {c: "wf_" + c for c in WF_COLS_RAW if c in wf.columns and c != "Base_Time"}
    wf = wf.rename(columns=wf_rename)
    wf = wf.rename(columns={"Subject_ID": "subject_id"})

    wf = wf.merge(adm, left_on="subject_id", right_on="subject_id", how="inner")
    wf = wf[(wf["wf_Base_Time"] >= wf["admittime"]) & (wf["wf_Base_Time"] <= wf["dischtime"])]
    keep = [c for c in ECG_CATALOG_COLS if c in wf.columns]
    ecg = wf[keep].drop_duplicates().sort_values(["subject_id", "wf_Base_Time"]).reset_index(drop=True)
    print(
        f"  ECG catalog rows: {len(ecg):,}, subjects={ecg['subject_id'].nunique():,}",
        flush=True,
    )
    return ecg


def _index_by_group(df: pd.DataFrame, group_col: str, time_col: str) -> dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Map group_id -> (times_ns int64 array, row positions, sub-frame)."""
    out: dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    if df.empty:
        return out
    sub = df.copy()
    sub["_t"] = pd.to_datetime(sub[time_col], errors="coerce")
    sub = sub[sub[group_col].notna() & sub["_t"].notna()].copy()
    sub[group_col] = sub[group_col].astype(np.int64)
    # pd.Timestamp.value (ns); Series.astype(int64) is wrong for MIMIC shifted years
    sub["_t_ns"] = sub["_t"].map(lambda t: t.value if pd.notna(t) else np.nan)
    sub = sub[sub["_t_ns"].notna()].copy()
    sub["_t_ns"] = sub["_t_ns"].astype(np.int64)
    for gid, grp in sub.groupby(group_col, sort=False):
        grp = grp.sort_values("_t_ns")
        out[int(gid)] = (
            grp["_t_ns"].to_numpy(dtype=np.int64),
            grp.index.to_numpy(dtype=np.int64),
            grp,
        )
    return out


def _window_slice(times_ns: np.ndarray, row_idx: np.ndarray, anchor_ns: int, lb_lo_ns: int, lb_hi_ns: int) -> np.ndarray:
    lo = anchor_ns - lb_lo_ns
    hi = anchor_ns - lb_hi_ns
    l = int(np.searchsorted(times_ns, lo, side="left"))
    r = int(np.searchsorted(times_ns, hi, side="right"))
    return row_idx[l:r]


def _save_partial_counts(
    path: Path,
    cxr_counts: np.ndarray,
    ecg_counts: np.ndarray,
    cxr_signal: np.ndarray,
    ecg_signal: np.ndarray,
    last_i: int,
) -> None:
    np.savez(
        path,
        cxr_counts=cxr_counts[: last_i + 1],
        ecg_counts=ecg_counts[: last_i + 1],
        cxr_signal=cxr_signal[: last_i + 1],
        ecg_signal=ecg_signal[: last_i + 1],
        last_i=last_i,
    )


def _load_partial_counts(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    cxr_counts = np.zeros(n, dtype=np.int32)
    ecg_counts = np.zeros(n, dtype=np.int32)
    cxr_signal = np.zeros(n, dtype=np.int8)
    ecg_signal = np.zeros(n, dtype=np.int8)
    if not path.exists():
        return cxr_counts, ecg_counts, cxr_signal, ecg_signal, -1
    data = np.load(path)
    last_i = int(data["last_i"])
    cxr_counts[: last_i + 1] = data["cxr_counts"]
    ecg_counts[: last_i + 1] = data["ecg_counts"]
    cxr_signal[: last_i + 1] = data["cxr_signal"]
    ecg_signal[: last_i + 1] = data["ecg_signal"]
    return cxr_counts, ecg_counts, cxr_signal, ecg_signal, last_i


def _open_match_writer(
    path: Path | None,
    matches_cols: list[str],
    append: bool,
) -> tuple[csv.DictWriter | None, object | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    mf = open(path, mode, newline="", encoding="utf-8")
    anchor_cols = ["anchor_hadm_id", "anchor_index", "anchor_time"]
    writer = csv.DictWriter(
        mf, fieldnames=anchor_cols + matches_cols, extrasaction="ignore", restval=""
    )
    if mode == "w":
        writer.writeheader()
    return writer, mf


def _write_match_rows(
    writer: csv.DictWriter | None,
    anchor_row: pd.Series,
    frame: pd.DataFrame,
    sel: np.ndarray,
    matches_cols: list[str],
) -> None:
    if writer is None or len(sel) == 0:
        return
    base = {
        "anchor_hadm_id": int(anchor_row["hadm_id"]),
        "anchor_index": anchor_row["index"],
        "anchor_time": anchor_row["index"],
    }
    for ri in sel:
        mrow = frame.loc[ri]
        out = dict(base)
        for c in matches_cols:
            out[c] = mrow.get(c)
        writer.writerow({k: _csv_val(out.get(k)) for k in writer.fieldnames})


def _match_all_anchors(
    anchors: pd.DataFrame,
    cxr_index: dict,
    ecg_index: dict,
    lb_lo_ns: int,
    lb_hi_ns: int,
    write_matches: bool,
    cxr_matches_path: Path | None,
    ecg_matches_path: Path | None,
    partial_path: Path,
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(anchors)
    cxr_counts, ecg_counts, cxr_signal, ecg_signal, loaded_last = _load_partial_counts(partial_path, n)
    start_i = max(int(checkpoint.get("last_anchor_index", -1)) + 1, loaded_last + 1)
    cxr_writer, cxr_f = _open_match_writer(
        cxr_matches_path if write_matches else None, CXR_CATALOG_COLS, append=start_i > 0
    )
    ecg_writer, ecg_f = _open_match_writer(
        ecg_matches_path if write_matches else None, ECG_CATALOG_COLS, append=start_i > 0
    )

    hadm_series = anchors["hadm_id"].astype(np.int64)
    subj_series = anchors["_subject_id"]
    t_series = anchors["_anchor_ns"]

    try:
        for i in tqdm(range(start_i, n), desc="Window match", initial=start_i, total=n):
            anchor_row = anchors.iloc[i]
            t_ns = int(t_series.iloc[i])

            hid = int(hadm_series.iloc[i])
            cxr_pack = cxr_index.get(hid)
            if cxr_pack is not None:
                sel = _window_slice(*cxr_pack[:2], t_ns, lb_lo_ns, lb_hi_ns)
                cxr_counts[i] = len(sel)
                cxr_signal[i] = 1 if len(sel) > 0 else 0
                _write_match_rows(cxr_writer, anchor_row, cxr_pack[2], sel, CXR_CATALOG_COLS)

            sid = subj_series.iloc[i]
            if pd.notna(sid):
                ecg_pack = ecg_index.get(int(sid))
                if ecg_pack is not None:
                    sel = _window_slice(*ecg_pack[:2], t_ns, lb_lo_ns, lb_hi_ns)
                    ecg_counts[i] = len(sel)
                    ecg_signal[i] = 1 if len(sel) > 0 else 0
                    _write_match_rows(ecg_writer, anchor_row, ecg_pack[2], sel, ECG_CATALOG_COLS)

            if i % 5000 == 0:
                checkpoint["last_anchor_index"] = i
                _save_checkpoint(checkpoint)
                _save_partial_counts(partial_path, cxr_counts, ecg_counts, cxr_signal, ecg_signal, i)
    finally:
        if cxr_f is not None:
            cxr_f.close()
        if ecg_f is not None:
            ecg_f.close()

    checkpoint["last_anchor_index"] = n - 1
    _save_checkpoint(checkpoint)
    _save_partial_counts(partial_path, cxr_counts, ecg_counts, cxr_signal, ecg_signal, n - 1)
    return cxr_counts, ecg_counts, cxr_signal, ecg_signal


def main() -> int:
    ap = argparse.ArgumentParser(description="CXR/ECG [t-24h, t-12h] window enrichment for p2f_or_s2f anchors")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT))
    ap.add_argument("--cxr-catalog", default=str(DEFAULT_CXR_CATALOG))
    ap.add_argument("--ecg-catalog", default=str(DEFAULT_ECG_CATALOG))
    ap.add_argument("--output", "-o", default=str(DEFAULT_ANCHOR_OUT), help="Anchor summary CSV")
    ap.add_argument("--cxr-matches", default=str(DEFAULT_CXR_MATCHES))
    ap.add_argument("--ecg-matches", default=str(DEFAULT_ECG_MATCHES))
    ap.add_argument("--lookback-min-hours", type=int, default=LOOKBACK_MIN_H)
    ap.add_argument("--lookback-max-hours", type=int, default=LOOKBACK_MAX_H)
    ap.add_argument("--max-rows", type=int, default=None, help="Process only first N anchors (smoke test)")
    ap.add_argument("--write-matches", action="store_true", help="Write long-format per-anchor match CSVs")
    ap.add_argument("--skip-catalog", action="store_true", help="Reuse existing catalog CSVs on disk")
    ap.add_argument("--fresh", action="store_true", help="Ignore checkpoint and overwrite outputs")
    args = ap.parse_args()

    input_path = Path(args.input)
    cxr_catalog_path = Path(args.cxr_catalog)
    ecg_catalog_path = Path(args.ecg_catalog)
    output_path = Path(args.output)
    cxr_matches_path = Path(args.cxr_matches)
    ecg_matches_path = Path(args.ecg_matches)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    if args.fresh:
        for p in (CHECKPOINT_FILE, output_path, cxr_matches_path, ecg_matches_path, PARTIAL_COUNTS_FILE):
            if p.exists():
                p.unlink()

    checkpoint = {} if args.fresh else _load_checkpoint()

    print(f"Loading anchors from {input_path}...", flush=True)
    read_kw = {"low_memory": False}
    if args.max_rows is not None:
        read_kw["nrows"] = args.max_rows
    df = pd.read_csv(input_path, **read_kw)
    n_in = len(df)
    print(f"  Anchor rows: {n_in:,}", flush=True)

    for c in ("hadm_id", "index"):
        if c not in df.columns:
            print(f"Missing column: {c}")
            return 1

    df = df.copy()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
    anchor_dt = pd.to_datetime(df["index"], errors="coerce")
    df["_anchor_ns"] = anchor_dt.map(lambda t: t.value if pd.notna(t) else np.nan)
    df = df[df["hadm_id"].notna() & df["_anchor_ns"].notna()].copy()
    df["hadm_id"] = df["hadm_id"].astype(np.int64)
    df["_anchor_ns"] = df["_anchor_ns"].astype(np.int64)
    df = df.reset_index(drop=True)
    print(f"  Valid anchors: {len(df):,}", flush=True)

    hadm_ids = set(df["hadm_id"].unique())

    print("Loading admissions for subject_id map...", flush=True)
    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        usecols=["subject_id", "hadm_id"],
    )
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions = admissions.dropna(subset=["hadm_id", "subject_id"])
    hadm_to_subject = (
        admissions.drop_duplicates(subset=["hadm_id"], keep="first")
        .set_index("hadm_id")["subject_id"]
        .astype(np.int64)
        .to_dict()
    )
    df["_subject_id"] = df["hadm_id"].map(hadm_to_subject)
    subject_ids = set(int(s) for s in df["_subject_id"].dropna().unique())

    if args.skip_catalog and cxr_catalog_path.exists() and ecg_catalog_path.exists():
        print(f"Loading existing CXR catalog: {cxr_catalog_path}", flush=True)
        cxr_cat = pd.read_csv(cxr_catalog_path, parse_dates=["supertable_datetime"], low_memory=False)
        print(f"Loading existing ECG catalog: {ecg_catalog_path}", flush=True)
        ecg_cat = pd.read_csv(ecg_catalog_path, parse_dates=["wf_Base_Time"], low_memory=False)
    else:
        cxr_cat = _build_cxr_catalog(hadm_ids)
        ecg_cat = _build_ecg_catalog(subject_ids, hadm_ids)
        cxr_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        cxr_cat.to_csv(cxr_catalog_path, index=False)
        ecg_cat.to_csv(ecg_catalog_path, index=False)
        print(f"Saved CXR catalog → {cxr_catalog_path}", flush=True)
        print(f"Saved ECG catalog → {ecg_catalog_path}", flush=True)

    lb_lo_ns = int(args.lookback_max_hours * 3600 * 1e9)
    lb_hi_ns = int(args.lookback_min_hours * 3600 * 1e9)
    print(
        f"Lookback window: [t-{args.lookback_max_hours}h, t-{args.lookback_min_hours}h]",
        flush=True,
    )

    cxr_index = _index_by_group(cxr_cat, "hadm_id", "supertable_datetime")
    ecg_index = _index_by_group(ecg_cat, "subject_id", "wf_Base_Time")

    cxr_counts, ecg_counts, cxr_signal, ecg_signal = _match_all_anchors(
        df,
        cxr_index,
        ecg_index,
        lb_lo_ns,
        lb_hi_ns,
        args.write_matches,
        cxr_matches_path if args.write_matches else None,
        ecg_matches_path if args.write_matches else None,
        PARTIAL_COUNTS_FILE,
        checkpoint,
    )

    out = df.drop(columns=["_anchor_ns", "_subject_id"], errors="ignore").copy()
    out["CXR_window_count"] = cxr_counts
    out["ECG_window_count"] = ecg_counts
    out["CXR_signal"] = cxr_signal
    out["ECG_signal"] = ecg_signal

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    if PARTIAL_COUNTS_FILE.exists():
        PARTIAL_COUNTS_FILE.unlink()

    print(f"\nSaved anchor summary → {output_path}")
    print(f"  CXR_signal=1: {int(cxr_signal.sum()):,} / {len(out):,} ({100 * cxr_signal.mean():.2f}%)")
    print(f"  ECG_signal=1: {int(ecg_signal.sum()):,} / {len(out):,} ({100 * ecg_signal.mean():.2f}%)")
    if args.write_matches:
        print(f"  CXR matches → {cxr_matches_path}")
        print(f"  ECG matches → {ecg_matches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
