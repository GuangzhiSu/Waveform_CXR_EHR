"""Build full ECG/CXR modality catalogs for the contrastive experiments.

The original ECG-CXR temporal experiments inherited the p2f/s2f modality
catalogs, which intentionally cover only that EHR cohort.  This script builds
drop-in replacement catalogs from the raw MIMIC-CXR-JPG and MIMIC-IV-ECG
metadata so pair builders can use all patients with both modalities.

Outputs match the columns consumed by ``build_pairs.py`` / ``build_seq_pairs.py``
and ``build_single_ecg_pairs.py``:

  * CXR: subject_id, dicom_id, hadm_id, supertable_datetime
  * ECG: subject_id, hadm_id, wf_Study_ID, wf_File_Name, wf_Base_Time,
         wf_End_Time, wf_DurationMin, wf_sigLen, wf_ECG_Time, wf_stayHours,
         wf_File_Path
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent

DEFAULT_CXR_ALL_IMAGES = "/hpc/group/kamaleswaranlab/mimic_cxr/all_images.csv"
DEFAULT_CXR_METADATA = (
    "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/"
    "mimic-cxr-2.0.0-metadata.csv.gz"
)
DEFAULT_ECG_ROOT = (
    "/hpc/group/kamaleswaranlab/Waveform/MIMIC_waveform/MIMIC_IV_ECG_Matched"
)
DEFAULT_ECG_RECORD_LIST = f"{DEFAULT_ECG_ROOT}/record_list.csv"
DEFAULT_CXR_OUT = PROJECT_ROOT / "data" / "ecg_cxr_full_cxr_catalog.csv"
DEFAULT_ECG_OUT = PROJECT_ROOT / "data" / "ecg_cxr_full_ecg_catalog.csv"


def _study_datetime(study_date: pd.Series, study_time: pd.Series) -> pd.Series:
    """Parse MIMIC-CXR StudyDate/StudyTime into a pandas datetime series."""
    date = pd.to_numeric(study_date, errors="coerce").astype("Int64").astype(str)
    time = (
        study_time.astype(str)
        .str.split(".", n=1)
        .str[0]
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
        .str[:6]
    )
    return pd.to_datetime(date + time, format="%Y%m%d%H%M%S", errors="coerce")


def build_cxr_catalog(all_images_path: str, metadata_path: str,
                      view_positions: set[str] | None) -> pd.DataFrame:
    cols = ["subject_id", "study_id", "dicom_id", "ViewPosition"]
    cxr = pd.read_csv(all_images_path, usecols=cols, low_memory=False)
    if view_positions:
        cxr = cxr[cxr["ViewPosition"].astype(str).isin(view_positions)].copy()

    meta = pd.read_csv(
        metadata_path,
        usecols=["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime"],
        low_memory=False,
    )
    cxr["dicom_id"] = cxr["dicom_id"].astype(str)
    meta["dicom_id"] = meta["dicom_id"].astype(str)
    merged = cxr.merge(meta, on=["dicom_id", "subject_id", "study_id"], how="left")
    merged["supertable_datetime"] = _study_datetime(
        merged["StudyDate"], merged["StudyTime"]
    )
    merged = merged[
        merged["subject_id"].notna()
        & merged["dicom_id"].notna()
        & merged["supertable_datetime"].notna()
    ].copy()
    merged["subject_id"] = merged["subject_id"].astype(np.int64)
    merged["hadm_id"] = ""
    out = merged[["subject_id", "dicom_id", "hadm_id", "supertable_datetime"]]
    return out.sort_values(["subject_id", "supertable_datetime", "dicom_id"])


def build_ecg_catalog(record_list_path: str, ecg_root: str) -> pd.DataFrame:
    cols = ["subject_id", "study_id", "file_name", "ecg_time", "path"]
    ecg = pd.read_csv(record_list_path, usecols=cols, low_memory=False)
    ecg["wf_Base_Time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    ecg = ecg[
        ecg["subject_id"].notna()
        & ecg["wf_Base_Time"].notna()
        & ecg["path"].notna()
    ].copy()
    ecg["subject_id"] = ecg["subject_id"].astype(np.int64)
    root = Path(ecg_root)
    ecg["wf_File_Path"] = ecg["path"].map(lambda p: str(root / str(p).strip()))
    ecg["wf_File_Name"] = ecg["file_name"].astype(str)
    ecg["wf_Study_ID"] = ecg["study_id"]
    ecg["wf_End_Time"] = ecg["wf_Base_Time"] + pd.to_timedelta(10, unit="s")
    ecg["wf_DurationMin"] = 10.0 / 60.0
    ecg["wf_sigLen"] = 5000
    ecg["wf_ECG_Time"] = ecg["wf_Base_Time"]
    ecg["hadm_id"] = ""
    ecg["wf_stayHours"] = ""
    out_cols = [
        "subject_id", "hadm_id", "wf_Study_ID", "wf_File_Name",
        "wf_Base_Time", "wf_End_Time", "wf_DurationMin", "wf_sigLen",
        "wf_ECG_Time", "wf_stayHours", "wf_File_Path",
    ]
    return ecg[out_cols].sort_values(["subject_id", "wf_Base_Time", "wf_File_Name"])


def _restrict_to_common_subjects(cxr: pd.DataFrame, ecg: pd.DataFrame):
    common = set(cxr["subject_id"].unique()) & set(ecg["subject_id"].unique())
    return (
        cxr[cxr["subject_id"].isin(common)].copy(),
        ecg[ecg["subject_id"].isin(common)].copy(),
        common,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_all_images", default=DEFAULT_CXR_ALL_IMAGES)
    ap.add_argument("--cxr_metadata", default=DEFAULT_CXR_METADATA)
    ap.add_argument("--ecg_record_list", default=DEFAULT_ECG_RECORD_LIST)
    ap.add_argument("--ecg_root", default=DEFAULT_ECG_ROOT)
    ap.add_argument("--cxr_out", default=str(DEFAULT_CXR_OUT))
    ap.add_argument("--ecg_out", default=str(DEFAULT_ECG_OUT))
    ap.add_argument(
        "--view_positions", nargs="*", default=["AP", "PA"],
        help="CXR view positions to keep. Use --view_positions with no values to keep all.",
    )
    ap.add_argument(
        "--keep_all_subjects", action="store_true",
        help="Do not prefilter to subjects that appear in both modalities.",
    )
    args = ap.parse_args()

    view_positions = set(args.view_positions) if args.view_positions else None
    print("=== build_full_catalogs: raw MIMIC ECG/CXR metadata ===", flush=True)
    print(f"  CXR all_images: {args.cxr_all_images}", flush=True)
    print(f"  CXR metadata  : {args.cxr_metadata}", flush=True)
    print(f"  ECG records   : {args.ecg_record_list}", flush=True)

    cxr = build_cxr_catalog(args.cxr_all_images, args.cxr_metadata, view_positions)
    ecg = build_ecg_catalog(args.ecg_record_list, args.ecg_root)

    common = None
    if not args.keep_all_subjects:
        cxr, ecg, common = _restrict_to_common_subjects(cxr, ecg)

    Path(args.cxr_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ecg_out).parent.mkdir(parents=True, exist_ok=True)
    cxr.to_csv(args.cxr_out, index=False)
    ecg.to_csv(args.ecg_out, index=False)

    print(
        f"  Wrote CXR catalog: {args.cxr_out} "
        f"rows={len(cxr):,} subjects={cxr['subject_id'].nunique():,}",
        flush=True,
    )
    print(
        f"  Wrote ECG catalog: {args.ecg_out} "
        f"rows={len(ecg):,} subjects={ecg['subject_id'].nunique():,}",
        flush=True,
    )
    if common is not None:
        print(f"  Common-subject filter: subjects={len(common):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
