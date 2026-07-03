"""Build patient-temporal (CXR_t1, ECG_interval, CXR_t2) interval samples.

Reuses the same modality catalogs as the CXR/ECG encoder experiments:
  * CXR: data/p2f_or_s2f_cxr_catalog.csv  (subject_id, dicom_id, supertable_datetime)
         + mimic-cxr metadata for dicom_id -> study_id (path building)
  * ECG: data/p2f_or_s2f_ecg_catalog.csv  (subject_id, wf_Base_Time, wf_File_Path, wf_File_Name)

For each patient we sort their CXRs by acquisition time (one node per distinct
timestamp).  A target CXR at t2 is paired with each prior CXR_t1 in
[t2 - MAX_INTERVAL_HOURS, t2].  The ECG context is the adjacent same-patient window
[max(t2 - ECG_LOOKBACK_HOURS, t1), t2].

Outputs (under cache/):
  * patient_temporal_pairs.json  : {"pairs": [...], "cxr_meta": {...}, "ecg_meta": {...}}
  * (paths are resolved here so precompute does not need the metadata again)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from io_utils import get_cxr_path, norm_dicom_id  # noqa: E402

_HOUR_NS = 3600 * 1e9


def _hours(ns: np.ndarray) -> np.ndarray:
    return ns.astype(np.float64) / _HOUR_NS


def load_cxr_nodes(cxr_csv: str, metadata_path: str, cxr_root: str,
                   min_cxrs: int = 2, check_paths: bool = True) -> dict:
    """patient_id -> list of CXR nodes sorted by time: {dicom_id, t_h, path}."""
    df = pd.read_csv(cxr_csv, low_memory=False)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df["supertable_datetime"] = pd.to_datetime(df["supertable_datetime"], errors="coerce")
    df["dicom_id"] = df["dicom_id"].map(norm_dicom_id)
    df = df[df["subject_id"].notna() & df["supertable_datetime"].notna()
            & df["dicom_id"].astype(str).str.len().gt(0)].copy()
    df["subject_id"] = df["subject_id"].astype(np.int64)

    if metadata_path and os.path.isfile(metadata_path):
        meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
        meta["dicom_id"] = meta["dicom_id"].map(norm_dicom_id)
        meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
        df = df.merge(meta, on="dicom_id", how="left")
    else:
        print(f"  WARNING: CXR metadata not found at {metadata_path}; study_id unavailable")
        df["study_id"] = np.nan

    df = df.sort_values(["subject_id", "supertable_datetime"])
    nodes: dict[int, list] = {}
    n_total, n_pathok = 0, 0
    for sid, grp in df.groupby("subject_id", sort=False):
        seen_times = set()
        seq = []
        for _, row in grp.iterrows():
            t = row["supertable_datetime"]
            tkey = int(t.value)
            if tkey in seen_times:
                continue  # one CXR node per distinct timestamp
            seen_times.add(tkey)
            study_id = row.get("study_id", np.nan)
            path = get_cxr_path(row["dicom_id"], int(sid), study_id, cxr_root)
            n_total += 1
            ok = bool(path and os.path.isfile(path)) if check_paths else bool(path)
            if ok:
                n_pathok += 1
            seq.append({
                "dicom_id": str(row["dicom_id"]),
                "t_h": float(tkey) / _HOUR_NS,
                "path": path,
                "path_ok": ok,
            })
        if len(seq) >= min_cxrs:
            nodes[int(sid)] = seq
    label = "path_on_disk" if check_paths else "path_assumed"
    print(f"  CXR nodes: patients(>={min_cxrs} CXR)={len(nodes):,}  "
          f"{label}={n_pathok:,}/{n_total:,}")
    return nodes


def load_ecg_nodes(ecg_csv: str) -> dict:
    """patient_id -> list of ECG nodes sorted by time: {ecg_id, t_h, path}."""
    df = pd.read_csv(ecg_csv, low_memory=False)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df["wf_Base_Time"] = pd.to_datetime(df["wf_Base_Time"], errors="coerce")
    df = df[df["subject_id"].notna() & df["wf_Base_Time"].notna()
            & df["wf_File_Path"].notna()].copy()
    df["subject_id"] = df["subject_id"].astype(np.int64)
    if "wf_File_Name" not in df.columns:
        df["wf_File_Name"] = df["wf_File_Path"].astype(str)
    df = df.sort_values(["subject_id", "wf_Base_Time"])
    nodes: dict[int, list] = {}
    for sid, grp in df.groupby("subject_id", sort=False):
        seq = []
        for _, row in grp.iterrows():
            seq.append({
                "ecg_id": str(row["wf_File_Name"]),
                "t_h": float(row["wf_Base_Time"].value) / _HOUR_NS,
                "path": str(row["wf_File_Path"]).strip(),
            })
        nodes[int(sid)] = seq
    print(f"  ECG nodes: patients={len(nodes):,}  rows={len(df):,}")
    return nodes


def build_pairs(cxr_nodes: dict, ecg_nodes: dict, args) -> tuple[list, dict, dict]:
    pairs = []
    cxr_meta: dict[str, dict] = {}
    ecg_meta: dict[str, dict] = {}
    n_no_ecg = 0
    for sid, cseq in cxr_nodes.items():
        eseq = ecg_nodes.get(sid)
        if not eseq:
            continue
        e_times = np.array([e["t_h"] for e in eseq], dtype=np.float64)
        for i in range(len(cseq)):
            t1 = cseq[i]["t_h"]
            for j in range(i + 1, min(i + 1 + args.max_skip, len(cseq))):
                t2 = cseq[j]["t_h"]
                dt = t2 - t1
                if dt < args.min_interval_hours or dt > args.max_interval_hours:
                    continue
                ecg_start = max(t2 - args.ecg_lookback_hours, t1)
                lo = np.searchsorted(e_times, ecg_start, side="left")
                hi = np.searchsorted(e_times, t2, side="right")
                idxs = list(range(lo, hi))
                if len(idxs) < args.min_ecgs_per_interval:
                    n_no_ecg += 1
                    continue
                if len(idxs) > args.max_ecgs_per_interval:
                    idxs = idxs[-args.max_ecgs_per_interval:]  # keep most recent
                ecg_ids = [eseq[k]["ecg_id"] for k in idxs]
                ecg_times = [eseq[k]["t_h"] for k in idxs]
                for k in idxs:
                    eid = eseq[k]["ecg_id"]
                    ecg_meta.setdefault(eid, {"path": eseq[k]["path"]})
                for node in (cseq[i], cseq[j]):
                    cxr_meta.setdefault(node["dicom_id"], {
                        "path": node["path"], "path_ok": node["path_ok"],
                    })
                pairs.append({
                    "patient_id": int(sid),
                    "t1_h": t1, "t2_h": t2,
                    "cxr_t1": cseq[i]["dicom_id"],
                    "cxr_t2": cseq[j]["dicom_id"],
                    "ecg_ids": ecg_ids,
                    "ecg_times_h": ecg_times,
                    "delta_h": float(dt),
                })
    print(f"  Built pairs: {len(pairs):,}  (skipped no-ecg intervals={n_no_ecg:,})")
    print(f"  Unique CXR={len(cxr_meta):,}  Unique ECG={len(ecg_meta):,}")
    n_pat = len(set(p["patient_id"] for p in pairs))
    per_pat = pd.Series([p["patient_id"] for p in pairs]).value_counts()
    print(f"  Patients with pairs={n_pat:,}  intervals/patient: "
          f"min={per_pat.min()} median={int(per_pat.median())} max={per_pat.max()}")
    return pairs, cxr_meta, ecg_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_csv", default=C.CXR_CATALOG_CSV)
    ap.add_argument("--ecg_csv", default=C.ECG_CATALOG_CSV)
    ap.add_argument("--metadata_path", default=C.CXR_METADATA_PATH)
    ap.add_argument("--cxr_root", default=C.CXR_ROOT)
    ap.add_argument("--out", default=C.PAIRS_JSON)
    ap.add_argument("--min_interval_hours", type=float, default=C.MIN_INTERVAL_HOURS)
    ap.add_argument("--max_interval_hours", type=float, default=C.MAX_INTERVAL_HOURS)
    ap.add_argument("--max_skip", type=int, default=C.MAX_SKIP)
    ap.add_argument("--min_ecgs_per_interval", type=int, default=C.MIN_ECGS_PER_INTERVAL)
    ap.add_argument("--max_ecgs_per_interval", type=int, default=C.MAX_ECGS_PER_INTERVAL)
    ap.add_argument("--ecg_lookback_hours", type=float, default=C.ECG_LOOKBACK_HOURS,
                    help="Use ECGs in [max(t2 - this many hours, t1), t2].")
    ap.add_argument("--require_cxr_on_disk", action="store_true",
                    help="Drop pairs whose CXR jpg is not present on disk.")
    ap.add_argument("--skip_cxr_path_check", action="store_true",
                    help="Do not os.stat every CXR during pair building; assume constructed paths exist.")
    args = ap.parse_args()

    print("=== build_pairs: patient-temporal (CXR_t1, ECG_interval, CXR_t2) ===")
    cxr_nodes = load_cxr_nodes(args.cxr_csv, args.metadata_path, args.cxr_root,
                               check_paths=not args.skip_cxr_path_check)
    ecg_nodes = load_ecg_nodes(args.ecg_csv)
    pairs, cxr_meta, ecg_meta = build_pairs(cxr_nodes, ecg_nodes, args)

    if args.require_cxr_on_disk:
        before = len(pairs)
        pairs = [p for p in pairs
                 if cxr_meta.get(p["cxr_t1"], {}).get("path_ok")
                 and cxr_meta.get(p["cxr_t2"], {}).get("path_ok")]
        print(f"  require_cxr_on_disk: kept {len(pairs):,}/{before:,} pairs")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pairs": pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.out}  ({len(pairs):,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
