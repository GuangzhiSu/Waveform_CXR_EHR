"""Build single-ECG -> future-CXR pairs (Experiments 1 & 2).

Each sample is a single ECG paired with a single future CXR of the *same
patient* whose acquisition time falls a chosen horizon ahead of the ECG:

    9h <= cxr_time - ecg_time <= 15h          (configurable)

i.e. ``ECG at time t  ->  CXR at time t + [MIN, MAX] hours``.

This is the simplest test of the central hypothesis: can a single ECG embedding
be aligned with a future CXR embedding, *without* ever seeing a prior CXR?

Reuses the same catalogs as ``build_pairs.py``:
  * CXR: data/p2f_or_s2f_cxr_catalog.csv + mimic-cxr metadata (dicom_id->study_id)
  * ECG: data/p2f_or_s2f_ecg_catalog.csv

Output (cache/single_ecg_pairs.json):
  {"pairs": [{patient_id, ecg_id, ecg_time_h, cxr_id, cxr_time_h, delta_h}, ...],
   "cxr_meta": {dicom_id: {path, path_ok}}, "ecg_meta": {ecg_id: {path}}}

``--restrict_to_cache`` keeps only ECG/CXR already embedded in the existing
caches so Experiments 1 & 2 can run with **no extra GPU precompute**.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from build_pairs import load_cxr_nodes, load_ecg_nodes  # noqa: E402


def build_single_pairs(cxr_nodes: dict, ecg_nodes: dict, args) -> tuple[list, dict, dict]:
    pairs = []
    cxr_meta: dict[str, dict] = {}
    ecg_meta: dict[str, dict] = {}
    n_no_future = 0
    for sid, eseq in ecg_nodes.items():
        cseq = cxr_nodes.get(sid)
        if not cseq:
            continue
        c_times = np.array([c["t_h"] for c in cseq], dtype=np.float64)
        per_patient = 0
        for e in eseq:
            if per_patient >= args.max_pairs_per_patient:
                break
            t = e["t_h"]
            lo = np.searchsorted(c_times, t + args.min_horizon_hours, side="left")
            hi = np.searchsorted(c_times, t + args.max_horizon_hours, side="right")
            idxs = list(range(lo, hi))
            if not idxs:
                n_no_future += 1
                continue
            # Keep the nearest-in-time future CXRs (smallest horizon first).
            idxs = idxs[: args.max_cxr_per_ecg]
            for k in idxs:
                if per_patient >= args.max_pairs_per_patient:
                    break
                cnode = cseq[k]
                ecg_meta.setdefault(e["ecg_id"], {"path": e["path"]})
                cxr_meta.setdefault(cnode["dicom_id"], {
                    "path": cnode["path"], "path_ok": cnode["path_ok"],
                })
                pairs.append({
                    "patient_id": int(sid),
                    "ecg_id": e["ecg_id"],
                    "ecg_time_h": float(t),
                    "cxr_id": cnode["dicom_id"],
                    "cxr_time_h": float(cnode["t_h"]),
                    "delta_h": float(cnode["t_h"] - t),
                })
                per_patient += 1
    print(f"  Built single-ECG pairs: {len(pairs):,}  (ECGs with no future CXR={n_no_future:,})")
    print(f"  Unique CXR={len(cxr_meta):,}  Unique ECG={len(ecg_meta):,}")
    if pairs:
        import pandas as pd
        per_pat = pd.Series([p["patient_id"] for p in pairs]).value_counts()
        deltas = np.array([p["delta_h"] for p in pairs])
        print(f"  Patients={per_pat.size:,}  pairs/patient: "
              f"min={per_pat.min()} median={int(per_pat.median())} max={per_pat.max()}")
        print(f"  horizon hours: min={deltas.min():.1f} median={np.median(deltas):.1f} "
              f"max={deltas.max():.1f}")
    return pairs, cxr_meta, ecg_meta


def _restrict_to_cache(pairs, cxr_meta, ecg_meta):
    cxr_ids = set(json.load(open(C.CXR_IDS_JSON))) if Path(C.CXR_IDS_JSON).is_file() else set()
    ecg_ids = set(json.load(open(C.ECG_IDS_JSON))) if Path(C.ECG_IDS_JSON).is_file() else set()
    before = len(pairs)
    pairs = [p for p in pairs if p["cxr_id"] in cxr_ids and p["ecg_id"] in ecg_ids]
    cxr_meta = {k: v for k, v in cxr_meta.items() if k in {p["cxr_id"] for p in pairs}}
    ecg_meta = {k: v for k, v in ecg_meta.items() if k in {p["ecg_id"] for p in pairs}}
    print(f"  restrict_to_cache: kept {len(pairs):,}/{before:,} pairs already embedded "
          f"(cxr_cache={len(cxr_ids):,}, ecg_cache={len(ecg_ids):,})")
    return pairs, cxr_meta, ecg_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_csv", default=C.CXR_CATALOG_CSV)
    ap.add_argument("--ecg_csv", default=C.ECG_CATALOG_CSV)
    ap.add_argument("--metadata_path", default=C.CXR_METADATA_PATH)
    ap.add_argument("--cxr_root", default=C.CXR_ROOT)
    ap.add_argument("--out", default=C.SINGLE_ECG_PAIRS_JSON)
    ap.add_argument("--min_horizon_hours", type=float, default=C.SINGLE_MIN_HORIZON_HOURS)
    ap.add_argument("--max_horizon_hours", type=float, default=C.SINGLE_MAX_HORIZON_HOURS)
    ap.add_argument("--max_cxr_per_ecg", type=int, default=C.SINGLE_MAX_CXR_PER_ECG)
    ap.add_argument("--max_pairs_per_patient", type=int, default=C.SINGLE_MAX_PAIRS_PER_PATIENT)
    ap.add_argument("--restrict_to_cache", action="store_true",
                    help="Keep only ECG/CXR already in the existing embedding cache (no GPU recompute).")
    ap.add_argument("--require_cxr_on_disk", action="store_true")
    ap.add_argument("--skip_cxr_path_check", action="store_true",
                    help="Do not os.stat every CXR during pair building; assume constructed paths exist.")
    args = ap.parse_args()

    print("=== build_single_ecg_pairs: single ECG -> future CXR "
          f"({args.min_horizon_hours:.0f}-{args.max_horizon_hours:.0f}h) ===")
    cxr_nodes = load_cxr_nodes(args.cxr_csv, args.metadata_path, args.cxr_root,
                               min_cxrs=1,
                               check_paths=not args.skip_cxr_path_check)
    ecg_nodes = load_ecg_nodes(args.ecg_csv)
    pairs, cxr_meta, ecg_meta = build_single_pairs(cxr_nodes, ecg_nodes, args)

    if args.require_cxr_on_disk:
        before = len(pairs)
        pairs = [p for p in pairs if cxr_meta.get(p["cxr_id"], {}).get("path_ok")]
        print(f"  require_cxr_on_disk: kept {len(pairs):,}/{before:,} pairs")
    if args.restrict_to_cache:
        pairs, cxr_meta, ecg_meta = _restrict_to_cache(pairs, cxr_meta, ecg_meta)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pairs": pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.out}  ({len(pairs):,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
