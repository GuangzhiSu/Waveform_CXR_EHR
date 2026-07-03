"""Build the two sequence pair sets (Experiments 3 & 4) in a single pass.

Reuses the slow catalog loaders from ``build_pairs.py`` once, then emits:

  * Exp 3  (seq_target_pairs.json) -- maximal, NO CXR_t1:
      every CXR with >=1 same-patient ECG in [t2 - lookback, t2 - min_horizon]
      becomes a target. Sample:
      {patient_id, t2_h, cxr_t2, ecg_ids, ecg_times_h, delta_h}
      where delta_h = t2 - (most recent ECG time).

  * Exp 4  (patient_temporal_pairs.json) -- with CXR_t1:
      every target t2 paired with each earlier CXR_t1 in [t2 - 24h, t2]
      by default (up to MAX_SKIP earlier CXRs), keeping same-patient ECGs in
      [max(t2 - 12h, t1), t2]. Sample:
      {patient_id, t1_h, t2_h, cxr_t1, cxr_t2, ecg_ids, ecg_times_h, delta_h}
      where delta_h = t2 - t1.

Exp 4 is a strict subset of "targets that also have a prior CXR", so its sample
count is <= Exp 3's, as intended (fewer constraints -> more samples upstream).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from build_pairs import load_cxr_nodes, load_ecg_nodes  # noqa: E402


def _cap_recent(idxs: list, max_ecgs: int) -> list:
    return idxs[-max_ecgs:] if len(idxs) > max_ecgs else idxs


def build(cxr_nodes: dict, ecg_nodes: dict, args):
    target_pairs, t1_pairs = [], []
    cxr_meta: dict[str, dict] = {}
    ecg_meta: dict[str, dict] = {}

    def add_cxr(node):
        cxr_meta.setdefault(node["dicom_id"], {"path": node["path"], "path_ok": node["path_ok"]})

    def add_ecgs(eseq, idxs):
        for k in idxs:
            ecg_meta.setdefault(eseq[k]["ecg_id"], {"path": eseq[k]["path"]})

    for sid, cseq in cxr_nodes.items():
        eseq = ecg_nodes.get(sid)
        if not eseq:
            continue
        e_times = np.array([e["t_h"] for e in eseq], dtype=np.float64)

        # ---- Exp 3: each CXR as a target, ECGs 12-24h before t2 by default ----
        for j in range(len(cseq)):
            t2 = cseq[j]["t_h"]
            lo = int(np.searchsorted(e_times, t2 - args.lookback_hours, side="left"))
            hi = int(np.searchsorted(e_times, t2 - args.min_horizon_hours, side="right"))
            idxs = _cap_recent(list(range(lo, hi)), args.max_ecgs)
            if len(idxs) < args.min_ecgs:
                continue
            add_cxr(cseq[j])
            add_ecgs(eseq, idxs)
            ecg_times = [eseq[k]["t_h"] for k in idxs]
            target_pairs.append({
                "patient_id": int(sid), "t2_h": t2,
                "cxr_t2": cseq[j]["dicom_id"],
                "ecg_ids": [eseq[k]["ecg_id"] for k in idxs],
                "ecg_times_h": ecg_times,
                "delta_h": float(t2 - max(ecg_times)),
            })

        # ---- Exp 4: t1 -> t2 with ECGs in [max(t2 - lookback, t1), t2] ----
        for i in range(len(cseq)):
            t1 = cseq[i]["t_h"]
            for j in range(i + 1, min(i + 1 + args.max_skip, len(cseq))):
                t2 = cseq[j]["t_h"]
                dt = t2 - t1
                if dt < args.min_interval_hours or dt > args.max_interval_hours:
                    continue
                ecg_start = max(t2 - args.ecg_lookback_hours, t1)
                lo = int(np.searchsorted(e_times, ecg_start, side="left"))
                hi = int(np.searchsorted(e_times, t2, side="right"))
                idxs = _cap_recent(list(range(lo, hi)), args.max_ecgs)
                if len(idxs) < args.min_ecgs:
                    continue
                add_cxr(cseq[i])
                add_cxr(cseq[j])
                add_ecgs(eseq, idxs)
                t1_pairs.append({
                    "patient_id": int(sid), "t1_h": t1, "t2_h": t2,
                    "cxr_t1": cseq[i]["dicom_id"], "cxr_t2": cseq[j]["dicom_id"],
                    "ecg_ids": [eseq[k]["ecg_id"] for k in idxs],
                    "ecg_times_h": [eseq[k]["t_h"] for k in idxs],
                    "delta_h": float(dt),
                })

    return target_pairs, t1_pairs, cxr_meta, ecg_meta


def _report(name, pairs):
    if not pairs:
        print(f"  [{name}] 0 pairs"); return
    pat = pd.Series([p["patient_id"] for p in pairs]).value_counts()
    necg = np.array([len(p["ecg_ids"]) for p in pairs])
    print(f"  [{name}] pairs={len(pairs):,}  patients={pat.size:,}  "
          f"pairs/pt med={int(pat.median())} max={pat.max()}  "
          f"ecgs/sample med={int(np.median(necg))} max={necg.max()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_csv", default=C.CXR_CATALOG_CSV)
    ap.add_argument("--ecg_csv", default=C.ECG_CATALOG_CSV)
    ap.add_argument("--metadata_path", default=C.CXR_METADATA_PATH)
    ap.add_argument("--cxr_root", default=C.CXR_ROOT)
    ap.add_argument("--target_out", default=C.SEQ_TARGET_PAIRS_JSON)
    ap.add_argument("--t1_out", default=C.PAIRS_JSON)
    ap.add_argument("--min_horizon_hours", type=float, default=C.SEQ_MIN_HORIZON_HOURS)
    ap.add_argument("--lookback_hours", type=float, default=C.SEQ_LOOKBACK_HOURS)
    ap.add_argument("--min_interval_hours", type=float, default=C.MIN_INTERVAL_HOURS)
    ap.add_argument("--max_interval_hours", type=float, default=C.MAX_INTERVAL_HOURS)
    ap.add_argument("--max_skip", type=int, default=C.MAX_SKIP)
    ap.add_argument("--min_ecgs", type=int, default=C.MIN_ECGS_PER_INTERVAL)
    ap.add_argument("--max_ecgs", type=int, default=C.MAX_ECGS_PER_INTERVAL)
    ap.add_argument("--ecg_lookback_hours", type=float, default=C.ECG_LOOKBACK_HOURS,
                    help="Exp4 ECGs are in [max(t2 - this many hours, t1), t2].")
    ap.add_argument("--require_cxr_on_disk", action="store_true")
    ap.add_argument("--skip_cxr_path_check", action="store_true",
                    help="Do not os.stat every CXR during pair building; assume constructed paths exist.")
    args = ap.parse_args()

    print("=== build_seq_pairs: Exp3 (no t1) + Exp4 (with t1) ===")
    cxr_nodes = load_cxr_nodes(args.cxr_csv, args.metadata_path, args.cxr_root,
                               min_cxrs=1,
                               check_paths=not args.skip_cxr_path_check)
    ecg_nodes = load_ecg_nodes(args.ecg_csv)
    target_pairs, t1_pairs, cxr_meta, ecg_meta = build(cxr_nodes, ecg_nodes, args)

    if args.require_cxr_on_disk:
        target_pairs = [p for p in target_pairs if cxr_meta.get(p["cxr_t2"], {}).get("path_ok")]
        t1_pairs = [p for p in t1_pairs
                    if cxr_meta.get(p["cxr_t1"], {}).get("path_ok")
                    and cxr_meta.get(p["cxr_t2"], {}).get("path_ok")]

    _report("Exp3 seq_target", target_pairs)
    _report("Exp4 seq_t1", t1_pairs)
    print(f"  Unique CXR={len(cxr_meta):,}  Unique ECG={len(ecg_meta):,}")

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.target_out, "w") as f:
        json.dump({"pairs": target_pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.target_out}  ({len(target_pairs):,} pairs)")
    with open(args.t1_out, "w") as f:
        json.dump({"pairs": t1_pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.t1_out}  ({len(t1_pairs):,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
