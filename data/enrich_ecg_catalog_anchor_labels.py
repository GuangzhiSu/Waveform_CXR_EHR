#!/usr/bin/env python
"""
Join each ECG in p2f_or_s2f_ecg_catalog.csv to anchor(s) whose lookback window contains that ECG.

Lookback (same as training): ECG time in [anchor_t - 24h, anchor_t - 12h)
  <=> anchor_t in (ECG_t + 12h, ECG_t + 24h] on the same hadm_id.

For each matching anchor, attach s2f/p2f presence and severity *change* labels (0/1/2)
from the anchor table (default: p2f_or_s2f_anchor_modality_window.csv).

Output: long format — one row per (ECG, anchor) pair; ECGs with no matching anchor
still appear once with empty anchor_* / NaN label fields.

Usage:
  python data/enrich_ecg_catalog_anchor_labels.py
  python data/enrich_ecg_catalog_anchor_labels.py --max-hadm 100   # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


BASE = Path(__file__).resolve().parents[1]
DEFAULT_ECG_CATALOG = BASE / "data" / "p2f_or_s2f_ecg_catalog.csv"
DEFAULT_ANCHOR_CSV = BASE / "data" / "p2f_or_s2f_anchor_modality_window.csv"
DEFAULT_OUTPUT = BASE / "data" / "p2f_or_s2f_ecg_catalog_labeled.csv"

LOOKBACK_MIN_H = 12
LOOKBACK_MAX_H = 24

ECG_COLS = [
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
ANCHOR_LABEL_COLS = [
    "has_s2f_vent_fio2",
    "has_p2f_vent_fio2",
    "s2f_vent_fio2_severity",
    "p2f_vent_fio2_severity",
    "s2f_vent_fio2_severity_change_12to24h",
    "p2f_vent_fio2_severity_change_12to24h",
]
OUT_ANCHOR_COLS = [
    "anchor_index",
    "anchor_hadm_id",
    "hours_ecg_to_anchor",
]


def _anchor_indices_for_ecg(
    anchor_times_ns: np.ndarray,
    ecg_ns: int,
    lb_lo_ns: int,
    lb_hi_ns: int,
) -> slice:
    """Anchor times A where ecg_ns in [A - lb_lo, A - lb_hi)."""
    l = int(np.searchsorted(anchor_times_ns, ecg_ns + lb_hi_ns, side="right"))
    r = int(np.searchsorted(anchor_times_ns, ecg_ns + lb_lo_ns, side="right"))
    return slice(l, r)


def _process_hadm(
    hadm_id: int,
    ecg_sub: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    lb_lo_ns: int,
    lb_hi_ns: int,
) -> list[dict]:
    if anchor_sub.empty:
        rows = []
        for _, er in ecg_sub.iterrows():
            row = {c: er[c] for c in ECG_COLS if c in er.index}
            row["anchor_index"] = ""
            row["anchor_hadm_id"] = hadm_id
            row["hours_ecg_to_anchor"] = np.nan
            for c in ANCHOR_LABEL_COLS:
                row[c] = np.nan
            row["n_matching_anchors"] = 0
            rows.append(row)
        return rows

    anchor_sub = anchor_sub.sort_values("_anchor_ns")
    a_times = anchor_sub["_anchor_ns"].to_numpy(dtype=np.int64)
    if a_times.size and a_times.max() < 1e16:
        raise ValueError(
            "anchor timestamps look truncated (not nanoseconds); check index parsing"
        )
    out: list[dict] = []

    for _, er in ecg_sub.iterrows():
        ecg_ns = int(pd.Timestamp(er["wf_Base_Time"]).value)
        sl = _anchor_indices_for_ecg(a_times, ecg_ns, lb_lo_ns, lb_hi_ns)
        matched = anchor_sub.iloc[sl]
        base = {c: er[c] for c in ECG_COLS if c in er.index}

        if matched.empty:
            row = dict(base)
            row["anchor_index"] = ""
            row["anchor_hadm_id"] = hadm_id
            row["hours_ecg_to_anchor"] = np.nan
            for c in ANCHOR_LABEL_COLS:
                row[c] = np.nan
            row["n_matching_anchors"] = 0
            out.append(row)
            continue

        for _, ar in matched.iterrows():
            row = dict(base)
            row["anchor_index"] = ar["index"]
            row["anchor_hadm_id"] = int(ar["hadm_id"])
            delta_h = (int(ar["_anchor_ns"]) - ecg_ns) / 1e9 / 3600.0
            row["hours_ecg_to_anchor"] = round(delta_h, 4)
            for c in ANCHOR_LABEL_COLS:
                row[c] = ar[c]
            row["n_matching_anchors"] = len(matched)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Label ECG catalog rows with matching anchor s2f/p2f outcomes")
    ap.add_argument("--ecg-catalog", default=str(DEFAULT_ECG_CATALOG))
    ap.add_argument("--anchor-csv", default=str(DEFAULT_ANCHOR_CSV))
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--lookback-min-hours", type=int, default=LOOKBACK_MIN_H)
    ap.add_argument("--lookback-max-hours", type=int, default=LOOKBACK_MAX_H)
    ap.add_argument("--max-hadm", type=int, default=None, help="Process only first N distinct hadm_id (smoke test)")
    args = ap.parse_args()

    ecg_path = Path(args.ecg_catalog)
    anchor_path = Path(args.anchor_csv)
    out_path = Path(args.output)

    if not ecg_path.is_file():
        print(f"ECG catalog not found: {ecg_path}")
        return 1
    if not anchor_path.is_file():
        print(f"Anchor CSV not found: {anchor_path}")
        return 1

    print(f"Loading ECG catalog: {ecg_path}", flush=True)
    ecg = pd.read_csv(ecg_path, parse_dates=["wf_Base_Time"], low_memory=False)
    for c in ("subject_id", "hadm_id", "wf_Base_Time"):
        if c not in ecg.columns:
            print(f"ECG catalog missing column: {c}")
            return 1
    keep_ecg = [c for c in ECG_COLS if c in ecg.columns]
    ecg = ecg[keep_ecg].copy()

    ecg["hadm_id"] = pd.to_numeric(ecg["hadm_id"], errors="coerce")
    ecg = ecg[ecg["hadm_id"].notna() & ecg["wf_Base_Time"].notna()].copy()
    ecg["hadm_id"] = ecg["hadm_id"].astype(np.int64)
    ecg_hadm = set(ecg["hadm_id"].unique())

    print(f"Loading anchors (hadm_id in ECG catalog only): {anchor_path}", flush=True)
    usecols = ["hadm_id", "index", *ANCHOR_LABEL_COLS]
    anchor_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(anchor_path, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk = chunk[chunk["hadm_id"].isin(ecg_hadm)]
        if not chunk.empty:
            anchor_parts.append(chunk)
    if not anchor_parts:
        print("No anchors overlap ECG catalog hadm_id set.")
        anchors = pd.DataFrame(columns=usecols)
    else:
        anchors = pd.concat(anchor_parts, ignore_index=True)
    del anchor_parts
    anchor_dt = pd.to_datetime(anchors["index"], errors="coerce")
    anchors["_anchor_ns"] = anchor_dt.map(lambda t: t.value if pd.notna(t) else np.nan)
    anchors = anchors[anchors["hadm_id"].notna() & anchors["_anchor_ns"].notna()].copy()
    anchors["_anchor_ns"] = anchors["_anchor_ns"].astype(np.int64)
    anchors["hadm_id"] = anchors["hadm_id"].astype(np.int64)

    lb_lo_ns = int(args.lookback_max_hours * 3600 * 1e9)
    lb_hi_ns = int(args.lookback_min_hours * 3600 * 1e9)
    print(
        f"Window: ECG in [anchor-{args.lookback_max_hours}h, anchor-{args.lookback_min_hours}h) "
        f"  (anchor ~12–24h after ECG)",
        flush=True,
    )

    hadm_ecg = ecg_hadm
    hadm_anc = set(anchors["hadm_id"].unique()) if len(anchors) else set()
    hadm_ids = sorted(hadm_ecg & hadm_anc)
    if args.max_hadm is not None:
        hadm_ids = hadm_ids[: args.max_hadm]
    print(f"ECG rows: {len(ecg):,}  anchors: {len(anchors):,}  hadm overlap: {len(hadm_ids):,}", flush=True)

    ecg_by = {int(h): g for h, g in ecg.groupby("hadm_id")}
    anc_by = {int(h): g for h, g in anchors.groupby("hadm_id")}

    all_rows: list[dict] = []
    for hid in tqdm(hadm_ids, desc="Join by hadm_id"):
        all_rows.extend(
            _process_hadm(hid, ecg_by[hid], anc_by.get(hid, pd.DataFrame()), lb_lo_ns, lb_hi_ns)
        )

    orphan_hadm = hadm_ecg - hadm_anc
    if args.max_hadm is None:
        for hid in orphan_hadm:
            all_rows.extend(
                _process_hadm(int(hid), ecg_by[int(hid)], pd.DataFrame(), lb_lo_ns, lb_hi_ns)
            )

    out = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_ecg = len(ecg)
    n_out = len(out)
    n_with_anchor = int((out["anchor_index"].astype(str).str.len() > 0).sum())
    n_multi = int((out["n_matching_anchors"] > 1).sum()) if "n_matching_anchors" in out.columns else 0
    print(f"\nSaved → {out_path}")
    print(f"  Input ECG rows: {n_ecg:,}")
    print(f"  Output rows (long format): {n_out:,}")
    print(f"  Rows with a matching anchor: {n_with_anchor:,} ({100 * n_with_anchor / max(n_out, 1):.1f}%)")
    print(f"  Rows where ECG matched >1 anchor: {n_multi:,}")
    print(
        "  Label columns: has_s2f/has_p2f; severity; "
        "s2f/p2f_vent_fio2_severity_change_12to24h (0/1/2 = change class at anchor)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
