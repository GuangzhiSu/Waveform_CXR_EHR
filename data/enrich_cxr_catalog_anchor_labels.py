#!/usr/bin/env python
"""
Join each CXR in p2f_or_s2f_cxr_catalog.csv to anchor(s) whose lookback window contains that CXR.

Lookback (same as training): CXR time in [anchor_t - 24h, anchor_t - 12h)
  <=> anchor_t in (CXR_t + 12h, CXR_t + 24h] on the same hadm_id.

For each matching anchor, attach s2f/p2f presence and severity *change* labels (0/1/2)
from the anchor table (default: p2f_or_s2f_anchor_modality_window.csv).

Output: long format — one row per (CXR, anchor) pair; CXRs with no matching anchor
still appear once with empty anchor_* / NaN label fields.

Usage:
  python data/enrich_cxr_catalog_anchor_labels.py
  python data/enrich_cxr_catalog_anchor_labels.py --max-hadm 100   # smoke test
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
DEFAULT_CXR_CATALOG = BASE / "data" / "p2f_or_s2f_cxr_catalog.csv"
DEFAULT_ANCHOR_CSV = BASE / "data" / "p2f_or_s2f_anchor_modality_window.csv"
DEFAULT_OUTPUT = BASE / "data" / "p2f_or_s2f_cxr_catalog_labeled.csv"

LOOKBACK_MIN_H = 12
LOOKBACK_MAX_H = 24

CXR_COLS = ["subject_id", "dicom_id", "hadm_id", "supertable_datetime"]
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
    "hours_cxr_to_anchor",
]


def _anchor_indices_for_cxr(
    anchor_times_ns: np.ndarray,
    cxr_ns: int,
    lb_lo_ns: int,
    lb_hi_ns: int,
) -> slice:
    """Anchor times A where cxr_ns in [A - lb_lo, A - lb_hi)."""
    l = int(np.searchsorted(anchor_times_ns, cxr_ns + lb_hi_ns, side="right"))
    r = int(np.searchsorted(anchor_times_ns, cxr_ns + lb_lo_ns, side="right"))
    return slice(l, r)


def _process_hadm(
    hadm_id: int,
    cxr_sub: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    lb_lo_ns: int,
    lb_hi_ns: int,
) -> list[dict]:
    if anchor_sub.empty:
        rows = []
        for _, cr in cxr_sub.iterrows():
            row = {c: cr[c] for c in CXR_COLS}
            row["anchor_index"] = ""
            row["anchor_hadm_id"] = hadm_id
            row["hours_cxr_to_anchor"] = np.nan
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

    for _, cr in cxr_sub.iterrows():
        cxr_ns = int(pd.Timestamp(cr["supertable_datetime"]).value)
        sl = _anchor_indices_for_cxr(a_times, cxr_ns, lb_lo_ns, lb_hi_ns)
        matched = anchor_sub.iloc[sl]
        base = {c: cr[c] for c in CXR_COLS}

        if matched.empty:
            row = dict(base)
            row["anchor_index"] = ""
            row["anchor_hadm_id"] = hadm_id
            row["hours_cxr_to_anchor"] = np.nan
            for c in ANCHOR_LABEL_COLS:
                row[c] = np.nan
            row["n_matching_anchors"] = 0
            out.append(row)
            continue

        for _, ar in matched.iterrows():
            row = dict(base)
            row["anchor_index"] = ar["index"]
            row["anchor_hadm_id"] = int(ar["hadm_id"])
            delta_h = (int(ar["_anchor_ns"]) - cxr_ns) / 1e9 / 3600.0
            row["hours_cxr_to_anchor"] = round(delta_h, 4)
            for c in ANCHOR_LABEL_COLS:
                row[c] = ar[c]
            row["n_matching_anchors"] = len(matched)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Label CXR catalog rows with matching anchor s2f/p2f outcomes")
    ap.add_argument("--cxr-catalog", default=str(DEFAULT_CXR_CATALOG))
    ap.add_argument("--anchor-csv", default=str(DEFAULT_ANCHOR_CSV))
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--lookback-min-hours", type=int, default=LOOKBACK_MIN_H)
    ap.add_argument("--lookback-max-hours", type=int, default=LOOKBACK_MAX_H)
    ap.add_argument("--max-hadm", type=int, default=None, help="Process only first N distinct hadm_id (smoke test)")
    args = ap.parse_args()

    cxr_path = Path(args.cxr_catalog)
    anchor_path = Path(args.anchor_csv)
    out_path = Path(args.output)

    if not cxr_path.is_file():
        print(f"CXR catalog not found: {cxr_path}")
        return 1
    if not anchor_path.is_file():
        print(f"Anchor CSV not found: {anchor_path}")
        return 1

    print(f"Loading CXR catalog: {cxr_path}", flush=True)
    cxr = pd.read_csv(cxr_path, parse_dates=["supertable_datetime"], low_memory=False)
    for c in CXR_COLS:
        if c not in cxr.columns:
            print(f"CXR catalog missing column: {c}")
            return 1

    cxr["hadm_id"] = pd.to_numeric(cxr["hadm_id"], errors="coerce")
    cxr = cxr[cxr["hadm_id"].notna() & cxr["supertable_datetime"].notna()].copy()
    cxr["hadm_id"] = cxr["hadm_id"].astype(np.int64)
    cxr_hadm = set(cxr["hadm_id"].unique())

    print(f"Loading anchors (hadm_id in CXR catalog only): {anchor_path}", flush=True)
    usecols = ["hadm_id", "index", *ANCHOR_LABEL_COLS]
    anchor_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(anchor_path, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk = chunk[chunk["hadm_id"].isin(cxr_hadm)]
        if not chunk.empty:
            anchor_parts.append(chunk)
    if not anchor_parts:
        print("No anchors overlap CXR catalog hadm_id set.")
        anchors = pd.DataFrame(columns=usecols)
    else:
        anchors = pd.concat(anchor_parts, ignore_index=True)
    del anchor_parts
    # pd.Timestamp.value (ns); DatetimeIndex.astype(int64) is wrong for MIMIC year ~2178
    anchor_dt = pd.to_datetime(anchors["index"], errors="coerce")
    anchors["_anchor_ns"] = anchor_dt.map(lambda t: t.value if pd.notna(t) else np.nan)
    anchors = anchors[anchors["hadm_id"].notna() & anchors["_anchor_ns"].notna()].copy()
    anchors["_anchor_ns"] = anchors["_anchor_ns"].astype(np.int64)
    anchors["hadm_id"] = anchors["hadm_id"].astype(np.int64)

    lb_lo_ns = int(args.lookback_max_hours * 3600 * 1e9)
    lb_hi_ns = int(args.lookback_min_hours * 3600 * 1e9)
    print(
        f"Window: CXR in [anchor-{args.lookback_max_hours}h, anchor-{args.lookback_min_hours}h) "
        f"  (anchor ~12–24h after CXR)",
        flush=True,
    )

    hadm_cxr = cxr_hadm
    hadm_anc = set(anchors["hadm_id"].unique()) if len(anchors) else set()
    hadm_ids = sorted(hadm_cxr & hadm_anc)
    if args.max_hadm is not None:
        hadm_ids = hadm_ids[: args.max_hadm]
    print(f"CXR rows: {len(cxr):,}  anchors: {len(anchors):,}  hadm overlap: {len(hadm_ids):,}", flush=True)

    cxr_by = {int(h): g for h, g in cxr.groupby("hadm_id")}
    anc_by = {int(h): g for h, g in anchors.groupby("hadm_id")}

    all_rows: list[dict] = []
    for hid in tqdm(hadm_ids, desc="Join by hadm_id"):
        all_rows.extend(
            _process_hadm(hid, cxr_by[hid], anc_by.get(hid, pd.DataFrame()), lb_lo_ns, lb_hi_ns)
        )

    # CXRs whose hadm_id has no anchors at all
    orphan_hadm = hadm_cxr - hadm_anc
    if args.max_hadm is None:
        for hid in orphan_hadm:
            all_rows.extend(
                _process_hadm(int(hid), cxr_by[int(hid)], pd.DataFrame(), lb_lo_ns, lb_hi_ns)
            )

    out = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_cxr = len(cxr)
    n_out = len(out)
    n_with_anchor = int((out["anchor_index"].astype(str).str.len() > 0).sum())
    n_multi = int((out["n_matching_anchors"] > 1).sum()) if "n_matching_anchors" in out.columns else 0
    print(f"\nSaved → {out_path}")
    print(f"  Input CXR rows: {n_cxr:,}")
    print(f"  Output rows (long format): {n_out:,}")
    print(f"  Rows with a matching anchor: {n_with_anchor:,} ({100 * n_with_anchor / max(n_out, 1):.1f}%)")
    print(f"  Rows where CXR matched >1 anchor: {n_multi:,}")
    print(
        "  Label columns: has_s2f/has_p2f; severity; "
        "s2f/p2f_vent_fio2_severity_change_12to24h (0/1/2 = change class at anchor)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
