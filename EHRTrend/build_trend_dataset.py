"""Build EHR trend anchors for classification: decrease/remain/increase."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def classify_state(p2f):
    """0=Severe, 1=Moderate, 2=Mild, 3=Normal (>300 or missing in lookback)."""
    try:
        v = float(p2f)
    except (TypeError, ValueError):
        return 3
    if not np.isfinite(v):
        return 3
    if v < 100:
        return 0
    if v < 200:
        return 1
    if v <= 300:
        return 2
    return 3


def trend_label(prev_state, curr_state):
    if curr_state > prev_state:
        return 2, "increase"
    if curr_state < prev_state:
        return 0, "decrease"
    return 1, "remain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--lookback_min_hours", type=int, default=12)
    ap.add_argument("--lookback_max_hours", type=int, default=24)
    args = ap.parse_args()

    src = Path(args.source_csv)
    out = Path(args.output_csv)
    if not src.exists():
        raise FileNotFoundError(f"source_csv not found: {src}")

    df = pd.read_csv(src, low_memory=False)
    for c in ("index", "subject_id", "p2f_vent_fio2"):
        if c not in df.columns:
            raise ValueError(f"Missing column in source CSV: {c}")

    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df["index"] = pd.to_datetime(df["index"], errors="coerce")
    df["p2f_vent_fio2"] = pd.to_numeric(df["p2f_vent_fio2"], errors="coerce")
    df = df[df["subject_id"].notna() & df["index"].notna() & df["p2f_vent_fio2"].notna()].copy()
    df["subject_id"] = df["subject_id"].astype(np.int64)

    # Anchor points: every row with valid p2f value
    anchors = df[["subject_id", "index", "p2f_vent_fio2"]].copy()
    anchors = anchors.reset_index(drop=True)

    # History pools by subject and time for efficient window scan
    hist = df[["subject_id", "index", "p2f_vent_fio2"]].copy()
    hist_t = hist["index"].astype("int64").to_numpy()
    hist_s = hist["subject_id"].to_numpy(dtype=np.int64)
    order = np.argsort(hist_t)
    hs = hist_s[order]
    ht = hist_t[order]
    hi = order
    by_subject = {}
    uniq, starts = np.unique(hs, return_index=True)
    for j, s in enumerate(uniq):
        a = starts[j]
        b = starts[j + 1] if j + 1 < len(starts) else len(hs)
        by_subject[int(s)] = (ht[a:b], hi[a:b])

    lb_lo = int(args.lookback_max_hours * 3600 * 1e9)
    lb_hi = int(args.lookback_min_hours * 3600 * 1e9)

    a_t = anchors["index"].astype("int64").to_numpy()
    a_s = anchors["subject_id"].to_numpy(dtype=np.int64)
    a_p = anchors["p2f_vent_fio2"].to_numpy(dtype=np.float64)

    prev_states = []
    curr_states = []
    trend_ids = []
    trend_names = []
    n_window = []

    for sid, t, p in zip(a_s, a_t, a_p):
        curr = classify_state(p)
        pack = by_subject.get(int(sid))
        prev = 3  # default normal if no prior points in [t-24h, t-12h]
        n_w = 0
        if pack is not None:
            t_hist, i_hist = pack
            lo = t - lb_lo
            hi = t - lb_hi
            l = np.searchsorted(t_hist, lo, side="left")
            r = np.searchsorted(t_hist, hi, side="right")
            idxs = i_hist[l:r]
            n_w = int(len(idxs))
            if n_w > 0:
                # Use most recent p2f in lookback window as previous state.
                i_prev = idxs[np.argmax(hist_t[idxs])]
                prev = classify_state(hist.iloc[i_prev]["p2f_vent_fio2"])

        tid, tname = trend_label(prev, curr)
        prev_states.append(prev)
        curr_states.append(curr)
        trend_ids.append(tid)
        trend_names.append(tname)
        n_window.append(n_w)

    out_df = anchors.copy()
    out_df["prev_state"] = prev_states
    out_df["curr_state"] = curr_states
    out_df["trend_label"] = trend_ids
    out_df["trend_name"] = trend_names
    out_df["n_window_points"] = n_window

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    print(f"Saved trend anchors: {out}  rows={len(out_df):,}")
    for c, n in [(0, "decrease"), (1, "remain"), (2, "increase")]:
        k = int((out_df["trend_label"] == c).sum())
        print(f"  {n}: {k:,}")


if __name__ == "__main__":
    main()
