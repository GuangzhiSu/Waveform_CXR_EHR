"""Helpers for CXR/ECG window datasets — align with CXRUni / ECGUni temporal indexing."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def ecg_path_ok(wf) -> bool:
    """WFDB record paths are stems; ``os.path.isfile(path)`` is False even when ``path.hea`` exists."""
    if pd.isna(wf):
        return False
    p = str(wf).strip()
    if not p:
        return False
    if os.path.isfile(p):
        return True
    return os.path.isfile(f"{p}.hea")


def build_modality_index(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, Dict[int, tuple]]:
    """Sort *df* by (group, modality time) and build per-group (times_ns, row_indices) lookup."""
    hist = df.copy()
    hist["_mod_time"] = pd.to_datetime(hist[time_col], errors="coerce")
    hist = hist[hist[group_col].notna() & hist["_mod_time"].notna()].copy()
    hist[group_col] = pd.to_numeric(hist[group_col], errors="coerce")
    hist = hist[hist[group_col].notna()].copy()
    hist[group_col] = hist[group_col].astype(np.int64)
    hist = hist.reset_index(drop=True)

    times_ns = hist["_mod_time"].astype("int64").to_numpy()
    groups = hist[group_col].to_numpy(dtype=np.int64)
    order = np.argsort(times_ns)
    groups = groups[order]
    times_ns = times_ns[order]
    row_idx = order

    by_group: Dict[int, tuple] = {}
    uniq, starts = np.unique(groups, return_index=True)
    for j, g in enumerate(uniq):
        a = starts[j]
        b = starts[j + 1] if j + 1 < len(starts) else len(groups)
        by_group[int(g)] = (times_ns[a:b], row_idx[a:b])
    return hist, by_group


def window_indices_for_anchor(
    by_group: Dict[int, tuple],
    group_id: int,
    anchor_time_ns: int,
    lb_lo_ns: int,
    lb_hi_ns: int,
) -> np.ndarray:
    """Return row indices whose modality time falls in [anchor - lb_lo, anchor - lb_hi)."""
    pack = by_group.get(int(group_id))
    if pack is None:
        return np.empty((0,), dtype=np.int64)
    t_hist, i_hist = pack
    lo = anchor_time_ns - lb_lo_ns
    hi = anchor_time_ns - lb_hi_ns
    l = np.searchsorted(t_hist, lo, side="left")
    r = np.searchsorted(t_hist, hi, side="right")
    return i_hist[l:r]
