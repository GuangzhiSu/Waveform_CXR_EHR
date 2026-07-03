"""Temporal EHR dataset for trend classification (decrease/remain/increase)."""
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _parse_mapping(raw) -> Optional[Dict[str, float]]:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return None
    out = {}
    if isinstance(obj, list):
        for kv in obj:
            if isinstance(kv, (list, tuple)) and len(kv) == 2:
                out[str(kv[0])] = float(kv[1])
    return out or None


@dataclass
class FeatureSpec:
    name: str
    mapping: Optional[Dict[str, float]]
    default_raw: str


def drop_anchors_without_window(
    n_anchors: int,
    window_sizes: np.ndarray,
    *,
    label: str = "anchors",
) -> np.ndarray:
    """Return row indices whose lookback window has at least one history row."""
    keep = np.flatnonzero(window_sizes > 0)
    n_skip = n_anchors - len(keep)
    if n_skip:
        print(
            f"  Skipped {n_skip:,}/{n_anchors:,} {label} with empty lookback window "
            f"({100.0 * n_skip / max(n_anchors, 1):.1f}%)"
        )
    if len(keep) == 0:
        raise ValueError(f"No {label} with non-empty lookback window")
    return keep.astype(np.int64, copy=False)


class EHRTrendDataset(Dataset):
    """One sample = all EHR rows in [t-24h, t-12h] for same subject; label is trend_label."""

    def __init__(
        self,
        anchor_csv: str,
        history_csv: str,
        schema_csv: str,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
    ):
        self.anchor_df = pd.read_csv(anchor_csv, low_memory=False)
        self.history_df = pd.read_csv(history_csv, low_memory=False)
        schema = pd.read_csv(schema_csv)

        for c in ("subject_id", "index", "trend_label"):
            if c not in self.anchor_df.columns:
                raise ValueError(f"Anchor CSV missing column: {c}")
        for c in ("subject_id", "index"):
            if c not in self.history_df.columns:
                raise ValueError(f"History CSV missing column: {c}")

        self.anchor_df["subject_id"] = pd.to_numeric(self.anchor_df["subject_id"], errors="coerce")
        self.history_df["subject_id"] = pd.to_numeric(self.history_df["subject_id"], errors="coerce")
        self.anchor_df["_ref_time"] = pd.to_datetime(self.anchor_df["index"], errors="coerce")
        self.history_df["_ref_time"] = pd.to_datetime(self.history_df["index"], errors="coerce")
        self.anchor_df = self.anchor_df[
            self.anchor_df["subject_id"].notna()
            & self.anchor_df["_ref_time"].notna()
            & self.anchor_df["trend_label"].notna()
        ].copy()
        self.history_df = self.history_df[
            self.history_df["subject_id"].notna() & self.history_df["_ref_time"].notna()
        ].copy()
        self.anchor_df["subject_id"] = self.anchor_df["subject_id"].astype(np.int64)
        self.history_df["subject_id"] = self.history_df["subject_id"].astype(np.int64)
        self.anchor_df["trend_label"] = self.anchor_df["trend_label"].astype(np.int64)

        use_schema = schema[schema["use_as_input"].map(_to_bool)].copy()
        common_cols = set(self.anchor_df.columns) & set(self.history_df.columns)
        specs: List[FeatureSpec] = []
        for _, r in use_schema.iterrows():
            f = str(r["Features"])
            if f not in common_cols:
                continue
            specs.append(
                FeatureSpec(
                    name=f,
                    mapping=_parse_mapping(r.get("onehot_mapping")),
                    default_raw=str(r.get("imputation_params_default_impute", "median")),
                )
            )
        if not specs:
            raise ValueError("No schema input features found in history/anchor CSV")

        self.feature_specs = specs
        self.feature_cols = [s.name for s in specs]
        self.input_dim = len(self.feature_cols)

        self.anchor_num = self._numeric_frame(self.anchor_df)
        self.history_num = self._numeric_frame(self.history_df)

        self.fill_values = self._build_fill_values()
        self.sorted_values = self._build_sorted_values()

        self.anchor_pct = self._to_percentiles(self.anchor_num)
        self.history_pct = self._to_percentiles(self.history_num)

        self.anchor_labels = self.anchor_df["trend_label"].to_numpy(dtype=np.int64)
        self.anchor_subject = self.anchor_df["subject_id"].to_numpy(dtype=np.int64)
        self.anchor_time_ns = self.anchor_df["_ref_time"].astype("int64").to_numpy()

        self.history_subject = self.history_df["subject_id"].to_numpy(dtype=np.int64)
        self.history_time_ns = self.history_df["_ref_time"].astype("int64").to_numpy()

        self.lb_lo_ns = int(lookback_max_hours * 3600 * 1e9)
        self.lb_hi_ns = int(lookback_min_hours * 3600 * 1e9)

        order = np.argsort(self.history_time_ns)
        hs = self.history_subject[order]
        ht = self.history_time_ns[order]
        hi = order
        self.by_subject = {}
        uniq, starts = np.unique(hs, return_index=True)
        for j, s in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(hs)
            self.by_subject[int(s)] = (ht[a:b], hi[a:b])

        n_anchors = len(self.anchor_df)
        window_sizes = np.array(
            [self._window_indices(i).size for i in range(n_anchors)], dtype=np.int32
        )
        keep = drop_anchors_without_window(n_anchors, window_sizes)
        self._subset_anchors(keep)
        seq_lens = np.array(
            [self._window_indices(i).size for i in range(len(self.anchor_df))], dtype=np.int32
        )

        print(
            f"  EHR trend dataset: n={len(self.anchor_df):,}, features={self.input_dim}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(
            f"  Sequence length stats: min={seq_lens.min()}, median={int(np.median(seq_lens))}, "
            f"max={seq_lens.max()}"
        )

    def _subset_anchors(self, keep: np.ndarray) -> None:
        self.anchor_df = self.anchor_df.iloc[keep].reset_index(drop=True)
        self.anchor_num = self.anchor_num.iloc[keep].reset_index(drop=True)
        self.anchor_pct = self.anchor_pct[keep]
        self.anchor_labels = self.anchor_labels[keep]
        self.anchor_subject = self.anchor_subject[keep]
        self.anchor_time_ns = self.anchor_time_ns[keep]

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for s in self.feature_specs:
            col = df[s.name]
            if s.mapping is not None:
                out[s.name] = col.astype(str).map(s.mapping)
            else:
                out[s.name] = pd.to_numeric(col, errors="coerce")
        x = pd.DataFrame(out)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x

    def _build_fill_values(self) -> Dict[str, float]:
        fills = {}
        for s in self.feature_specs:
            col = self.history_num[s.name]
            med = float(col.median()) if col.notna().any() else 0.0
            d = str(s.default_raw).strip().lower()
            if d in {"", "nan", "median"}:
                fills[s.name] = med
            else:
                try:
                    fills[s.name] = float(d)
                except ValueError:
                    fills[s.name] = med
        return fills

    def _build_sorted_values(self) -> Dict[str, np.ndarray]:
        out = {}
        for s in self.feature_specs:
            v = self.history_num[s.name].dropna().to_numpy(dtype=np.float64)
            if v.size == 0:
                out[s.name] = np.array([self.fill_values[s.name]], dtype=np.float64)
            else:
                out[s.name] = np.sort(v)
        return out

    def _to_percentiles(self, xdf: pd.DataFrame) -> np.ndarray:
        n = len(xdf)
        out = np.zeros((n, self.input_dim), dtype=np.float32)
        for j, s in enumerate(self.feature_specs):
            vals = xdf[s.name].to_numpy(dtype=np.float64)
            fill = self.fill_values[s.name]
            vals = np.where(np.isfinite(vals), vals, fill)
            arr = self.sorted_values[s.name]
            idx = np.searchsorted(arr, vals, side="right")
            out[:, j] = (idx / max(len(arr), 1)).astype(np.float32)
        return out

    def _window_indices(self, idx: int) -> np.ndarray:
        sid = int(self.anchor_subject[idx])
        t = int(self.anchor_time_ns[idx])
        pack = self.by_subject.get(sid)
        if pack is None:
            return np.empty((0,), dtype=np.int64)
        t_hist, i_hist = pack
        lo = t - self.lb_lo_ns
        hi = t - self.lb_hi_ns
        l = np.searchsorted(t_hist, lo, side="left")
        r = np.searchsorted(t_hist, hi, side="right")
        return i_hist[l:r]

    def __len__(self):
        return len(self.anchor_df)

    def __getitem__(self, idx):
        win = self._window_indices(idx)
        if win.size == 0:
            raise IndexError(f"anchor {idx} has empty lookback window (should have been filtered)")
        seq = self.history_pct[win]
        return {
            "ehr_seq": torch.from_numpy(seq).float(),
            "label": int(self.anchor_labels[idx]),
        }
