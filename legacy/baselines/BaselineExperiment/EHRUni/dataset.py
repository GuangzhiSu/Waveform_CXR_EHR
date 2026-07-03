"""Temporal EHR dataset: lookback rows -> percentile vectors per row -> variable-length sequence."""
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("index", "recorded_time", "supertable_datetime"):
        if c in df.columns:
            ok = pd.to_datetime(df[c], errors="coerce").notna().mean()
            if ok > 0.5:
                return c
    return None


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


def _impute_column_temporal_nearest(v: np.ndarray) -> np.ndarray:
    """
    Per-feature column within a time window (same ordering as time).
    - If exactly one finite value: broadcast it to all timesteps.
    - If multiple finite values: fill each NaN with the value at the nearest finite timestep (index distance).
    - If all NaN: return as-is (caller fills with global default).
    """
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(v)
    n = int(finite.sum())
    t = len(v)
    if n == 0 or n == t:
        return v
    out = v.copy()
    obs_idx = np.where(finite)[0]
    if n == 1:
        out[:] = out[obs_idx[0]]
        return out
    missing = np.where(~finite)[0]
    for k in missing:
        dist = np.abs(obs_idx.astype(np.float64) - float(k))
        nearest = int(obs_idx[int(np.argmin(dist))])
        out[k] = out[nearest]
    return out


class EHRClassificationDataset(Dataset):
    """One sample = all EHR rows in [t-24h, t-12h] for same subject, percentile-encoded."""

    def __init__(
        self,
        anchor_csv: str,
        history_csv: str,
        schema_csv: str,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        window_temporal_impute: bool = True,
    ):
        self.anchor_df = pd.read_csv(anchor_csv, low_memory=False)
        self.history_df = pd.read_csv(history_csv, low_memory=False)
        schema = pd.read_csv(schema_csv)

        if "p2f_class" not in self.anchor_df.columns:
            raise ValueError("Anchor CSV must contain p2f_class")
        self.anchor_df = self.anchor_df[self.anchor_df["p2f_class"].notna()].copy()
        self.anchor_df["p2f_class"] = self.anchor_df["p2f_class"].astype(int)

        a_time_col = _find_time_col(self.anchor_df)
        h_time_col = _find_time_col(self.history_df)
        if not a_time_col or not h_time_col:
            raise ValueError("Could not find parseable time column in anchor/history CSV")

        self.anchor_df["_ref_time"] = pd.to_datetime(self.anchor_df[a_time_col], errors="coerce")
        self.history_df["_ref_time"] = pd.to_datetime(self.history_df[h_time_col], errors="coerce")
        self.anchor_df = self.anchor_df[self.anchor_df["_ref_time"].notna()].copy()
        self.history_df = self.history_df[self.history_df["_ref_time"].notna()].copy()

        self.anchor_df["subject_id"] = pd.to_numeric(self.anchor_df.get("subject_id"), errors="coerce")
        self.history_df["subject_id"] = pd.to_numeric(self.history_df.get("subject_id"), errors="coerce")
        self.anchor_df = self.anchor_df[self.anchor_df["subject_id"].notna()].copy()
        self.history_df = self.history_df[self.history_df["subject_id"].notna()].copy()
        self.anchor_df["subject_id"] = self.anchor_df["subject_id"].astype(np.int64)
        self.history_df["subject_id"] = self.history_df["subject_id"].astype(np.int64)

        # Select input features based on schema file
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
            raise ValueError("No input feature columns found from schema in both anchor/history CSV")
        self.feature_specs = specs
        self.feature_cols = [s.name for s in specs]
        self.input_dim = len(self.feature_cols)
        self.window_temporal_impute = window_temporal_impute

        self.anchor_num = self._numeric_frame(self.anchor_df)
        self.history_num = self._numeric_frame(self.history_df)

        self.fill_values = self._build_fill_values()
        self.sorted_values = self._build_sorted_values()

        self.anchor_pct = self._to_percentiles(self.anchor_num)
        self.history_pct = self._to_percentiles(self.history_num)

        self.anchor_labels = self.anchor_df["p2f_class"].to_numpy(dtype=np.int64)
        self.anchor_subject = self.anchor_df["subject_id"].to_numpy(dtype=np.int64)
        self.anchor_time_ns = self.anchor_df["_ref_time"].astype("int64").to_numpy()

        self.history_subject = self.history_df["subject_id"].to_numpy(dtype=np.int64)
        self.history_time_ns = self.history_df["_ref_time"].astype("int64").to_numpy()

        self.lb_lo_ns = int(lookback_max_hours * 3600 * 1e9)
        self.lb_hi_ns = int(lookback_min_hours * 3600 * 1e9)

        self.by_subject = {}
        order = np.argsort(self.history_time_ns)
        hs = self.history_subject[order]
        ht = self.history_time_ns[order]
        hi = order
        uniq, starts = np.unique(hs, return_index=True)
        for j, s in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(hs)
            self.by_subject[int(s)] = (ht[a:b], hi[a:b])

        self.seq_lens = np.array([self._window_indices(i).size for i in range(len(self.anchor_df))], dtype=np.int32)
        self.seq_lens[self.seq_lens == 0] = 1

        print(
            f"  Temporal EHR dataset: n={len(self.anchor_df):,}, features={self.input_dim}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(f"  Window temporal nearest imputation (raw -> then percentiles): {window_temporal_impute}")
        print(
            f"  Sequence length stats (window rows): min={self.seq_lens.min()}, "
            f"median={int(np.median(self.seq_lens))}, max={self.seq_lens.max()}"
        )

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
            if d == "median" or d == "" or d == "nan":
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

    def _impute_raw_block(self, raw: np.ndarray) -> np.ndarray:
        """raw (T, F) possibly with NaN; per-column temporal nearest then global fill for still-missing columns."""
        t, fv = raw.shape
        if fv != self.input_dim:
            raise ValueError(f"raw feature dim {fv} != input_dim {self.input_dim}")
        imputed = np.empty((t, fv), dtype=np.float64)
        for j in range(fv):
            col = _impute_column_temporal_nearest(raw[:, j])
            if not np.isfinite(col).all():
                fill = self.fill_values[self.feature_specs[j].name]
                col = np.where(np.isfinite(col), col, fill)
            imputed[:, j] = col
        return imputed

    def _row_to_percentiles(self, vals: np.ndarray) -> np.ndarray:
        """vals (F,) finite floats -> percentile row (F,) float32."""
        out = np.zeros(self.input_dim, dtype=np.float32)
        for j, s in enumerate(self.feature_specs):
            x = float(vals[j]) if np.isfinite(vals[j]) else float(self.fill_values[s.name])
            arr = self.sorted_values[s.name]
            idx = np.searchsorted(arr, x, side="right")
            out[j] = idx / max(len(arr), 1)
        return out

    def _percentile_block_from_imputed(self, imputed: np.ndarray) -> np.ndarray:
        t = imputed.shape[0]
        out = np.zeros((t, self.input_dim), dtype=np.float32)
        for i in range(t):
            out[i] = self._row_to_percentiles(imputed[i])
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
        if self.window_temporal_impute:
            if win.size == 0:
                raw = self.anchor_num.iloc[idx : idx + 1].to_numpy(dtype=np.float64)
            else:
                raw = self.history_num.iloc[win].to_numpy(dtype=np.float64)
            imputed = self._impute_raw_block(raw)
            seq = self._percentile_block_from_imputed(imputed)
        else:
            if win.size == 0:
                seq = self.anchor_pct[idx : idx + 1]
            else:
                seq = self.history_pct[win]
        return {
            "ehr_seq": torch.from_numpy(seq).float(),
            "label": int(self.anchor_labels[idx]),
        }
