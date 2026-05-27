"""EHR sequence dataset for next-step embedding + dual severity-change heads (anchor + per-step)."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import _parse_mapping, _to_bool, FeatureSpec


def _as_group_id(hadm: float, subj: Optional[float]) -> int:
    if subj is not None and not (isinstance(subj, float) and np.isnan(subj)):
        return int(subj)
    return int(hadm)


def _to_has_flag(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "1.0"}:
        return True
    if s in {"0", "false", "no", "0.0", ""}:
        return False
    return bool(pd.notna(v) and v)


def _to_class_label(x, invalid_sentinel: int = -1) -> int:
    if pd.isna(x):
        return invalid_sentinel
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return invalid_sentinel
    if v < 0 or v > 2:
        return invalid_sentinel
    return v


class EHRNextStepDataset(Dataset):
    """
    Each anchor row = supertable row at time t; history = rows with same group_id in
    [t-24h, t-12h] sorted by time. Per-step labels come from each history row's
    s2f/p2f change columns; anchor labels from the anchor row.
    group_id = subject_id (from optional enriched join) if present, else hadm_id.
    """

    def __init__(
        self,
        anchor_source_csv: str,
        history_csv: str,
        schema_csv: str,
        enriched_csv: Optional[str] = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
    ):
        main = pd.read_csv(anchor_source_csv, low_memory=False)
        self.history_df = pd.read_csv(history_csv, low_memory=False)
        schema = pd.read_csv(schema_csv)

        for c in ("hadm_id", "index"):
            if c not in main.columns:
                raise ValueError(f"Anchor/source CSV missing column: {c}")
        for c in ("hadm_id", "index"):
            if c not in self.history_df.columns:
                raise ValueError(f"History CSV missing column: {c}")

        main = main.copy()
        main["hadm_id"] = pd.to_numeric(main["hadm_id"], errors="coerce")
        self.history_df["hadm_id"] = pd.to_numeric(self.history_df["hadm_id"], errors="coerce")
        main["_ref_time"] = pd.to_datetime(main["index"], errors="coerce")
        self.history_df["_ref_time"] = pd.to_datetime(self.history_df["index"], errors="coerce")
        main = main[main["hadm_id"].notna() & main["_ref_time"].notna()].copy()
        self.history_df = self.history_df[
            self.history_df["hadm_id"].notna() & self.history_df["_ref_time"].notna()
        ].copy()
        main["hadm_id"] = main["hadm_id"].astype(np.int64)
        self.history_df["hadm_id"] = self.history_df["hadm_id"].astype(np.int64)

        enr_subj = None
        if enriched_csv is not None:
            enr = pd.read_csv(enriched_csv, low_memory=False, usecols=["hadm_id", "index", "subject_id"])
            enr["hadm_id"] = pd.to_numeric(enr["hadm_id"], errors="coerce")
            enr["_ref_time"] = pd.to_datetime(enr["index"], errors="coerce")
            enr["subject_id"] = pd.to_numeric(enr["subject_id"], errors="coerce")
            enr = enr[enr["hadm_id"].notna() & enr["_ref_time"].notna() & enr["subject_id"].notna()]
            enr = enr.drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")
            enr_subj = enr.rename(columns={"subject_id": "_enr_subject_id"})[
                ["hadm_id", "_ref_time", "_enr_subject_id"]
            ]

        def assign_group(df: pd.DataFrame) -> pd.DataFrame:
            if enr_subj is None:
                df["_group_id"] = df["hadm_id"].astype(np.int64)
                return df
            out = df.merge(enr_subj, on=["hadm_id", "_ref_time"], how="left")

            def row_gid(r):
                s = r["_enr_subject_id"]
                hid = int(r["hadm_id"])
                return _as_group_id(hid, float(s) if pd.notna(s) else None)

            out["_group_id"] = out.apply(row_gid, axis=1)
            del out["_enr_subject_id"]
            return out

        main = assign_group(main)
        self.history_df = assign_group(self.history_df)

        for col in (
            "s2f_vent_fio2_severity_change_12to24h",
            "p2f_vent_fio2_severity_change_12to24h",
            "has_s2f_vent_fio2",
            "has_p2f_vent_fio2",
        ):
            if col not in main.columns:
                raise ValueError(f"Source CSV missing label column: {col}")
        if "has_s2f_vent_fio2" not in self.history_df.columns:
            raise ValueError("History CSV must include has_s2f_vent_fio2 / has_p2f_vent_fio2 and change columns")

        use_schema = schema[schema["use_as_input"].map(_to_bool)].copy()
        common_cols = set(main.columns) & set(self.history_df.columns)
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

        self.anchor_df = main.reset_index(drop=True)
        self.history_df = self.history_df.reset_index(drop=True)

        self.anchor_num = self._numeric_frame(self.anchor_df)
        self.history_num = self._numeric_frame(self.history_df)
        self.fill_values = self._build_fill_values()
        self.sorted_values = self._build_sorted_values()
        self.anchor_pct = self._to_percentiles(self.anchor_num)
        self.history_pct = self._to_percentiles(self.history_num)

        n = len(self.anchor_df)
        self.anchor_time_ns = self.anchor_df["_ref_time"].astype("int64").to_numpy()
        self.anchor_group = self.anchor_df["_group_id"].to_numpy(dtype=np.int64)
        self.anchor_has_s2f = np.array([_to_has_flag(self.anchor_df.iloc[i]["has_s2f_vent_fio2"]) for i in range(n)], dtype=bool)
        self.anchor_has_p2f = np.array([_to_has_flag(self.anchor_df.iloc[i]["has_p2f_vent_fio2"]) for i in range(n)], dtype=bool)
        self.anchor_s2f_cls = np.array(
            [_to_class_label(self.anchor_df.iloc[i]["s2f_vent_fio2_severity_change_12to24h"]) for i in range(n)], dtype=np.int64
        )
        self.anchor_p2f_cls = np.array(
            [_to_class_label(self.anchor_df.iloc[i]["p2f_vent_fio2_severity_change_12to24h"]) for i in range(n)], dtype=np.int64
        )

        self.hist_s2f_cls = np.array(
            [_to_class_label(self.history_df.iloc[i]["s2f_vent_fio2_severity_change_12to24h"]) for i in range(len(self.history_df))],
            dtype=np.int64,
        )
        self.hist_p2f_cls = np.array(
            [_to_class_label(self.history_df.iloc[i]["p2f_vent_fio2_severity_change_12to24h"]) for i in range(len(self.history_df))],
            dtype=np.int64,
        )
        self.hist_has_s2f = np.array([_to_has_flag(self.history_df.iloc[i]["has_s2f_vent_fio2"]) for i in range(len(self.history_df))], dtype=bool)
        self.hist_has_p2f = np.array([_to_has_flag(self.history_df.iloc[i]["has_p2f_vent_fio2"]) for i in range(len(self.history_df))], dtype=bool)

        self.history_group = self.history_df["_group_id"].to_numpy(dtype=np.int64)
        self.history_time_ns = self.history_df["_ref_time"].astype("int64").to_numpy()

        self.lb_lo_ns = int(lookback_max_hours * 3600 * 1e9)
        self.lb_hi_ns = int(lookback_min_hours * 3600 * 1e9)

        order = np.argsort(self.history_time_ns)
        hg = self.history_group[order]
        ht = self.history_time_ns[order]
        hi = order
        self.by_group: Dict[int, tuple] = {}
        uniq, starts = np.unique(hg, return_index=True)
        for j, g in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(hg)
            self.by_group[int(g)] = (ht[a:b], hi[a:b])

        seq_lens = np.array([max(1, int(self._window_indices(i).size)) for i in range(len(self.anchor_df))], dtype=np.int32)
        print(
            f"  EHR next-step dataset: n={len(self.anchor_df):,}, features={self.input_dim}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(
            f"  Sequence length stats: min={seq_lens.min()}, median={int(np.median(seq_lens))}, max={seq_lens.max()}"
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
        return x.replace([np.inf, -np.inf], np.nan)

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
        o = np.zeros((n, self.input_dim), dtype=np.float32)
        for j, s in enumerate(self.feature_specs):
            vals = xdf[s.name].to_numpy(dtype=np.float64)
            fill = self.fill_values[s.name]
            vals = np.where(np.isfinite(vals), vals, fill)
            arr = self.sorted_values[s.name]
            idx = np.searchsorted(arr, vals, side="right")
            o[:, j] = (idx / max(len(arr), 1)).astype(np.float32)
        return o

    def _window_indices(self, idx: int) -> np.ndarray:
        gid = int(self.anchor_group[idx])
        t = int(self.anchor_time_ns[idx])
        pack = self.by_group.get(gid)
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

    def __getitem__(self, idx: int):
        win = self._window_indices(idx)
        if win.size == 0:
            seq = np.zeros((1, self.input_dim), dtype=np.float32)
            s2f_step = np.array([-1], dtype=np.int64)
            p2f_step = np.array([-1], dtype=np.int64)
            s2f_ok = np.array([False], dtype=bool)
            p2f_ok = np.array([False], dtype=bool)
        else:
            seq = self.history_pct[win]
            s2f_step = self.hist_s2f_cls[win]
            p2f_step = self.hist_p2f_cls[win]
            s2f_ok = self.hist_has_s2f[win] & (s2f_step >= 0)
            p2f_ok = self.hist_has_p2f[win] & (p2f_step >= 0)

        return {
            "ehr_seq": torch.from_numpy(seq).float(),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
            "s2f_step": torch.from_numpy(s2f_step).long(),
            "p2f_step": torch.from_numpy(p2f_step).long(),
            "s2f_step_valid": torch.from_numpy(s2f_ok).bool(),
            "p2f_step_valid": torch.from_numpy(p2f_ok).bool(),
        }
