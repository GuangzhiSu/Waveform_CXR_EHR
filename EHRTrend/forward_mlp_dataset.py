"""Single-row EHR (percentile features) + forward [t+12h, t+24h] severity-change labels."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import FeatureSpec, _parse_mapping, _to_bool
from ehr_nextstep_dataset import _as_group_id, _to_has_flag


NS_PER_HOUR = int(3600 * 1e9)


def _trend_from_severities(curr: int, fut: int) -> int:
    """0=increase (worse), 1=remain, 2=decrease (better); severity 0=normal .. 3=severe."""
    if fut > curr:
        return 0
    if fut == curr:
        return 1
    return 2


def _assign_group(df: pd.DataFrame, enriched_csv: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    enr_subj = None
    if enriched_csv is not None:
        enr = pd.read_csv(enriched_csv, low_memory=False, usecols=["hadm_id", "index", "subject_id"])
        enr["hadm_id"] = pd.to_numeric(enr["hadm_id"], errors="coerce")
        enr["_ref_time"] = pd.to_datetime(enr["index"], errors="coerce")
        enr["subject_id"] = pd.to_numeric(enr["subject_id"], errors="coerce")
        enr = enr[enr["hadm_id"].notna() & enr["_ref_time"].notna() & enr["subject_id"].notna()]
        enr = enr.drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")
        enr_subj = enr.rename(columns={"subject_id": "_enr_subject_id"})[["hadm_id", "_ref_time", "_enr_subject_id"]]

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


def _compute_forward_labels_for_group(
    t_ns: np.ndarray,
    sev: np.ndarray,
    has_row: np.ndarray,
    forward_min_hours: int,
    forward_max_hours: int,
) -> np.ndarray:
    """Per-row label in {0,1,2} or -1 if no current modality / no future sample in window."""
    n = len(t_ns)
    out = np.full(n, -1, dtype=np.int64)
    lo_off = int(forward_min_hours * NS_PER_HOUR)
    hi_off = int(forward_max_hours * NS_PER_HOUR)

    sev_clean = np.full(n, -1, dtype=np.int64)
    v = np.asarray(sev, dtype=np.float64)
    finite = np.isfinite(v) & (v >= 0.0) & (v <= 3.0)
    sev_clean[has_row & finite] = np.rint(v[has_row & finite]).astype(np.int64)

    pos = np.nonzero(sev_clean >= 0)[0]
    if pos.size == 0:
        return out
    tv = t_ns[pos]
    sv = sev_clean[pos]

    for i in range(n):
        c = int(sev_clean[i])
        if c < 0:
            continue
        lo = t_ns[i] + lo_off
        hi = t_ns[i] + hi_off
        l = int(np.searchsorted(tv, lo, side="left"))
        r = int(np.searchsorted(tv, hi, side="right"))
        if l >= r:
            continue
        sub = pos[l:r]
        mm = sub > i
        if not np.any(mm):
            continue
        j = int(sub[mm][0])
        fut = int(sev_clean[j])
        out[i] = _trend_from_severities(c, fut)
    return out


class EHRForwardChangeDataset(Dataset):
    """
    One sample = percentile-encoded EHR row at time t (schema inputs only).
    Labels = forward change in [t+12h, t+24h] for s2f / p2f severity (first chronologically
    measurement in that window with valid severity), or -1 if unavailable.

    Rows may only record one modality (has_s2f / has_p2f). Training uses ``s2f_valid`` /
    ``p2f_valid`` so each head is supervised only when that modality is present and a
    forward label exists; the other head is ignored for that row.
    """

    def __init__(
        self,
        source_csv: str,
        schema_csv: str,
        enriched_csv: Optional[str] = None,
        forward_min_hours: int = 12,
        forward_max_hours: int = 24,
    ):
        self.forward_min_hours = forward_min_hours
        self.forward_max_hours = forward_max_hours

        main = pd.read_csv(source_csv, low_memory=False)
        for c in ("hadm_id", "index", "has_s2f_vent_fio2", "has_p2f_vent_fio2", "s2f_vent_fio2_severity", "p2f_vent_fio2_severity"):
            if c not in main.columns:
                raise ValueError(f"Source CSV missing column: {c}")

        main["hadm_id"] = pd.to_numeric(main["hadm_id"], errors="coerce")
        main["_ref_time"] = pd.to_datetime(main["index"], errors="coerce")
        main = main[main["hadm_id"].notna() & main["_ref_time"].notna()].copy()
        main["hadm_id"] = main["hadm_id"].astype(np.int64)
        main = _assign_group(main, enriched_csv)

        schema = pd.read_csv(schema_csv)
        use_schema = schema[schema["use_as_input"].map(_to_bool)].copy()
        specs: List[FeatureSpec] = []
        for _, r in use_schema.iterrows():
            f = str(r["Features"])
            if f not in main.columns:
                continue
            specs.append(
                FeatureSpec(
                    name=f,
                    mapping=_parse_mapping(r.get("onehot_mapping")),
                    default_raw=str(r.get("imputation_params_default_impute", "median")),
                )
            )
        if not specs:
            raise ValueError("No schema input features found in source CSV")

        self.feature_specs = specs
        self.feature_cols = [s.name for s in specs]
        self.input_dim = len(self.feature_cols)

        self.df = main.reset_index(drop=True)
        self.num = self._numeric_frame(self.df)
        self.fill_values = self._build_fill_values()
        self.sorted_values = self._build_sorted_values()
        self.x_pct = self._to_percentiles(self.num)

        t_ns = self.df["_ref_time"].astype("int64").to_numpy()
        grp = self.df["_group_id"].to_numpy(dtype=np.int64)
        s_sev = pd.to_numeric(self.df["s2f_vent_fio2_severity"], errors="coerce").to_numpy()
        p_sev = pd.to_numeric(self.df["p2f_vent_fio2_severity"], errors="coerce").to_numpy()
        has_s = self.df["has_s2f_vent_fio2"].map(_to_has_flag).to_numpy(dtype=bool)
        has_p = self.df["has_p2f_vent_fio2"].map(_to_has_flag).to_numpy(dtype=bool)

        order = np.lexsort((t_ns, grp))
        g_sorted = grp[order]
        t_sorted = t_ns[order]

        s_lab = np.full(len(self.df), -1, dtype=np.int64)
        p_lab = np.full(len(self.df), -1, dtype=np.int64)
        uniq, starts = np.unique(g_sorted, return_index=True)
        for j, _gid in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(g_sorted)
            sl = order[a:b]
            tt = t_sorted[a:b]
            s_sev_g = s_sev[sl]
            p_sev_g = p_sev[sl]
            has_s_g = has_s[sl]
            has_p_g = has_p[sl]
            s_out = _compute_forward_labels_for_group(
                tt, s_sev_g, has_s_g, forward_min_hours, forward_max_hours
            )
            p_out = _compute_forward_labels_for_group(
                tt, p_sev_g, has_p_g, forward_min_hours, forward_max_hours
            )
            s_lab[sl] = s_out
            p_lab[sl] = p_out

        self.s_forward = s_lab
        self.p_forward = p_lab
        self.has_s = has_s
        self.has_p = has_p

        n = len(self.df)
        print(
            f"  EHR forward-change (single-row) dataset: n={n:,}, features={self.input_dim}, "
            f"forward=[t+{forward_min_hours}h, t+{forward_max_hours}h]"
        )
        for name, lab in (("s2f", self.s_forward), ("p2f", self.p_forward)):
            vc = {k: int((lab == k).sum()) for k in (-1, 0, 1, 2)}
            print(f"    {name} label counts (-1=no label): {vc}")
        sup_s = int((self.has_s & (self.s_forward >= 0)).sum())
        sup_p = int((self.has_p & (self.p_forward >= 0)).sum())
        both = int((self.has_s & (self.s_forward >= 0) & self.has_p & (self.p_forward >= 0)).sum())
        print(f"    supervised rows: s2f_head={sup_s:,}, p2f_head={sup_p:,}, both_heads={both:,}")

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
            col = self.num[s.name]
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
            v = self.num[s.name].dropna().to_numpy(dtype=np.float64)
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        sy = int(self.s_forward[idx])
        py = int(self.p_forward[idx])
        return {
            "x": torch.from_numpy(self.x_pct[idx]).float(),
            "s2f_y": sy,
            "p2f_y": py,
            "s2f_valid": bool(self.has_s[idx]) and sy >= 0,
            "p2f_valid": bool(self.has_p[idx]) and py >= 0,
        }
