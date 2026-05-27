"""ECG-only window dataset: all ECG waveforms in [anchor_t - 24h, anchor_t - 12h]; anchor s2f/p2f change labels."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_BE = Path(__file__).resolve().parents[1] / "BaselineExperiment"
if _BE.is_dir() and str(_BE) not in sys.path:
    sys.path.insert(0, str(_BE))

from ECGUni.dataset import load_ecg, normalize_ecg_per_lead  # noqa: E402
from modality_window_utils import build_modality_index, ecg_path_ok, window_indices_for_anchor  # noqa: E402


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


def _build_hadm_subject_map(df: pd.DataFrame) -> dict[int, int]:
    """Map hadm_id -> MIMIC subject_id from any history row that carries subject_id."""
    if "subject_id" not in df.columns:
        return {}
    sub = df[["hadm_id", "subject_id"]].copy()
    sub["hadm_id"] = pd.to_numeric(sub["hadm_id"], errors="coerce")
    sub["subject_id"] = pd.to_numeric(sub["subject_id"], errors="coerce")
    sub = sub.dropna(subset=["hadm_id", "subject_id"])
    if sub.empty:
        return {}
    sub["hadm_id"] = sub["hadm_id"].astype(np.int64)
    sub["subject_id"] = sub["subject_id"].astype(np.int64)
    return sub.groupby("hadm_id")["subject_id"].first().to_dict()


def _resolve_subject_series(df: pd.DataFrame, hadm_subj: dict[int, int]) -> pd.Series:
    """Prefer row subject_id / enriched merge; else hadm_id -> subject_id map (matches ECGUni grouping)."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if "subject_id" in df.columns:
        out = pd.to_numeric(df["subject_id"], errors="coerce")
    if "_enr_subject_id" in df.columns:
        out = out.fillna(pd.to_numeric(df["_enr_subject_id"], errors="coerce"))
    missing = out.isna()
    if missing.any() and hadm_subj:
        mapped = df.loc[missing, "hadm_id"].map(hadm_subj)
        out.loc[missing] = mapped
    return out

class ECGWindowDataset(Dataset):
    """
    History from enriched supertable (wf_File_Path).
    Anchors from p2f_or_s2f; per-row severity labels merged onto history.
    Each sample = ECG waveforms for all rows in [t-24h, t-12h].
    """

    def __init__(
        self,
        anchor_source_csv: str,
        history_csv: str,
        label_lookup_csv: str,
        enriched_csv: Optional[str] = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        ecg_target_len: int = 5000,
        normalize_ecg_per_lead: bool = True,
    ):
        self.ecg_target_len = int(ecg_target_len)
        self.normalize_ecg_per_lead = normalize_ecg_per_lead

        main = pd.read_csv(anchor_source_csv, low_memory=False)
        self.history_df = pd.read_csv(history_csv, low_memory=False)

        for c in ("hadm_id", "index"):
            if c not in main.columns:
                raise ValueError(f"Anchor CSV missing: {c}")
        for c in ("hadm_id", "index", "wf_File_Path", "wf_Base_Time", "subject_id"):
            if c not in self.history_df.columns:
                raise ValueError(f"History CSV must include {c} (use p2f_vent_fio2_enriched).")

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
            return out

        main = assign_group(main)
        self.history_df = assign_group(self.history_df)

        hadm_subj = _build_hadm_subject_map(self.history_df)

        lb = pd.read_csv(label_lookup_csv, low_memory=False)
        need = (
            "hadm_id",
            "index",
            "s2f_vent_fio2_severity_change_12to24h",
            "p2f_vent_fio2_severity_change_12to24h",
            "has_s2f_vent_fio2",
            "has_p2f_vent_fio2",
        )
        for c in need:
            if c not in lb.columns:
                raise ValueError(f"label_lookup_csv missing {c}")
        lb = lb[list(need)].copy()
        lb["hadm_id"] = pd.to_numeric(lb["hadm_id"], errors="coerce")
        lb["_ref_time"] = pd.to_datetime(lb["index"], errors="coerce")
        lb = lb[lb["hadm_id"].notna() & lb["_ref_time"].notna()]
        lb["hadm_id"] = lb["hadm_id"].astype(np.int64)
        lb = lb.drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")
        keep_cols = list(need)
        keep_cols[1] = "_ref_time"
        self.history_df = self.history_df.merge(
            lb[keep_cols],
            on=["hadm_id", "_ref_time"],
            how="left",
        )

        self.anchor_df = main.reset_index(drop=True)
        self.history_df = self.history_df.reset_index(drop=True)

        anchor_subject = _resolve_subject_series(self.anchor_df, hadm_subj)
        n_resolved = int(anchor_subject.notna().sum())
        print(
            f"  Anchor subject_id resolved: {n_resolved}/{len(self.anchor_df):,} "
            f"({100.0 * n_resolved / max(len(self.anchor_df), 1):.1f}%)"
        )

        n = len(self.anchor_df)
        self.anchor_time_ns = self.anchor_df["_ref_time"].astype("int64").to_numpy()
        self.anchor_group = self.anchor_df["_group_id"].to_numpy(dtype=np.int64)
        self.anchor_subject = np.full(n, -1, dtype=np.int64)
        ok = anchor_subject.notna().to_numpy()
        self.anchor_subject[ok] = anchor_subject[ok].astype(np.int64).to_numpy()
        self.anchor_has_s2f = np.array(
            [_to_has_flag(self.anchor_df.iloc[i]["has_s2f_vent_fio2"]) for i in range(n)], dtype=bool
        )
        self.anchor_has_p2f = np.array(
            [_to_has_flag(self.anchor_df.iloc[i]["has_p2f_vent_fio2"]) for i in range(n)], dtype=bool
        )
        self.anchor_s2f_cls = np.array(
            [_to_class_label(self.anchor_df.iloc[i]["s2f_vent_fio2_severity_change_12to24h"]) for i in range(n)],
            dtype=np.int64,
        )
        self.anchor_p2f_cls = np.array(
            [_to_class_label(self.anchor_df.iloc[i]["p2f_vent_fio2_severity_change_12to24h"]) for i in range(n)],
            dtype=np.int64,
        )

        self.lb_lo_ns = int(lookback_max_hours * 3600 * 1e9)
        self.lb_hi_ns = int(lookback_min_hours * 3600 * 1e9)

        ecg_rows = self.history_df[
            self.history_df["wf_Base_Time"].notna() & self.history_df["wf_File_Path"].notna()
        ].copy()
        ecg_rows = ecg_rows[ecg_rows["wf_File_Path"].astype(str).str.strip().astype(bool)]
        ecg_rows["subject_id"] = _resolve_subject_series(ecg_rows, hadm_subj)
        ecg_rows = ecg_rows[ecg_rows["subject_id"].notna()].copy()
        self.ecg_hist_df, self.by_group_ecg = build_modality_index(
            ecg_rows, group_col="subject_id", time_col="wf_Base_Time"
        )

        seq_lens = np.array(
            [max(1, int(self._window_indices(i).size)) for i in range(len(self.anchor_df))], dtype=np.int32
        )
        print(
            f"  ECGWindowDataset: n={len(self.anchor_df):,}, ecg_len={self.ecg_target_len}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(
            f"  ECG history rows (wf_Base_Time): n={len(self.ecg_hist_df):,}, "
            f"subjects={len(self.by_group_ecg):,}"
        )
        print(
            f"  Sequence length stats: min={seq_lens.min()}, median={int(np.median(seq_lens))}, max={seq_lens.max()}"
        )
        self._print_modality_coverage()

    def _row_has_ecg(self, row: pd.Series) -> bool:
        return ecg_path_ok(row.get("wf_File_Path"))

    def _print_modality_coverage(self, sample_size: int = 500) -> None:
        n = len(self.anchor_df)
        if n == 0:
            return
        rng = np.random.RandomState(0)
        idxs = rng.choice(n, size=min(sample_size, n), replace=False)
        any_valid = 0
        for idx in idxs:
            win = self._window_indices(int(idx))
            if win.size == 0:
                continue
            if any(self._row_has_ecg(self.ecg_hist_df.iloc[int(wi)]) for wi in win):
                any_valid += 1
        print(
            f"  ECG coverage (sample n={len(idxs)}): "
            f"{any_valid}/{len(idxs)} anchors have >=1 valid ECG in window "
            f"({100.0 * any_valid / len(idxs):.1f}%)"
        )

    def _window_indices(self, idx: int) -> np.ndarray:
        sid = int(self.anchor_subject[idx])
        if sid < 0:
            return np.empty((0,), dtype=np.int64)
        return window_indices_for_anchor(
            self.by_group_ecg,
            sid,
            int(self.anchor_time_ns[idx]),
            self.lb_lo_ns,
            self.lb_hi_ns,
        )

    def _resize_ecg(self, sig: torch.Tensor) -> torch.Tensor:
        c, L = sig.shape
        T = self.ecg_target_len
        if L == T:
            return sig
        if L > T:
            start = max(0, (L - T) // 2)
            return sig[:, start : start + T]
        out = torch.zeros(c, T, dtype=sig.dtype)
        out[:, :L] = sig
        return out

    def _load_ecg(self, row: pd.Series) -> Tuple[torch.Tensor, bool]:
        wf = row.get("wf_File_Path")
        has_ecg = ecg_path_ok(wf)
        if has_ecg:
            sig = load_ecg(str(wf).strip())
            if self.normalize_ecg_per_lead:
                sig = normalize_ecg_per_lead(sig)
            return self._resize_ecg(sig.float()), True
        return torch.zeros(12, self.ecg_target_len), False

    def __len__(self):
        return len(self.anchor_df)

    def __getitem__(self, idx: int):
        win = self._window_indices(idx)
        if win.size == 0:
            ecgs = [torch.zeros(12, self.ecg_target_len)]
            m_ecg = [False]
        else:
            ecgs, m_ecg = [], []
            for wi in win:
                row = self.ecg_hist_df.iloc[int(wi)]
                ecg, he = self._load_ecg(row)
                ecgs.append(ecg)
                m_ecg.append(he)

        return {
            "ecg_seq": torch.stack(ecgs, dim=0),
            "ecg_mask": torch.tensor(m_ecg, dtype=torch.bool),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
        }
