"""Anchor-centric ECG dataset from p2f_or_s2f_ecg_catalog_labeled.csv (pre-joined window + labels)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_BE = Path(__file__).resolve().parents[1] / "BaselineExperiment"
if _BE.is_dir() and str(_BE) not in sys.path:
    sys.path.insert(0, str(_BE))

from ECGUni.dataset import load_ecg, normalize_ecg_per_lead  # noqa: E402

_ECG_LOAD_EPS = 1e-8


def _wfdb_record_on_disk(path: str) -> bool:
    """WFDB records use ``record.hea`` + ``record.dat``; bare path is often not a regular file."""
    p = path.strip()
    if not p:
        return False
    return os.path.exists(p) or os.path.exists(p + ".hea")


def _ecg_loaded_ok(ecg: torch.Tensor) -> bool:
    return bool(ecg.numel() > 0 and float(ecg.abs().mean()) > _ECG_LOAD_EPS)


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


class ECGLabeledCatalogDataset(Dataset):
    """
    One sample = one anchor with all ECGs listed in the labeled catalog for that anchor.

    Source: ``p2f_or_s2f_ecg_catalog_labeled.csv`` (long table: one row per ECG–anchor pair).
    """

    def __init__(
        self,
        labeled_csv: str,
        ecg_target_len: int = 1000,
        normalize_ecg_per_lead_flag: bool = True,
        lookback_min_hours: float = 12.0,
        lookback_max_hours: float = 24.0,
        require_hours_in_window: bool = True,
        drop_invalid_anchors: bool = True,
    ):
        self.ecg_target_len = int(ecg_target_len)
        self.normalize_ecg_per_lead_flag = normalize_ecg_per_lead_flag
        self.lookback_min_hours = lookback_min_hours
        self.lookback_max_hours = lookback_max_hours
        self.drop_invalid_anchors = drop_invalid_anchors
        self._ecg_path_ok_cache: dict[str, bool] = {}

        raw = pd.read_csv(labeled_csv, low_memory=False)
        for c in (
            "subject_id",
            "hadm_id",
            "wf_Base_Time",
            "wf_File_Path",
            "anchor_index",
            "anchor_hadm_id",
            "has_s2f_vent_fio2",
            "has_p2f_vent_fio2",
            "s2f_vent_fio2_severity_change_12to24h",
            "p2f_vent_fio2_severity_change_12to24h",
        ):
            if c not in raw.columns:
                raise ValueError(f"Labeled ECG CSV missing column: {c}")

        raw = raw.copy()
        raw["anchor_index"] = raw["anchor_index"].astype(str).str.strip()
        matched = raw[raw["anchor_index"] != ""].copy()
        n_raw = len(raw)
        n_drop = n_raw - len(matched)
        print(f"  ECGLabeledCatalogDataset: labeled rows={n_raw:,}  with_anchor={len(matched):,}  dropped={n_drop:,}")

        if "hours_ecg_to_anchor" in matched.columns:
            hrs = pd.to_numeric(matched["hours_ecg_to_anchor"], errors="coerce")
            in_win = (hrs > lookback_min_hours) & (hrs <= lookback_max_hours)
            if require_hours_in_window:
                before = len(matched)
                matched = matched[in_win].copy()
                print(
                    f"  Filter hours_ecg_to_anchor in ({lookback_min_hours}, {lookback_max_hours}]: "
                    f"{len(matched):,} / {before:,} rows"
                )
            else:
                print(
                    f"  hours_ecg_to_anchor in ({lookback_min_hours}, {lookback_max_hours}]: "
                    f"{int(in_win.sum()):,}/{len(matched):,} ({100*in_win.mean():.1f}%)"
                )

        matched["wf_Base_Time"] = pd.to_datetime(matched["wf_Base_Time"], errors="coerce")
        matched = matched[matched["wf_Base_Time"].notna()].copy()

        matched["_anchor_key"] = (
            matched["anchor_hadm_id"].astype(str) + "|" + matched["anchor_index"].astype(str)
        )

        groups: List[pd.DataFrame] = []
        for _, grp in matched.groupby("_anchor_key", sort=False):
            grp = grp.sort_values("wf_Base_Time").drop_duplicates(subset=["wf_File_Name"], keep="first")
            groups.append(grp.reset_index(drop=True))

        self.groups = groups
        n = len(groups)
        self.anchor_has_s2f = np.zeros(n, dtype=bool)
        self.anchor_has_p2f = np.zeros(n, dtype=bool)
        self.anchor_s2f_cls = np.full(n, -1, dtype=np.int64)
        self.anchor_p2f_cls = np.full(n, -1, dtype=np.int64)
        self.n_ecg_rows = []
        self.n_ecg_rows_dropped_at_load = 0

        for i, grp in enumerate(groups):
            row0 = grp.iloc[0]
            self.anchor_has_s2f[i] = _to_has_flag(row0["has_s2f_vent_fio2"])
            self.anchor_has_p2f[i] = _to_has_flag(row0["has_p2f_vent_fio2"])
            self.anchor_s2f_cls[i] = _to_class_label(row0["s2f_vent_fio2_severity_change_12to24h"])
            self.anchor_p2f_cls[i] = _to_class_label(row0["p2f_vent_fio2_severity_change_12to24h"])
            self.n_ecg_rows.append(len(grp))

        if drop_invalid_anchors:
            self._drop_anchors_without_valid_ecg()

        seq_lens = np.array([max(1, x) for x in self.n_ecg_rows], dtype=np.int32)
        print(f"  Unique anchors (samples): n={n:,}")
        print(
            f"  ECGs per anchor: min={seq_lens.min()}, median={int(np.median(seq_lens))}, "
            f"max={seq_lens.max()}, mean={seq_lens.mean():.2f}"
        )
        self._print_label_stats()
        self._print_loadable_ecg_sample()

    def _probe_ecg_path_ok(self, path: str) -> bool:
        p = path.strip()
        if not p:
            return False
        cached = self._ecg_path_ok_cache.get(p)
        if cached is not None:
            return cached
        if not _wfdb_record_on_disk(p):
            self._ecg_path_ok_cache[p] = False
            return False
        ecg = load_ecg(p, target_len=self.ecg_target_len)
        if self.normalize_ecg_per_lead_flag:
            ecg = normalize_ecg_per_lead(ecg)
        ok = _ecg_loaded_ok(ecg)
        self._ecg_path_ok_cache[p] = ok
        return ok

    def _anchor_has_valid_ecg(self, grp: pd.DataFrame) -> bool:
        for _, row in grp.iterrows():
            path = row.get("wf_File_Path")
            if isinstance(path, str) and path.strip() and self._probe_ecg_path_ok(path):
                return True
        return False

    def _drop_anchors_without_valid_ecg(self) -> None:
        n_before = len(self.groups)
        keep: List[int] = []
        for i, grp in enumerate(self.groups):
            if self._anchor_has_valid_ecg(grp):
                keep.append(i)
            elif (i + 1) % 5000 == 0:
                print(f"  Scanning ECG load validity: {i + 1:,}/{n_before:,} anchors...")
        n_drop = n_before - len(keep)
        if n_drop == 0:
            print(f"  Drop invalid ECG anchors: 0 / {n_before:,} (all anchors have >=1 valid waveform)")
            return
        print(f"  Drop invalid ECG anchors: {n_drop:,} / {n_before:,} (no loadable non-zero waveform)")
        self.groups = [self.groups[i] for i in keep]
        self.anchor_has_s2f = self.anchor_has_s2f[keep]
        self.anchor_has_p2f = self.anchor_has_p2f[keep]
        self.anchor_s2f_cls = self.anchor_s2f_cls[keep]
        self.anchor_p2f_cls = self.anchor_p2f_cls[keep]
        self.n_ecg_rows = [self.n_ecg_rows[i] for i in keep]

    def _print_label_stats(self) -> None:
        n = len(self)
        s_cnt = np.bincount(
            self.anchor_s2f_cls[(self.anchor_has_s2f) & (self.anchor_s2f_cls >= 0)],
            minlength=3,
        )
        p_cnt = np.bincount(
            self.anchor_p2f_cls[(self.anchor_has_p2f) & (self.anchor_p2f_cls >= 0)],
            minlength=3,
        )
        n_s = int(((self.anchor_has_s2f) & (self.anchor_s2f_cls >= 0)).sum())
        n_p = int(((self.anchor_has_p2f) & (self.anchor_p2f_cls >= 0)).sum())
        print(
            f"  Labels: s2f n={n_s:,} counts={s_cnt.tolist()}  "
            f"majority_acc={float(s_cnt.max())/max(n_s,1):.4f}"
        )
        print(
            f"  Labels: p2f n={n_p:,} counts={p_cnt.tolist()}  "
            f"majority_acc={float(p_cnt.max())/max(n_p,1):.4f}"
        )

    def _print_loadable_ecg_sample(self, sample_size: int = 300) -> None:
        n = len(self)
        if n == 0:
            return
        rng = np.random.RandomState(0)
        idxs = rng.choice(n, size=min(sample_size, n), replace=False)
        ok_anchors = sum(1 for idx in idxs if self._count_loadable(int(idx)) > 0)
        print(
            f"  Loadable ECG (sample {len(idxs)} anchors): {ok_anchors}/{len(idxs)} "
            f"({100.0 * ok_anchors / len(idxs):.1f}%) have >=1 WFDB record (.hea) on disk"
        )

    def _count_loadable(self, idx: int) -> int:
        grp = self.groups[idx]
        return sum(1 for _, row in grp.iterrows() if self._row_has_ecg(row))

    def _row_has_ecg(self, row: pd.Series) -> bool:
        """Fast check for init diagnostics (WFDB ``.hea`` present; same rule as ECGUni/verify_inputs)."""
        path = row.get("wf_File_Path")
        return bool(isinstance(path, str) and path.strip() and _wfdb_record_on_disk(path.strip()))

    def _load_ecg(self, row: pd.Series) -> Tuple[torch.Tensor, bool]:
        """Match ECGUni: always wfdb.rdsamp when path is set; do not gate on os.path.isfile(path)."""
        path = row.get("wf_File_Path")
        if not isinstance(path, str) or not path.strip():
            return torch.zeros(12, self.ecg_target_len), False
        p = path.strip()
        if not _wfdb_record_on_disk(p):
            return torch.zeros(12, self.ecg_target_len), False
        ecg = load_ecg(p, target_len=self.ecg_target_len)
        if self.normalize_ecg_per_lead_flag:
            ecg = normalize_ecg_per_lead(ecg)
        return ecg, _ecg_loaded_ok(ecg)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        grp = self.groups[idx]
        ecgs, m_ecg = [], []
        for _, row in grp.iterrows():
            wav, ok = self._load_ecg(row)
            if ok:
                ecgs.append(wav)
                m_ecg.append(True)
            else:
                self.n_ecg_rows_dropped_at_load += 1
        if not ecgs:
            ecgs = [torch.zeros(12, self.ecg_target_len)]
            m_ecg = [False]
        return {
            "ecg_seq": torch.stack(ecgs, dim=0),
            "ecg_mask": torch.tensor(m_ecg, dtype=torch.bool),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
        }
