"""CXR-only window dataset: all CXRs in [anchor_t - 24h, anchor_t - 12h]; anchor s2f/p2f change labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_BE = Path(__file__).resolve().parents[1] / "BaselineExperiment"
for _p in (_BE, _BE / "CXRUni"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cxr_classification.dataset import (  # noqa: E402
    _first_non_empty_study_id,
    _norm_dicom_id,
    get_cxr_path,
    load_cxr,
)
from modality_window_utils import build_modality_index, window_indices_for_anchor  # noqa: E402


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


class CXRWindowDataset(Dataset):
    """
    History from enriched supertable (dicom_id, subject_id, study_id).
    Anchors from p2f_or_s2f; per-row severity labels merged onto history.
    Each sample = CXR images for all rows in [t-24h, t-12h].
    """

    def __init__(
        self,
        anchor_source_csv: str,
        history_csv: str,
        label_lookup_csv: str,
        enriched_csv: Optional[str] = None,
        cxr_root: str = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path: Optional[str] = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        cxr_split: str = "train",
        imagenet_normalize: bool = True,
    ):
        self.cxr_root = cxr_root
        self.cxr_split = cxr_split
        self.imagenet_normalize = imagenet_normalize

        main = pd.read_csv(anchor_source_csv, low_memory=False)
        self.history_df = pd.read_csv(history_csv, low_memory=False)

        for c in ("hadm_id", "index"):
            if c not in main.columns:
                raise ValueError(f"Anchor CSV missing: {c}")
        for c in ("hadm_id", "index", "dicom_id", "subject_id", "supertable_datetime"):
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
            del out["_enr_subject_id"]
            return out

        main = assign_group(main)
        self.history_df = assign_group(self.history_df)

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

        if metadata_path and os.path.isfile(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
            if "dicom_id" in self.history_df.columns:
                self.history_df["dicom_id"] = self.history_df["dicom_id"].map(_norm_dicom_id)
            if "study_id" in self.history_df.columns:
                self.history_df = self.history_df.drop(columns=["study_id"], errors="ignore")
            self.history_df = self.history_df.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        self.anchor_df = main.reset_index(drop=True)
        self.history_df = self.history_df.reset_index(drop=True)

        n = len(self.anchor_df)
        self.anchor_time_ns = self.anchor_df["_ref_time"].astype("int64").to_numpy()
        self.anchor_group = self.anchor_df["_group_id"].to_numpy(dtype=np.int64)
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

        cxr_rows = self.history_df[
            self.history_df["supertable_datetime"].notna()
            & self.history_df["dicom_id"].notna()
            & self.history_df["subject_id"].notna()
        ].copy()
        self.cxr_hist_df, self.by_group_cxr = build_modality_index(
            cxr_rows, group_col="subject_id", time_col="supertable_datetime"
        )

        seq_lens = np.array(
            [max(1, int(self._window_indices(i).size)) for i in range(len(self.anchor_df))], dtype=np.int32
        )
        print(
            f"  CXRWindowDataset: n={len(self.anchor_df):,}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(
            f"  CXR history rows (supertable_datetime): n={len(self.cxr_hist_df):,}, "
            f"subjects={len(self.by_group_cxr):,}"
        )
        print(
            f"  Sequence length stats: min={seq_lens.min()}, median={int(np.median(seq_lens))}, max={seq_lens.max()}"
        )
        self._print_modality_coverage()

    def _row_has_cxr(self, row: pd.Series) -> bool:
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        if not (pd.notna(dicom_id) and pd.notna(subject_id)):
            return False
        study_id = _first_non_empty_study_id(row)
        path = get_cxr_path(_norm_dicom_id(dicom_id), int(subject_id), study_id, self.cxr_root)
        return bool(path and os.path.isfile(path))

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
            if any(self._row_has_cxr(self.cxr_hist_df.iloc[int(wi)]) for wi in win):
                any_valid += 1
        print(
            f"  CXR coverage (sample n={len(idxs)}): "
            f"{any_valid}/{len(idxs)} anchors have >=1 valid CXR in window "
            f"({100.0 * any_valid / len(idxs):.1f}%)"
        )

    def _window_indices(self, idx: int) -> np.ndarray:
        return window_indices_for_anchor(
            self.by_group_cxr,
            int(self.anchor_group[idx]),
            int(self.anchor_time_ns[idx]),
            self.lb_lo_ns,
            self.lb_hi_ns,
        )

    def _load_cxr(self, row: pd.Series) -> Tuple[torch.Tensor, bool]:
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        study_id = _first_non_empty_study_id(row)
        has_cxr = pd.notna(dicom_id) and pd.notna(subject_id)
        if has_cxr:
            dicom_id = _norm_dicom_id(dicom_id)
            path = get_cxr_path(dicom_id, int(subject_id), study_id, self.cxr_root)
            has_cxr = bool(path and os.path.isfile(path))
            if has_cxr:
                return load_cxr(path, self.cxr_split, imagenet_normalize=self.imagenet_normalize), True
        return torch.zeros(3, 224, 224), False

    def __len__(self):
        return len(self.anchor_df)

    def __getitem__(self, idx: int):
        win = self._window_indices(idx)
        if win.size == 0:
            cxrs = [torch.zeros(3, 224, 224)]
            m_cxr = [False]
        else:
            cxrs, m_cxr = [], []
            for wi in win:
                row = self.cxr_hist_df.iloc[int(wi)]
                cxr, hc = self._load_cxr(row)
                cxrs.append(cxr)
                m_cxr.append(hc)

        return {
            "cxr_seq": torch.stack(cxrs, dim=0),
            "cxr_mask": torch.tensor(m_cxr, dtype=torch.bool),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
        }
