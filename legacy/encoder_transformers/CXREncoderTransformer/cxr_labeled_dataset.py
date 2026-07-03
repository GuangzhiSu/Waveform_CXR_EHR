"""Anchor-centric CXR dataset from p2f_or_s2f_cxr_catalog_labeled.csv (pre-joined window + labels)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_EWT = Path(__file__).resolve().parents[1] / "EHRWindowTransformer"
_BE = Path(__file__).resolve().parents[1] / "BaselineExperiment"
for _p in (_EWT, _BE, _BE / "CXRUni"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cxr_classification.dataset import (  # noqa: E402
    _first_non_empty_study_id,
    _norm_dicom_id,
    get_cxr_path,
    load_cxr,
)


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


class CXRLabeledCatalogDataset(Dataset):
    """
    One sample = one anchor with all CXRs listed in the labeled catalog for that anchor.

    Source: ``p2f_or_s2f_cxr_catalog_labeled.csv`` (long table: one row per CXR–anchor pair).
    Rows with empty ``anchor_index`` are dropped (no supervision window).
    """

    def __init__(
        self,
        labeled_csv: str,
        cxr_root: str = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path: Optional[str] = None,
        cxr_split: str = "train",
        imagenet_normalize: bool = True,
        lookback_min_hours: float = 12.0,
        lookback_max_hours: float = 24.0,
        require_hours_in_window: bool = True,
    ):
        self.cxr_root = cxr_root
        self.cxr_split = cxr_split
        self.imagenet_normalize = imagenet_normalize
        self.lookback_min_hours = lookback_min_hours
        self.lookback_max_hours = lookback_max_hours

        raw = pd.read_csv(labeled_csv, low_memory=False)
        for c in (
            "subject_id",
            "dicom_id",
            "hadm_id",
            "supertable_datetime",
            "anchor_index",
            "anchor_hadm_id",
            "has_s2f_vent_fio2",
            "has_p2f_vent_fio2",
            "s2f_vent_fio2_severity_change_12to24h",
            "p2f_vent_fio2_severity_change_12to24h",
        ):
            if c not in raw.columns:
                raise ValueError(f"Labeled CXR CSV missing column: {c}")

        raw = raw.copy()
        raw["anchor_index"] = raw["anchor_index"].astype(str).str.strip()
        matched = raw[raw["anchor_index"] != ""].copy()
        n_raw = len(raw)
        n_drop = n_raw - len(matched)
        print(f"  CXRLabeledCatalogDataset: labeled rows={n_raw:,}  with_anchor={len(matched):,}  dropped={n_drop:,}")

        if "hours_cxr_to_anchor" in matched.columns:
            hrs = pd.to_numeric(matched["hours_cxr_to_anchor"], errors="coerce")
            in_win = (hrs > lookback_min_hours) & (hrs <= lookback_max_hours)
            if require_hours_in_window:
                before = len(matched)
                matched = matched[in_win].copy()
                print(
                    f"  Filter hours_cxr_to_anchor in ({lookback_min_hours}, {lookback_max_hours}]: "
                    f"{len(matched):,} / {before:,} rows"
                )
            else:
                print(
                    f"  hours_cxr_to_anchor in ({lookback_min_hours}, {lookback_max_hours}]: "
                    f"{int(in_win.sum()):,}/{len(matched):,} ({100*in_win.mean():.1f}%)"
                )

        matched["supertable_datetime"] = pd.to_datetime(matched["supertable_datetime"], errors="coerce")
        matched["dicom_id"] = matched["dicom_id"].map(_norm_dicom_id)
        matched["subject_id"] = pd.to_numeric(matched["subject_id"], errors="coerce")
        matched = matched[matched["supertable_datetime"].notna() & matched["subject_id"].notna()].copy()
        matched["subject_id"] = matched["subject_id"].astype(np.int64)

        if metadata_path and os.path.isfile(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
            matched = matched.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        matched["_anchor_key"] = (
            matched["anchor_hadm_id"].astype(str) + "|" + matched["anchor_index"].astype(str)
        )

        groups: List[pd.DataFrame] = []
        for _, grp in matched.groupby("_anchor_key", sort=False):
            grp = grp.sort_values("supertable_datetime").drop_duplicates(subset=["dicom_id"], keep="first")
            groups.append(grp.reset_index(drop=True))

        self.groups = groups
        n = len(groups)
        self.anchor_has_s2f = np.zeros(n, dtype=bool)
        self.anchor_has_p2f = np.zeros(n, dtype=bool)
        self.anchor_s2f_cls = np.full(n, -1, dtype=np.int64)
        self.anchor_p2f_cls = np.full(n, -1, dtype=np.int64)
        self.anchor_index_str = []
        self.n_cxr_rows = []

        for i, grp in enumerate(groups):
            row0 = grp.iloc[0]
            self.anchor_index_str.append(str(row0["anchor_index"]))
            self.anchor_has_s2f[i] = _to_has_flag(row0["has_s2f_vent_fio2"])
            self.anchor_has_p2f[i] = _to_has_flag(row0["has_p2f_vent_fio2"])
            self.anchor_s2f_cls[i] = _to_class_label(row0["s2f_vent_fio2_severity_change_12to24h"])
            self.anchor_p2f_cls[i] = _to_class_label(row0["p2f_vent_fio2_severity_change_12to24h"])
            self.n_cxr_rows.append(len(grp))

        seq_lens = np.array([max(1, x) for x in self.n_cxr_rows], dtype=np.int32)
        print(f"  Unique anchors (samples): n={n:,}")
        print(
            f"  CXRs per anchor: min={seq_lens.min()}, median={int(np.median(seq_lens))}, "
            f"max={seq_lens.max()}, mean={seq_lens.mean():.2f}"
        )
        self._print_label_stats()
        self._print_loadable_cxr_sample()

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

    def _print_loadable_cxr_sample(self, sample_size: int = 300) -> None:
        n = len(self)
        if n == 0:
            return
        rng = np.random.RandomState(0)
        idxs = rng.choice(n, size=min(sample_size, n), replace=False)
        ok_anchors = 0
        for idx in idxs:
            if self._count_loadable(int(idx)) > 0:
                ok_anchors += 1
        print(
            f"  Loadable CXR (sample {len(idxs)} anchors): {ok_anchors}/{len(idxs)} "
            f"({100.0 * ok_anchors / len(idxs):.1f}%) have >=1 image on disk"
        )

    def _count_loadable(self, idx: int) -> int:
        grp = self.groups[idx]
        n = 0
        for _, row in grp.iterrows():
            if self._row_has_cxr(row):
                n += 1
        return n

    def _row_has_cxr(self, row: pd.Series) -> bool:
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        if not (pd.notna(dicom_id) and pd.notna(subject_id)):
            return False
        study_id = _first_non_empty_study_id(row)
        path = get_cxr_path(_norm_dicom_id(dicom_id), int(subject_id), study_id, self.cxr_root)
        return bool(path and os.path.isfile(path))

    def _load_cxr(self, row: pd.Series, cxr_split: Optional[str] = None) -> Tuple[torch.Tensor, bool]:
        split = cxr_split if cxr_split is not None else self.cxr_split
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        study_id = _first_non_empty_study_id(row)
        if pd.notna(dicom_id) and pd.notna(subject_id):
            dicom_id = _norm_dicom_id(dicom_id)
            path = get_cxr_path(dicom_id, int(subject_id), study_id, self.cxr_root)
            if path and os.path.isfile(path):
                return load_cxr(path, split, imagenet_normalize=self.imagenet_normalize), True
        return torch.zeros(3, 224, 224), False

    def get_anchor_item(self, idx: int, cxr_split: Optional[str] = None):
        """Load one anchor sample; optional cxr_split overrides instance default."""
        grp = self.groups[idx]
        cxrs, m_cxr = [], []
        for _, row in grp.iterrows():
            img, ok = self._load_cxr(row, cxr_split=cxr_split)
            cxrs.append(img)
            m_cxr.append(ok)
        if not cxrs:
            cxrs = [torch.zeros(3, 224, 224)]
            m_cxr = [False]
        return {
            "cxr_seq": torch.stack(cxrs, dim=0),
            "cxr_mask": torch.tensor(m_cxr, dtype=torch.bool),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
        }

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        return self.get_anchor_item(idx)


class CXRLabeledCatalogView(Dataset):
    """Index view into CXRLabeledCatalogDataset with a fixed train/val/test crop split."""

    def __init__(
        self,
        base: CXRLabeledCatalogDataset,
        indices: np.ndarray,
        cxr_split: str,
    ):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cxr_split = cxr_split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base.get_anchor_item(int(self.indices[i]), cxr_split=self.cxr_split)
