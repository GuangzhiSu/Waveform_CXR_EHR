"""CXR window dataset using p2f_or_s2f_cxr_catalog.csv for lookback [t-24h, t-12h] CXR sequences."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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
from modality_window_utils import build_modality_index, window_indices_for_anchor  # noqa: E402

CATALOG_REQUIRED_COLS = ("subject_id", "dicom_id", "hadm_id", "supertable_datetime")
ANCHOR_REQUIRED_COLS = ("hadm_id", "index")
LABEL_COLS = (
    "s2f_vent_fio2_severity_change_12to24h",
    "p2f_vent_fio2_severity_change_12to24h",
    "has_s2f_vent_fio2",
    "has_p2f_vent_fio2",
)


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


def _validate_catalog_columns(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in CATALOG_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CXR catalog {path} missing columns: {missing}. Expected {CATALOG_REQUIRED_COLS}")


def _validate_anchor_columns(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in ANCHOR_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Anchor CSV {path} missing columns: {missing}")
    for c in LABEL_COLS:
        if c not in df.columns:
            raise ValueError(f"Anchor/label CSV {path} missing label column: {c}")


def _attach_anchor_labels(anchor: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """Merge label columns onto anchors by (hadm_id, index time)."""
    if all(c in anchor.columns for c in LABEL_COLS):
        return anchor
    lb = label_df[list({*ANCHOR_REQUIRED_COLS, *LABEL_COLS})].copy()
    lb["hadm_id"] = pd.to_numeric(lb["hadm_id"], errors="coerce")
    lb["_ref_time"] = pd.to_datetime(lb["index"], errors="coerce")
    lb = lb[lb["hadm_id"].notna() & lb["_ref_time"].notna()]
    lb["hadm_id"] = lb["hadm_id"].astype(np.int64)
    lb = lb.drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")
    keep = ["hadm_id", "_ref_time", *LABEL_COLS]
    out = anchor.merge(lb[keep], on=["hadm_id", "_ref_time"], how="left", suffixes=("", "_lb"))
    return out


class CXRCatalogWindowDataset(Dataset):
    """
    Anchors + labels from ``p2f_or_s2f_vent_fio2_valid_rows.csv`` (or label_lookup_csv).
    CXR history from ``p2f_or_s2f_cxr_catalog.csv`` (subject_id, dicom_id, supertable_datetime).

    Window = all catalog CXRs for the anchor's subject_id with acquisition time in
    [anchor_t - 24h, anchor_t - 12h].
    """

    def __init__(
        self,
        anchor_source_csv: str,
        cxr_catalog_csv: str,
        label_lookup_csv: Optional[str] = None,
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
        catalog = pd.read_csv(cxr_catalog_csv, low_memory=False)
        _validate_anchor_columns(main, anchor_source_csv)
        _validate_catalog_columns(catalog, cxr_catalog_csv)

        label_path = label_lookup_csv or anchor_source_csv
        labels = pd.read_csv(label_path, low_memory=False)
        _validate_anchor_columns(labels, label_path)

        main = main.copy()
        main["hadm_id"] = pd.to_numeric(main["hadm_id"], errors="coerce")
        main["_ref_time"] = pd.to_datetime(main["index"], errors="coerce")
        main = main[main["hadm_id"].notna() & main["_ref_time"].notna()].copy()
        main["hadm_id"] = main["hadm_id"].astype(np.int64)
        main = _attach_anchor_labels(main, labels)

        catalog = catalog.copy()
        catalog["hadm_id"] = pd.to_numeric(catalog["hadm_id"], errors="coerce")
        catalog["subject_id"] = pd.to_numeric(catalog["subject_id"], errors="coerce")
        catalog["supertable_datetime"] = pd.to_datetime(catalog["supertable_datetime"], errors="coerce")
        catalog["dicom_id"] = catalog["dicom_id"].map(_norm_dicom_id)
        catalog = catalog[
            catalog["hadm_id"].notna()
            & catalog["subject_id"].notna()
            & catalog["supertable_datetime"].notna()
            & catalog["dicom_id"].astype(str).str.len().gt(0)
        ].copy()
        catalog["hadm_id"] = catalog["hadm_id"].astype(np.int64)
        catalog["subject_id"] = catalog["subject_id"].astype(np.int64)

        enr_subj = None
        if enriched_csv and str(enriched_csv).strip() and Path(enriched_csv).is_file():
            enr = pd.read_csv(enriched_csv, low_memory=False, usecols=["hadm_id", "index", "subject_id"])
            enr["hadm_id"] = pd.to_numeric(enr["hadm_id"], errors="coerce")
            enr["_ref_time"] = pd.to_datetime(enr["index"], errors="coerce")
            enr["subject_id"] = pd.to_numeric(enr["subject_id"], errors="coerce")
            enr = enr[enr["hadm_id"].notna() & enr["_ref_time"].notna() & enr["subject_id"].notna()]
            enr = enr.drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")
            enr_subj = enr.rename(columns={"subject_id": "_enr_subject_id"})[
                ["hadm_id", "_ref_time", "_enr_subject_id"]
            ]

        hadm_to_subj = (
            catalog.groupby("hadm_id")["subject_id"]
            .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0]))
            .to_dict()
        )

        def assign_group(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if enr_subj is not None:
                out = out.merge(enr_subj, on=["hadm_id", "_ref_time"], how="left")

            def row_gid(r):
                if enr_subj is not None:
                    s = r.get("_enr_subject_id")
                    if pd.notna(s):
                        return int(s)
                hid = int(r["hadm_id"])
                return hadm_to_subj.get(hid, hid)

            out["_group_id"] = out.apply(row_gid, axis=1)
            if "_enr_subject_id" in out.columns:
                del out["_enr_subject_id"]
            return out

        main = assign_group(main)

        if metadata_path and os.path.isfile(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
            catalog = catalog.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        self.anchor_df = main.reset_index(drop=True)
        self.cxr_catalog_df = catalog.reset_index(drop=True)

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

        self.cxr_hist_df, self.by_group_cxr = build_modality_index(
            self.cxr_catalog_df,
            group_col="subject_id",
            time_col="supertable_datetime",
        )

        n_in_catalog_groups = len(self.by_group_cxr)
        anchors_with_group = sum(1 for g in self.anchor_group if int(g) in self.by_group_cxr)
        seq_lens = np.array(
            [max(1, int(self._window_indices(i).size)) for i in range(n)], dtype=np.int32
        )
        valid_cxr_in_window = np.array(
            [self._window_valid_cxr_count(i) for i in range(min(n, 2000))], dtype=np.int32
        ) if n else np.array([], dtype=np.int32)

        print(f"  CXRCatalogWindowDataset: anchors n={n:,}, catalog rows n={len(self.cxr_catalog_df):,}")
        print(f"  CXR catalog: subjects={n_in_catalog_groups:,}, lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]")
        print(
            f"  Anchors with subject in catalog: {anchors_with_group:,}/{n:,} "
            f"({100.0 * anchors_with_group / max(n, 1):.1f}%)"
        )
        print(
            f"  Sequence length stats: min={seq_lens.min()}, median={int(np.median(seq_lens))}, "
            f"max={seq_lens.max()}"
        )
        if valid_cxr_in_window.size:
            has_any = int((valid_cxr_in_window > 0).sum())
            print(
                f"  Valid CXR files in window (first {valid_cxr_in_window.size} anchors): "
                f"{has_any}/{valid_cxr_in_window.size} ({100.0 * has_any / max(len(valid_cxr_in_window), 1):.1f}%)"
            )
        self._print_modality_coverage()

    def _window_indices(self, idx: int) -> np.ndarray:
        return window_indices_for_anchor(
            self.by_group_cxr,
            int(self.anchor_group[idx]),
            int(self.anchor_time_ns[idx]),
            self.lb_lo_ns,
            self.lb_hi_ns,
        )

    def _row_has_cxr(self, row: pd.Series) -> bool:
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        if not (pd.notna(dicom_id) and pd.notna(subject_id)):
            return False
        study_id = _first_non_empty_study_id(row)
        path = get_cxr_path(_norm_dicom_id(dicom_id), int(subject_id), study_id, self.cxr_root)
        return bool(path and os.path.isfile(path))

    def _window_valid_cxr_count(self, idx: int) -> int:
        win = self._window_indices(idx)
        if win.size == 0:
            return 0
        return sum(1 for wi in win if self._row_has_cxr(self.cxr_hist_df.iloc[int(wi)]))

    def _print_modality_coverage(self, sample_size: int = 500) -> None:
        n = len(self.anchor_df)
        if n == 0:
            return
        rng = np.random.RandomState(0)
        idxs = rng.choice(n, size=min(sample_size, n), replace=False)
        any_valid = 0
        for idx in idxs:
            if self._window_valid_cxr_count(int(idx)) > 0:
                any_valid += 1
        print(
            f"  CXR coverage (sample n={len(idxs)}): "
            f"{any_valid}/{len(idxs)} anchors have >=1 loadable CXR in window "
            f"({100.0 * any_valid / len(idxs):.1f}%)"
        )

    def _load_cxr(self, row: pd.Series) -> Tuple[torch.Tensor, bool]:
        dicom_id = row.get("dicom_id")
        subject_id = row.get("subject_id")
        study_id = _first_non_empty_study_id(row)
        if pd.notna(dicom_id) and pd.notna(subject_id):
            dicom_id = _norm_dicom_id(dicom_id)
            path = get_cxr_path(dicom_id, int(subject_id), study_id, self.cxr_root)
            if path and os.path.isfile(path):
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
                cxr, ok = self._load_cxr(row)
                cxrs.append(cxr)
                m_cxr.append(ok)

        return {
            "cxr_seq": torch.stack(cxrs, dim=0),
            "cxr_mask": torch.tensor(m_cxr, dtype=torch.bool),
            "anchor_s2f_cls": int(self.anchor_s2f_cls[idx]),
            "anchor_p2f_cls": int(self.anchor_p2f_cls[idx]),
            "anchor_has_s2f": bool(self.anchor_has_s2f[idx]),
            "anchor_has_p2f": bool(self.anchor_has_p2f[idx]),
        }
