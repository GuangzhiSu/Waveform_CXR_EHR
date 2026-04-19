"""
Temporal CXR dataset: all chest X-rays in [t-24h, t-12h] per anchor (same window as EHR/ECG baselines).
Uses supertable_datetime as CXR time for windowing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_BASE = Path(__file__).resolve().parents[1]
if str(_BASE / "CXRUni") not in sys.path:
    sys.path.insert(0, str(_BASE / "CXRUni"))
from cxr_classification.dataset import (
    _first_non_empty_study_id,
    _norm_dicom_id,
    get_cxr_path,
    load_cxr,
)


class CXRTemporalClassificationDataset(Dataset):
    """
    One sample = all CXR studies in lookback window for same subject;
    label = p2f_class at anchor time (index).
    """

    def __init__(
        self,
        csv_path: str,
        cxr_root: str,
        metadata_path: str | None = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        split: str = "train",
        imagenet_normalize: bool = True,
    ):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.cxr_root = cxr_root
        self.split = split
        self.imagenet_normalize = imagenet_normalize

        if "p2f_class" not in self.df.columns:
            raise ValueError("CSV must have p2f_class")
        for c in ("index", "subject_id", "dicom_id", "supertable_datetime"):
            if c not in self.df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        self.df = self.df[self.df["p2f_class"].notna()].copy()
        self.df["p2f_class"] = self.df["p2f_class"].astype(int)
        self.df["subject_id"] = pd.to_numeric(self.df["subject_id"], errors="coerce")
        self.df["index"] = pd.to_datetime(self.df["index"], errors="coerce")
        self.df["supertable_datetime"] = pd.to_datetime(self.df["supertable_datetime"], errors="coerce")
        self.df = self.df[self.df["subject_id"].notna() & self.df["index"].notna()].copy()
        self.df["subject_id"] = self.df["subject_id"].astype(np.int64)
        self.df = self.df.reset_index(drop=True)

        if metadata_path and os.path.exists(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "subject_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
            self.df["dicom_id"] = self.df["dicom_id"].map(_norm_dicom_id)
            self.df = self.df.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        anchor = self.df[["subject_id", "index", "p2f_class"]].drop_duplicates(
            subset=["subject_id", "index"], keep="first"
        )
        self.anchor_df = anchor.reset_index(drop=True)

        hist = self.df[self.df["supertable_datetime"].notna()].copy().reset_index(drop=True)
        self.hist_df = hist
        self.hist_subject = hist["subject_id"].to_numpy(dtype=np.int64)
        self.hist_time_ns = hist["supertable_datetime"].astype("int64").to_numpy()
        order = np.argsort(self.hist_time_ns)
        hs = self.hist_subject[order]
        ht = self.hist_time_ns[order]
        hi = order
        self.by_subject = {}
        uniq, starts = np.unique(hs, return_index=True)
        for j, s in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(hs)
            self.by_subject[int(s)] = (ht[a:b], hi[a:b])

        self.anchor_subject = self.anchor_df["subject_id"].to_numpy(dtype=np.int64)
        self.anchor_time_ns = self.anchor_df["index"].astype("int64").to_numpy()
        self.labels = self.anchor_df["p2f_class"].to_numpy(dtype=np.int64)

        self.lb_lo_ns = int(lookback_max_hours * 3600 * 1e9)
        self.lb_hi_ns = int(lookback_min_hours * 3600 * 1e9)

        seq_lens = np.array([self._window_indices(i).size for i in range(len(self.anchor_df))], dtype=np.int32)
        seq_lens[seq_lens == 0] = 1
        print(
            f"  CXR temporal dataset: anchors={len(self.anchor_df):,}, history_rows={len(self.hist_df):,}, "
            f"lookback=[t-{lookback_max_hours}h, t-{lookback_min_hours}h]"
        )
        print(f"  CXR sequence lengths: min={seq_lens.min()} median={int(np.median(seq_lens))} max={seq_lens.max()}")

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

    def __getitem__(self, idx: int):
        win = self._window_indices(idx)
        if win.size == 0:
            sid = int(self.anchor_subject[idx])
            t = int(self.anchor_time_ns[idx])
            nearest = self.df[
                (self.df["subject_id"] == sid)
                & (self.df["index"].astype("int64") == t)
                & self.df["supertable_datetime"].notna()
            ]
            if len(nearest) > 0:
                seq_rows = nearest.iloc[[0]]
            else:
                seq_rows = self.hist_df.iloc[[0]]
        else:
            seq_rows = self.hist_df.iloc[win]

        imgs = []
        for _, row in seq_rows.iterrows():
            dicom_id = row["dicom_id"]
            subject_id = row["subject_id"]
            study_id = _first_non_empty_study_id(row)
            p = get_cxr_path(dicom_id, subject_id, study_id, self.cxr_root)
            if p and os.path.isfile(p):
                imgs.append(load_cxr(p, self.split, imagenet_normalize=self.imagenet_normalize))
            else:
                imgs.append(torch.zeros(3, 224, 224))

        cxr_seq = torch.stack(imgs, dim=0)
        return {"cxr_seq": cxr_seq, "label": int(self.labels[idx])}
