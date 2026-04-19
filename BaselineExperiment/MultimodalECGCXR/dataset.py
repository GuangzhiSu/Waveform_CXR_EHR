"""
Multimodal dataset: paired ECG + CXR per supertable ``index`` (see ``p2f_ecg_cxr_multimodal.csv``).

Loading matches unimodal pipelines:
  - ECG: ``load_ecg`` + optional per-lead z-score (``ECGClassificationDataset``).
  - CXR: ``load_cxr`` + ImageNet norm + RandomCrop/CenterCrop (``CXRClassificationDataset``).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_BE = Path(__file__).resolve().parents[1]
if str(_BE) not in sys.path:
    sys.path.insert(0, str(_BE))
if str(_BE / "CXRUni") not in sys.path:
    sys.path.insert(0, str(_BE / "CXRUni"))

from cxr_classification.dataset import (  # noqa: E402
    _norm_dicom_id,
    get_cxr_path,
    load_cxr,
    _first_non_empty_study_id,
)
from ECGUni.dataset import load_ecg, normalize_ecg_per_lead  # noqa: E402


class MultimodalECGCXRDataset(Dataset):
    """One row: ECG waveform + CXR image + shared ``p2f_class``."""

    def __init__(
        self,
        csv_path=None,
        df=None,
        cxr_root="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path=None,
        split="train",
        indices=None,
        imagenet_normalize: bool = True,
        normalize_ecg_per_lead: bool = True,
    ):
        if df is not None:
            self.df = df
        else:
            if not csv_path:
                raise ValueError("Provide csv_path or df.")
            self.df = pd.read_csv(csv_path, low_memory=False)

            if "p2f_class" not in self.df.columns:
                raise ValueError("CSV must have p2f_class.")

            self.df = self.df[self.df["p2f_class"].notna()].copy()
            self.df["p2f_class"] = self.df["p2f_class"].astype(int)
            self.df = self.df.reset_index(drop=True)

            if metadata_path and os.path.exists(metadata_path):
                meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
                meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
                meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
                self.df["dicom_id"] = self.df["dicom_id"].map(_norm_dicom_id)
                if "study_id" in self.df.columns:
                    self.df = self.df.drop(columns=["study_id"], errors="ignore")
                self.df = self.df.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        self.cxr_root = cxr_root
        self.split = split
        self.imagenet_normalize = imagenet_normalize
        self.normalize_ecg_per_lead = normalize_ecg_per_lead
        self._indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return len(self.df)

    def __getitem__(self, idx):
        row_idx = int(self._indices[idx]) if self._indices is not None else idx
        row = self.df.iloc[row_idx]

        wf_path = row.get("wf_File_Path")
        if pd.notna(wf_path) and str(wf_path).strip():
            signal = load_ecg(str(wf_path).strip())
        else:
            signal = torch.zeros(12, 1000)
        if self.normalize_ecg_per_lead:
            signal = normalize_ecg_per_lead(signal)

        dicom_id = row["dicom_id"]
        subject_id = row["subject_id"]
        study_id = _first_non_empty_study_id(row)
        cxr_path = get_cxr_path(dicom_id, subject_id, study_id, self.cxr_root)
        if cxr_path and os.path.isfile(cxr_path):
            cxr = load_cxr(cxr_path, self.split, imagenet_normalize=self.imagenet_normalize)
        else:
            cxr = torch.zeros(3, 224, 224)

        label = int(row["p2f_class"])
        return {"signal": signal, "cxr": cxr, "label": label, "wf_File_Path": wf_path}
