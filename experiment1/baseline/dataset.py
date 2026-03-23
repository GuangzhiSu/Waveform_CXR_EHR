"""
Dataset for cxr_supertable_waveform_matched: CXR + ECG waveform.
EHR oxygenation (e.g. spo2) is used as ground truth only, not as model input.

Goal: Predict oxygenation from ECG + CXR; EHR provides the labels.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import resample
from torch.utils.data import Dataset
from torchvision import transforms
import wfdb

# Reuse MedTVT-R1 data loading where possible
_MEDTVT_ROOT = Path(__file__).resolve().parents[2] / "MedTVT-R1"
if str(_MEDTVT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_MEDTVT_ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Oxygenation columns usable as ground truth (EHR, not model input)
OXYGENATION_COLS = ["spo2", "partial_pressure_of_oxygen_(pao2)", "s2f_vent_fio2", "p2f_vent_fio2", "saturation_of_oxygen_(sao2)"]


def get_cxr_path(dicom_id, subject_id, study_id, cxr_root):
    """Build MIMIC-CXR-JPG path: files/p{first3}/p{subject_id}/s{study_id}/{dicom_id}.jpg"""
    s = str(int(subject_id))
    first3 = s[:3] if len(s) >= 3 else s.zfill(3)
    return os.path.join(cxr_root, "files", f"p{first3}", f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")


def load_cxr(path, split="train"):
    """Load and preprocess CXR image."""
    img = Image.open(path).convert("RGB")
    crop = transforms.RandomCrop((224, 224)) if split == "train" else transforms.CenterCrop((224, 224))
    transform = transforms.Compose([
        transforms.Resize(400),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img)


def load_ecg(path, target_len=1000):
    """Load ECG from WFDB path. Resample to target_len for xresnet1d101."""
    rec = wfdb.rdsamp(path)
    ecg = torch.from_numpy(rec[0].T).float()  # (12, L)
    L = ecg.shape[1]
    ecg_np = resample(ecg.numpy(), target_len, axis=1)
    ecg = torch.from_numpy(ecg_np).float()
    if torch.any(torch.isnan(ecg)):
        ecg = torch.nan_to_num(ecg, nan=0.0)
    return ecg  # (12, target_len)


class WaveformCXREHRDataset(Dataset):
    """Dataset: CXR + ECG → predict oxygenation. EHR oxygenation = ground truth only.

    Only includes rows with valid (non-NaN) oxygenation in target_col.
    """

    def __init__(
        self,
        csv_path,
        cxr_root="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz",
        target_col="spo2",
        split="train",
        load_cxr=True,
        load_signal=True,
    ):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.cxr_root = cxr_root
        self.target_col = target_col
        self.split = split
        self.load_cxr = load_cxr
        self.load_signal = load_signal

        if target_col not in self.df.columns:
            raise ValueError(f"target_col '{target_col}' not in CSV. Available: {list(self.df.columns)[:20]}...")

        # Keep only rows with valid oxygenation (ground truth from EHR)
        self.df[self.target_col] = pd.to_numeric(self.df[target_col], errors="coerce")
        self.df = self.df[self.df[target_col].notna() & np.isfinite(self.df[target_col])].copy()
        self.df = self.df.reset_index(drop=True)

        # Merge with metadata to get study_id for CXR path
        if metadata_path and os.path.exists(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "subject_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            self.df = self.df.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dicom_id = row["dicom_id"]
        subject_id = row["subject_id"]
        study_id = row.get("study_id", row.get("wf_Study_ID"))
        wf_path = row["wf_File_Path"]

        # CXR (model input) - skip load if not needed (e.g. ECG-only baseline)
        if self.load_cxr:
            cxr_path = get_cxr_path(dicom_id, subject_id, study_id, self.cxr_root)
            if os.path.exists(cxr_path):
                cxr = load_cxr(cxr_path, self.split)
            else:
                cxr = torch.zeros(3, 224, 224)
        else:
            cxr = torch.zeros(3, 224, 224)

        # ECG (model input) - skip load if not needed (e.g. CXR-only baseline)
        if self.load_signal:
            if pd.notna(wf_path) and os.path.exists(wf_path):
                ecg = load_ecg(wf_path)
            else:
                ecg = torch.zeros(12, 1000)
        else:
            ecg = torch.zeros(12, 1000)

        # Ground truth: EHR oxygenation at timestamp
        target = float(row[self.target_col])
        target = torch.tensor(target, dtype=torch.float32)

        return {
            "cxr": cxr,
            "signal": ecg,
            "target": target,
            "dicom_id": dicom_id,
        }
