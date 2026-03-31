"""
Dataset for ECG-only ARDS severity classification.
Uses p2f_ecg_all_classified.csv with p2f_class (0=Severe, 1=Moderate, 2=Mild).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset
import wfdb

# Avoid div-by-zero on flat / missing segments
_EPS = 1e-6


def normalize_ecg_per_lead(ecg: torch.Tensor) -> torch.Tensor:
    """
    Z-score each of the 12 leads over time (per-sample). Matches common ECG pipelines and
    stabilizes scale vs raw WFDB units (mV) for pretrained encoders.
    """
    if ecg.numel() == 0:
        return ecg
    m = ecg.mean(dim=1, keepdim=True)
    s = ecg.std(dim=1, keepdim=True).clamp(min=_EPS)
    out = (ecg - m) / s
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_ecg(path, target_len=1000):
    """Load ECG from WFDB path. Resample to target_len for xresnet1d101."""
    try:
        rec = wfdb.rdsamp(path)
        ecg = torch.from_numpy(rec[0].T).float()  # (12, L)
    except Exception:
        return torch.zeros(12, target_len)
    ecg_np = resample(ecg.numpy(), target_len, axis=1)
    ecg = torch.from_numpy(ecg_np).float()
    if torch.any(torch.isnan(ecg)):
        ecg = torch.nan_to_num(ecg, nan=0.0)
    return ecg  # (12, target_len)


class ECGClassificationDataset(Dataset):
    """Dataset: ECG waveform -> ARDS severity class (0=Severe, 1=Moderate, 2=Mild)."""

    def __init__(self, csv_path, split="train", normalize_per_lead: bool = True):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.split = split
        self.normalize_per_lead = normalize_per_lead

        if "p2f_class" not in self.df.columns:
            raise ValueError("CSV must have p2f_class column. Run extract_ecg_all_p2f_classified.py first.")

        self.df = self.df[self.df["p2f_class"].notna()].copy()
        self.df["p2f_class"] = self.df["p2f_class"].astype(int)
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wf_path = row["wf_File_Path"]

        if pd.notna(wf_path) and str(wf_path).strip() and os.path.exists(wf_path):
            ecg = load_ecg(wf_path)
        else:
            ecg = torch.zeros(12, 1000)

        if self.normalize_per_lead:
            ecg = normalize_ecg_per_lead(ecg)

        label = int(row["p2f_class"])
        return {"signal": ecg, "label": label, "wf_File_Path": wf_path}
