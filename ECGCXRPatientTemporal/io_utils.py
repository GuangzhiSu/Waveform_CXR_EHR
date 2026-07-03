"""I/O helpers owned by the ECG-CXR temporal contrastive pipeline."""
from __future__ import annotations

import os

import pandas as pd


def norm_dicom_id(value) -> str:
    """Normalize MIMIC-CXR dicom ids without assuming they are numeric."""
    if pd.isna(value) or value == "":
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (ValueError, TypeError, OverflowError):
        return text


def mimic_numeric_path_segment(value) -> str:
    """Return the integer-like path segment used by MIMIC folder names."""
    if pd.isna(value) or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


def get_cxr_path(dicom_id, subject_id, study_id, cxr_root: str) -> str:
    """Build the MIMIC-CXR-JPG file path for a dicom id."""
    subj = mimic_numeric_path_segment(subject_id)
    if not subj:
        return ""
    part = subj[:2] if len(subj) >= 2 else subj.zfill(2)
    study = mimic_numeric_path_segment(study_id)
    dicom = norm_dicom_id(dicom_id)
    if not study or not dicom:
        return ""
    return os.path.join(cxr_root, "files", f"p{part}", f"p{subj}", f"s{study}", f"{dicom}.jpg")


def load_ecg(path: str, target_len: int = 1000) -> torch.Tensor:
    """Load a 12-lead ECG waveform and resample to ``target_len`` samples."""
    import torch
    import wfdb
    from scipy.signal import resample

    try:
        record = wfdb.rdsamp(path)
        ecg = torch.from_numpy(record[0].T).float()
    except Exception:
        return torch.zeros(12, target_len)

    ecg_np = resample(ecg.numpy(), target_len, axis=1)
    ecg = torch.from_numpy(ecg_np).float()
    return torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
