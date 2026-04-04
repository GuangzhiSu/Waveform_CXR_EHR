"""
Dataset for CXR-only ARDS severity classification.
Uses p2f_cxr_classified.csv with p2f_class (0=Severe, 1=Moderate, 2=Mild).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _norm_dicom_id(x):
    """MIMIC-CXR dicom_id may be numeric or a string id (e.g. hex/uuid-style); never force float()."""
    if pd.isna(x) or x == "":
        return ""
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError, OverflowError):
        return s


def _mimic_numeric_path_segment(x):
    """MIMIC folder names use integer IDs; CSV may store floats — avoid ``p19210871.0`` style paths."""
    if pd.isna(x) or x == "":
        return ""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError, OverflowError):
        return str(x).strip()


def _first_non_empty_study_id(row) -> str:
    """Prefer metadata ``study_id``; if merge missed (NaN), use ``wf_Study_ID`` from supertable."""
    sid = row.get("study_id", np.nan)
    if pd.notna(sid) and str(sid).strip() != "":
        return sid
    wf = row.get("wf_Study_ID", np.nan)
    if pd.notna(wf) and str(wf).strip() != "":
        return wf
    return ""


def get_cxr_path(dicom_id, subject_id, study_id, cxr_root):
    """Build MIMIC-CXR-JPG path (PhysioNet layout).

    Example: ``files/p10/p10000032/s50414267/<dicom_id>.jpg`` — the first subfolder uses the
    **first two digits** of ``subject_id`` (e.g. 10000032 → ``p10``), not three.
    See https://physionet.org/content/mimic-cxr-jpg/2.0.0/ (Data Description).
    """
    subj = _mimic_numeric_path_segment(subject_id)
    if not subj:
        return ""
    # Partition folder: p{first_two_chars_of_subject_id} — NOT subj[:3]
    part = subj[:2] if len(subj) >= 2 else subj.zfill(2)
    study = _mimic_numeric_path_segment(study_id)
    dicom_str = _norm_dicom_id(dicom_id)
    if not study or not dicom_str:
        return ""
    return os.path.join(cxr_root, "files", f"p{part}", f"p{subj}", f"s{study}", f"{dicom_str}.jpg")


def load_cxr(path, split="train", imagenet_normalize: bool = True):
    """Load and preprocess CXR image. Default: ImageNet mean/std (matches ViT pretraining)."""
    img = Image.open(path).convert("RGB")
    crop = transforms.RandomCrop((224, 224)) if split == "train" else transforms.CenterCrop((224, 224))
    steps = [
        transforms.Resize(400),
        crop,
        transforms.ToTensor(),
    ]
    if imagenet_normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)(img)


class CXRClassificationDataset(Dataset):
    """Dataset: CXR image -> ARDS severity class (0=Severe, 1=Moderate, 2=Mild)."""

    def __init__(
        self,
        csv_path=None,
        df=None,
        cxr_root="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz",
        split="train",
        indices=None,
        imagenet_normalize: bool = True,
    ):
        """
        Args:
            csv_path: Path to classified CSV (ignored if ``df`` is provided).
            df: Pre-loaded merged DataFrame (e.g. share across train/val/test with different ``split``).
            indices: Row indices into ``df`` for this split (None = use all rows).
            split: ``\"train\"`` uses random crop + aug; anything else uses center crop (eval-style).
            imagenet_normalize: If True, apply ImageNet mean/std after ToTensor (standard for ViT).
        """
        if df is not None:
            self.df = df
        else:
            if not csv_path:
                raise ValueError("Provide csv_path or df.")
            self.df = pd.read_csv(csv_path, low_memory=False)

            if "p2f_class" not in self.df.columns:
                raise ValueError("CSV must have p2f_class column. Run extract_cxr_p2f_classified.py first.")

            self.df = self.df[self.df["p2f_class"].notna()].copy()
            self.df["p2f_class"] = self.df["p2f_class"].astype(int)
            self.df = self.df.reset_index(drop=True)

            # Merge with metadata to get study_id for CXR path
            if metadata_path and os.path.exists(metadata_path):
                meta = pd.read_csv(metadata_path, usecols=["dicom_id", "subject_id", "study_id"])
                meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
                meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
                self.df["dicom_id"] = self.df["dicom_id"].map(_norm_dicom_id)
                self.df = self.df.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")

        self.cxr_root = cxr_root
        self.split = split
        self.imagenet_normalize = imagenet_normalize
        self._indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return len(self.df)

    def __getitem__(self, idx):
        row_idx = int(self._indices[idx]) if self._indices is not None else idx
        row = self.df.iloc[row_idx]
        dicom_id = row["dicom_id"]
        subject_id = row["subject_id"]
        study_id = _first_non_empty_study_id(row)

        cxr_path = get_cxr_path(dicom_id, subject_id, study_id, self.cxr_root)
        if cxr_path and os.path.isfile(cxr_path):
            cxr = load_cxr(cxr_path, self.split, imagenet_normalize=self.imagenet_normalize)
        else:
            cxr = torch.zeros(3, 224, 224)

        label = int(row["p2f_class"])
        return {"cxr": cxr, "label": label, "dicom_id": dicom_id}
