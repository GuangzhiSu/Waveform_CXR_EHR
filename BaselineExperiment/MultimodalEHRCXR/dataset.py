"""
Joint EHR + CXR aligned on (subject_id, anchor time).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from EHRUni.dataset import EHRClassificationDataset

from .cxr_temporal_dataset import CXRTemporalClassificationDataset


def _anchor_key(subject_id: int, time_ns: int) -> tuple:
    return (int(subject_id), int(time_ns))


class MultimodalEHRCXRDataset(Dataset):
    """EHR percentile sequence + CXR image sequence in the same lookback window."""

    def __init__(
        self,
        anchor_csv: str,
        history_csv: str,
        schema_csv: str,
        cxr_pool_csv: str,
        cxr_root: str,
        metadata_path: str | None = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        split: str = "train",
        imagenet_normalize: bool = True,
    ):
        self.ehr = EHRClassificationDataset(
            anchor_csv=anchor_csv,
            history_csv=history_csv,
            schema_csv=schema_csv,
            lookback_min_hours=lookback_min_hours,
            lookback_max_hours=lookback_max_hours,
        )
        self.cxr = CXRTemporalClassificationDataset(
            csv_path=cxr_pool_csv,
            cxr_root=cxr_root,
            metadata_path=metadata_path,
            lookback_min_hours=lookback_min_hours,
            lookback_max_hours=lookback_max_hours,
            split=split,
            imagenet_normalize=imagenet_normalize,
        )

        ehr_map = {
            _anchor_key(self.ehr.anchor_subject[i], self.ehr.anchor_time_ns[i]): i
            for i in range(len(self.ehr))
        }
        cxr_map = {
            _anchor_key(self.cxr.anchor_subject[i], self.cxr.anchor_time_ns[i]): i
            for i in range(len(self.cxr))
        }
        common = sorted(set(ehr_map.keys()) & set(cxr_map.keys()))
        if not common:
            raise ValueError(
                "No overlapping anchors between EHR and CXR datasets. "
                "Check subject_id + index alignment across CSVs."
            )

        self.ehr_idx = [ehr_map[k] for k in common]
        self.cxr_idx = [cxr_map[k] for k in common]

        labels_e = [int(self.ehr.anchor_labels[i]) for i in self.ehr_idx]
        labels_x = [int(self.cxr.labels[i]) for i in self.cxr_idx]
        for a, b in zip(labels_e, labels_x):
            if a != b:
                raise ValueError(f"Label mismatch for aligned anchor: EHR={a} CXR={b}")

        self.labels = np.array(labels_e, dtype=np.int64)
        self.input_dim = self.ehr.input_dim

        print(
            f"  MultimodalEHRCXR: aligned samples={len(self):,} "
            f"(EHR-only={len(self.ehr):,}, CXR-only={len(self.cxr):,})"
        )

    def __len__(self):
        return len(self.ehr_idx)

    def __getitem__(self, i: int):
        ei = self.ehr_idx[i]
        xi = self.cxr_idx[i]
        e = self.ehr[ei]
        x = self.cxr[xi]
        return {
            "ehr_seq": e["ehr_seq"],
            "cxr_seq": x["cxr_seq"],
            "label": int(self.labels[i]),
        }
