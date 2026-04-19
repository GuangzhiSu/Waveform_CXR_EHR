"""
Joint EHR + ECG samples aligned on (subject_id, anchor time).
Keeps only anchors present in both EHR and ECG temporal datasets.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ECGUni.dataset import ECGTemporalClassificationDataset
from EHRUni.dataset import EHRClassificationDataset


def _anchor_key(subject_id: int, time_ns: int) -> tuple:
    return (int(subject_id), int(time_ns))


class MultimodalEHRECGDataset(Dataset):
    """
    For each aligned anchor, returns EHR percentile sequence in lookback window
    and ECG waveforms in the same lookback window.
    """

    def __init__(
        self,
        anchor_csv: str,
        history_csv: str,
        schema_csv: str,
        ecg_pool_csv: str,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        normalize_ecg_per_lead: bool = True,
    ):
        self.ehr = EHRClassificationDataset(
            anchor_csv=anchor_csv,
            history_csv=history_csv,
            schema_csv=schema_csv,
            lookback_min_hours=lookback_min_hours,
            lookback_max_hours=lookback_max_hours,
        )
        self.ecg = ECGTemporalClassificationDataset(
            csv_path=ecg_pool_csv,
            lookback_min_hours=lookback_min_hours,
            lookback_max_hours=lookback_max_hours,
            normalize_per_lead=normalize_ecg_per_lead,
        )

        ehr_map = {
            _anchor_key(self.ehr.anchor_subject[i], self.ehr.anchor_time_ns[i]): i
            for i in range(len(self.ehr))
        }
        ecg_map = {
            _anchor_key(self.ecg.anchor_subject[i], self.ecg.anchor_time_ns[i]): i
            for i in range(len(self.ecg))
        }
        common = sorted(set(ehr_map.keys()) & set(ecg_map.keys()))
        if not common:
            raise ValueError(
                "No overlapping anchors between EHR and ECG datasets. "
                "Check anchor times and CSVs (subject_id + index/_ref_time)."
            )

        self.ehr_idx = [ehr_map[k] for k in common]
        self.ecg_idx = [ecg_map[k] for k in common]

        labels_e = [int(self.ehr.anchor_labels[i]) for i in self.ehr_idx]
        labels_c = [int(self.ecg.labels[i]) for i in self.ecg_idx]
        for a, b in zip(labels_e, labels_c):
            if a != b:
                raise ValueError(f"Label mismatch for aligned anchor: EHR={a} ECG={b}")

        self.labels = np.array(labels_e, dtype=np.int64)
        self.input_dim = self.ehr.input_dim

        print(
            f"  MultimodalEHRECG: aligned samples={len(self):,} "
            f"(EHR-only={len(self.ehr):,}, ECG-only={len(self.ecg):,})"
        )

    def __len__(self):
        return len(self.ehr_idx)

    def __getitem__(self, i: int):
        ei = self.ehr_idx[i]
        ci = self.ecg_idx[i]
        e = self.ehr[ei]
        c = self.ecg[ci]
        return {
            "ehr_seq": e["ehr_seq"],
            "signal_seq": c["signal_seq"],
            "label": int(self.labels[i]),
        }
