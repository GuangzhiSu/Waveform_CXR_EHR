"""Patient-temporal dataset over cached frozen embeddings.

Each item is one interval sample:
  (patient_id, t1, t2, CXR_t1 emb, [ECG embs in (t1,t2]] + relative-time feats, CXR_t2 emb)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import config as C
    _W_BEFORE = float(C.ECG_WINDOW_BEFORE_HOURS)
    _W_AFTER = float(C.ECG_WINDOW_AFTER_HOURS)
except Exception:  # pragma: no cover - fallback if config not importable
    _W_BEFORE, _W_AFTER = 12.0, 3.0


def _rel_time_feats(t1: float, t2: float, ecg_times: np.ndarray,
                    w_before: float = 12.0, w_after: float = 3.0) -> np.ndarray:
    """Per-ECG relative-time features, anchored at t1 (ECG window = [t1-w_before, t1+w_after]).

    Returns [signed_offset_from_t1, time_to_t2, normalized_position]:
      * signed_offset_from_t1: (ecg - t1) / w_before -> negative before t1, ~0..(w_after/w_before) after.
      * time_to_t2: (t2 - ecg) / horizon -> how far each ECG is from the prediction target
        (also leaks the prediction horizon, since horizon = t2 - t1 varies in [12, 20]).
      * normalized_position: index order within the interval.
    """
    horizon = max(t2 - t1, 1e-6)
    signed = (ecg_times - t1) / max(w_before, 1e-6)
    to = (t2 - ecg_times) / horizon
    L = len(ecg_times)
    if L > 1:
        pos = np.arange(L, dtype=np.float64) / (L - 1)
    else:
        pos = np.full(L, 0.5, dtype=np.float64)
    return np.stack([signed, to, pos], axis=1).astype(np.float32)  # (L, 3)


@dataclass
class PatientTemporalData:
    """Loads pairs + cached embeddings, filters valid pairs, builds split indices."""

    pairs_json: str
    cxr_emb_npy: str
    cxr_ids_json: str
    ecg_emb_npy: str
    ecg_ids_json: str
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    pairs: list = field(default_factory=list)
    cxr_emb: np.ndarray = None
    ecg_emb: np.ndarray = None
    split_indices: dict = field(default_factory=dict)
    patient_to_split: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = json.load(open(self.pairs_json))
        all_pairs = raw["pairs"]
        self.cxr_emb = np.load(self.cxr_emb_npy)
        self.ecg_emb = np.load(self.ecg_emb_npy)
        cxr_ids = json.load(open(self.cxr_ids_json))
        ecg_ids = json.load(open(self.ecg_ids_json))
        cxr_idx = {d: i for i, d in enumerate(cxr_ids)}
        ecg_idx = {e: i for i, e in enumerate(ecg_ids)}

        kept = []
        for p in all_pairs:
            c1 = cxr_idx.get(p["cxr_t1"])
            c2 = cxr_idx.get(p["cxr_t2"])
            if c1 is None or c2 is None:
                continue
            e_rows = [ecg_idx[e] for e in p["ecg_ids"] if e in ecg_idx]
            e_times = [t for e, t in zip(p["ecg_ids"], p["ecg_times_h"]) if e in ecg_idx]
            if not e_rows:
                continue
            kept.append({
                "patient_id": int(p["patient_id"]),
                "t1_h": float(p["t1_h"]), "t2_h": float(p["t2_h"]),
                "c1": int(c1), "c2": int(c2),
                "ecg_rows": e_rows, "ecg_times": e_times,
            })
        self.pairs = kept
        print(f"  PatientTemporalData: kept {len(kept):,}/{len(all_pairs):,} pairs with cached embeddings")
        self._build_splits()

    def _build_splits(self):
        patients = sorted({p["patient_id"] for p in self.pairs})
        rng = np.random.RandomState(self.seed)
        rng.shuffle(patients)
        n = len(patients)
        n_tr = int(self.train_split * n)
        n_va = int(self.val_split * n)
        split_of = {}
        for i, pid in enumerate(patients):
            if i < n_tr:
                split_of[pid] = "train"
            elif i < n_tr + n_va:
                split_of[pid] = "val"
            else:
                split_of[pid] = "test"
        self.patient_to_split = split_of
        self.split_indices = {"train": [], "val": [], "test": []}
        for idx, p in enumerate(self.pairs):
            self.split_indices[split_of[p["patient_id"]]].append(idx)
        print(f"  Splits (by patient): "
              + ", ".join(f"{k}={len(v):,} intervals" for k, v in self.split_indices.items()))


class PatientTemporalDataset(Dataset):
    def __init__(self, data: PatientTemporalData, indices: list):
        self.data = data
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def patient_ids(self) -> np.ndarray:
        return np.array([self.data.pairs[i]["patient_id"] for i in self.indices], dtype=np.int64)

    def __getitem__(self, i: int):
        p = self.data.pairs[self.indices[i]]
        ecg_rows = np.asarray(p["ecg_rows"], dtype=np.int64)
        ecg_times = np.asarray(p["ecg_times"], dtype=np.float64)
        order = np.argsort(ecg_times)
        ecg_rows, ecg_times = ecg_rows[order], ecg_times[order]
        ecg_feats = torch.from_numpy(self.data.ecg_emb[ecg_rows].astype(np.float32))  # (L, D_ecg)
        rel = torch.from_numpy(
            _rel_time_feats(p["t1_h"], p["t2_h"], ecg_times, _W_BEFORE, _W_AFTER)
        )                                                                              # (L, 3)
        return {
            "patient_id": p["patient_id"],
            "c2_row": int(p["c2"]),  # cxr_emb row index of CXR_t2 (for retrieval gallery dedup)
            "c1": torch.from_numpy(self.data.cxr_emb[p["c1"]].astype(np.float32)),      # (D_cxr,)
            "c2": torch.from_numpy(self.data.cxr_emb[p["c2"]].astype(np.float32)),      # (D_cxr,)
            "ecg_feats": ecg_feats,
            "ecg_rel": rel,
        }


def collate_fn(batch: list) -> dict:
    B = len(batch)
    Lmax = max(b["ecg_feats"].shape[0] for b in batch)
    D_ecg = batch[0]["ecg_feats"].shape[1]
    D_rel = batch[0]["ecg_rel"].shape[1]
    ecg = torch.zeros(B, Lmax, D_ecg)
    rel = torch.zeros(B, Lmax, D_rel)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)  # True = valid ECG token
    for i, b in enumerate(batch):
        L = b["ecg_feats"].shape[0]
        ecg[i, :L] = b["ecg_feats"]
        rel[i, :L] = b["ecg_rel"]
        mask[i, :L] = True
    return {
        "patient_id": torch.tensor([b["patient_id"] for b in batch], dtype=torch.long),
        "c2_row": torch.tensor([b["c2_row"] for b in batch], dtype=torch.long),
        "c1": torch.stack([b["c1"] for b in batch]),
        "c2": torch.stack([b["c2"] for b in batch]),
        "ecg_feats": ecg,
        "ecg_rel": rel,
        "ecg_mask": mask,
    }
