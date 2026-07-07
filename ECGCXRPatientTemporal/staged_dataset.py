"""Unified dataset for the staged ECG->CXR experiments.

Handles both pair kinds over the same cached frozen embeddings:

  * ``single``  : one ECG -> one future CXR (Experiments 1 & 2).
  * ``sequence``: an ECG sequence (+ optional CXR_t1) -> CXR_t2 (Experiments 3 & 4).

Every item exposes a common schema so a single model / collate / metrics path
works for all experiments:

  patient_id, c2_row (gallery dedup id), c2 (target CXR feat),
  ecg_feats (L, D_ecg), ecg_t2t (L,)  [hours from each ECG to the target CXR],
  ecg_times_h (L,)                    [absolute ECG times in dataset hours],
  c2_time_h                           [absolute target CXR time in dataset hours],
  delta_t (scalar, hours)             [single: that ECG's horizon; sequence: t2 - t1],
  c1 (D_cxr)                          [sequence only].

ECG perturbations for the Experiment-4 shortcut controls:
  * ``zero``    -> handled in the model (ECG features multiplied by 0).
  * ``shuffle`` -> handled here: each item's ECG is replaced by a *different
    patient's* ECG (features + relative times), keeping the true target CXR.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

_TARGET_TIME_FALLBACK = 12.0  # hours, used only if a horizon is missing


@dataclass
class StagedData:
    """Loads a pairs file + cached embeddings, filters, and builds patient splits."""

    pairs_json: str
    kind: str                      # "single" | "sequence"
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
    has_cxr_t1: bool = False

    def __post_init__(self):
        raw = json.load(open(self.pairs_json))
        all_pairs = raw["pairs"]
        self.cxr_emb = np.load(self.cxr_emb_npy)
        self.ecg_emb = np.load(self.ecg_emb_npy)
        cxr_idx = {d: i for i, d in enumerate(json.load(open(self.cxr_ids_json)))}
        ecg_idx = {e: i for i, e in enumerate(json.load(open(self.ecg_ids_json)))}

        kept = []
        if self.kind == "single":
            for p in all_pairs:
                e = ecg_idx.get(p["ecg_id"])
                c = cxr_idx.get(p["cxr_id"])
                if e is None or c is None:
                    continue
                kept.append({
                    "patient_id": int(p["patient_id"]),
                    "c2": int(c),
                    "ecg_rows": [int(e)],
                    "ecg_t2t": [float(p["delta_h"])],      # ECG -> target horizon
                    "ecg_times_h": [float(p["ecg_time_h"])],
                    "c2_time_h": float(p["cxr_time_h"]),
                    "delta_t": float(p["delta_h"]),
                    "c1": None,
                })
        elif self.kind == "sequence":
            has_t1 = bool(all_pairs) and ("cxr_t1" in all_pairs[0])
            for p in all_pairs:
                c2 = cxr_idx.get(p["cxr_t2"])
                if c2 is None:
                    continue
                c1 = None
                if has_t1:
                    c1 = cxr_idx.get(p["cxr_t1"])
                    if c1 is None:
                        continue
                e_rows, e_t2t, e_abs = [], [], []
                t2 = float(p["t2_h"])
                for eid, et in zip(p["ecg_ids"], p["ecg_times_h"]):
                    r = ecg_idx.get(eid)
                    if r is None:
                        continue
                    e_rows.append(int(r))
                    et = float(et)
                    e_t2t.append(t2 - et)                   # hours from ECG to target CXR_t2
                    e_abs.append(et)
                if not e_rows:
                    continue
                # Prediction horizon: builder-provided delta_h (t2 - t1 for Exp4,
                # t2 - last_ecg for Exp3); fall back to time-to-target of last ECG.
                delta = float(p.get("delta_h", min(e_t2t)))
                kept.append({
                    "patient_id": int(p["patient_id"]),
                    "c2": int(c2),
                    "ecg_rows": e_rows,
                    "ecg_t2t": e_t2t,
                    "ecg_times_h": e_abs,
                    "c2_time_h": t2,
                    "delta_t": delta,
                    "c1": int(c1) if c1 is not None else None,
                })
            self.has_cxr_t1 = has_t1
        else:
            raise ValueError(f"Unknown kind={self.kind!r}")

        self.pairs = kept
        print(f"  StagedData[{self.kind}]: kept {len(kept):,}/{len(all_pairs):,} pairs "
              "with cached embeddings")
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
            split_of[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        self.patient_to_split = split_of
        self.split_indices = {"train": [], "val": [], "test": []}
        for idx, p in enumerate(self.pairs):
            self.split_indices[split_of[p["patient_id"]]].append(idx)
        print("  Splits (by patient): "
              + ", ".join(f"{k}={len(v):,}" for k, v in self.split_indices.items()))


class StagedDataset(Dataset):
    """One split of a :class:`StagedData`. ``ecg_perturb`` adds shortcut controls."""

    def __init__(self, data: StagedData, indices: list, ecg_perturb: str = "none",
                 seed: int = 0):
        self.data = data
        self.indices = list(indices)
        self.ecg_perturb = ecg_perturb
        self.has_c1 = bool(getattr(data, "has_cxr_t1", False))
        self._donor = None
        if ecg_perturb == "shuffle":
            self._build_donor_map(seed)

    def _build_donor_map(self, seed: int):
        """Map each item -> a donor item from a *different* patient (for ECG shuffling)."""
        rng = np.random.RandomState(seed)
        pids = np.array([self.data.pairs[i]["patient_id"] for i in self.indices])
        n = len(self.indices)
        donor = np.arange(n)
        order = rng.permutation(n)
        for pos in range(n):
            i = order[pos]
            # try random draws to find a different-patient donor
            for _ in range(20):
                j = rng.randint(n)
                if pids[j] != pids[i]:
                    donor[i] = j
                    break
        self._donor = donor

    def __len__(self):
        return len(self.indices)

    def patient_ids(self) -> np.ndarray:
        return np.array([self.data.pairs[i]["patient_id"] for i in self.indices],
                        dtype=np.int64)

    def target_rows(self) -> np.ndarray:
        return np.array([self.data.pairs[i]["c2"] for i in self.indices],
                        dtype=np.int64)

    def _ecg_from(self, pair) -> tuple:
        rows = np.asarray(pair["ecg_rows"], dtype=np.int64)
        t2t = np.asarray(pair["ecg_t2t"], dtype=np.float64)
        times = np.asarray(pair["ecg_times_h"], dtype=np.float64)
        order = np.argsort(-t2t)  # furthest-from-target first -> chronological order
        rows, t2t, times = rows[order], t2t[order], times[order]
        feats = torch.from_numpy(self.data.ecg_emb[rows].astype(np.float32))  # (L, D)
        return (feats, torch.from_numpy(t2t.astype(np.float32)),
                torch.from_numpy(times.astype(np.float32)))                 # (L,)

    def __getitem__(self, i: int):
        pair = self.data.pairs[self.indices[i]]
        ecg_src = pair
        if self.ecg_perturb == "shuffle":
            ecg_src = self.data.pairs[self.indices[int(self._donor[i])]]
        ecg_feats, ecg_t2t, ecg_times = self._ecg_from(ecg_src)

        item = {
            "patient_id": pair["patient_id"],
            "c2_row": int(pair["c2"]),
            "c2": torch.from_numpy(self.data.cxr_emb[pair["c2"]].astype(np.float32)),
            "ecg_feats": ecg_feats,
            "ecg_t2t": ecg_t2t,
            "ecg_times_h": ecg_times,
            "c2_time_h": float(pair["c2_time_h"]),
            "delta_t": float(pair["delta_t"]),
        }
        if self.has_c1 and pair["c1"] is not None:
            item["c1"] = torch.from_numpy(self.data.cxr_emb[pair["c1"]].astype(np.float32))
        return item


def collate_fn(batch: list) -> dict:
    B = len(batch)
    Lmax = max(b["ecg_feats"].shape[0] for b in batch)
    D_ecg = batch[0]["ecg_feats"].shape[1]
    ecg = torch.zeros(B, Lmax, D_ecg)
    t2t = torch.zeros(B, Lmax)
    ecg_times = torch.zeros(B, Lmax)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["ecg_feats"].shape[0]
        ecg[i, :L] = b["ecg_feats"]
        t2t[i, :L] = b["ecg_t2t"]
        ecg_times[i, :L] = b["ecg_times_h"]
        mask[i, :L] = True
    out = {
        "patient_id": torch.tensor([b["patient_id"] for b in batch], dtype=torch.long),
        "c2_row": torch.tensor([b["c2_row"] for b in batch], dtype=torch.long),
        "c2": torch.stack([b["c2"] for b in batch]),
        "ecg_feats": ecg,
        "ecg_t2t": t2t,
        "ecg_times_h": ecg_times,
        "ecg_mask": mask,
        "c2_time_h": torch.tensor([b["c2_time_h"] for b in batch], dtype=torch.float32),
        "delta_t": torch.tensor([b["delta_t"] for b in batch], dtype=torch.float32),
    }
    if "c1" in batch[0]:
        out["c1"] = torch.stack([b["c1"] for b in batch])
    return out
