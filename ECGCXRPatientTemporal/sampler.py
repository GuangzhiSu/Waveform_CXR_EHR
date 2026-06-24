"""N-patients x K-intervals batch sampler.

Each batch contains N patients and K intervals per patient (batch size = N*K), so
both cross-patient negatives and same-patient (temporal) negatives are present in
every batch.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class NPatientsKIntervalsSampler(Sampler):
    def __init__(self, patient_ids: np.ndarray, n_patients: int, k_intervals: int,
                 num_batches: int | None = None, seed: int = 42, drop_last: bool = True):
        self.patient_ids = np.asarray(patient_ids)
        self.n_patients = int(n_patients)
        self.k_intervals = int(k_intervals)
        self.seed = int(seed)
        self.drop_last = drop_last
        self._epoch = 0

        self.by_patient: dict[int, list] = {}
        for idx, pid in enumerate(self.patient_ids):
            self.by_patient.setdefault(int(pid), []).append(idx)
        # Need >= 2 intervals to form within-patient temporal negatives; allow K with replacement.
        self.eligible = [pid for pid, idxs in self.by_patient.items() if len(idxs) >= 1]

        if num_batches is None:
            total = len(self.patient_ids)
            num_batches = max(1, total // (self.n_patients * self.k_intervals))
        self.num_batches = int(num_batches)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        eligible = list(self.eligible)
        for _ in range(self.num_batches):
            if len(eligible) >= self.n_patients:
                chosen = rng.choice(eligible, size=self.n_patients, replace=False)
            else:
                chosen = rng.choice(eligible, size=self.n_patients, replace=True)
            batch = []
            for pid in chosen:
                idxs = self.by_patient[int(pid)]
                replace = len(idxs) < self.k_intervals
                picks = rng.choice(idxs, size=self.k_intervals, replace=replace)
                batch.extend(int(x) for x in picks)
            yield batch
