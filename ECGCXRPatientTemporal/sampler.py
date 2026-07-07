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
                 num_batches: int | None = None, seed: int = 42, drop_last: bool = True,
                 target_rows: np.ndarray | None = None,
                 min_targets_per_patient: int = 1,
                 sample_unique_targets: bool = False):
        self.patient_ids = np.asarray(patient_ids)
        self.n_patients = int(n_patients)
        self.k_intervals = int(k_intervals)
        self.seed = int(seed)
        self.drop_last = drop_last
        self._epoch = 0
        self.target_rows = None if target_rows is None else np.asarray(target_rows)
        self.min_targets_per_patient = int(min_targets_per_patient)
        self.sample_unique_targets = bool(sample_unique_targets)

        self.by_patient: dict[int, list] = {}
        self.targets_by_patient: dict[int, dict[int, list]] = {}
        for idx, pid in enumerate(self.patient_ids):
            self.by_patient.setdefault(int(pid), []).append(idx)
            if self.target_rows is not None:
                target = int(self.target_rows[idx])
                self.targets_by_patient.setdefault(int(pid), {}).setdefault(target, []).append(idx)

        # Temporal-focused runs can require patients with multiple distinct CXR_t2
        # targets, so every selected patient can contribute real within-patient
        # negatives. The default keeps the original behavior.
        self.eligible = []
        for pid, idxs in self.by_patient.items():
            if not idxs:
                continue
            if self.target_rows is None:
                n_targets = len(idxs)
            else:
                n_targets = len(self.targets_by_patient.get(pid, {}))
            if n_targets >= self.min_targets_per_patient:
                self.eligible.append(pid)

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
                batch.extend(self._sample_patient(int(pid), rng))
            yield batch

    def _sample_patient(self, pid: int, rng: np.random.RandomState) -> list[int]:
        idxs = self.by_patient[pid]
        if not self.sample_unique_targets or self.target_rows is None:
            replace = len(idxs) < self.k_intervals
            picks = rng.choice(idxs, size=self.k_intervals, replace=replace)
            return [int(x) for x in picks]

        target_groups = self.targets_by_patient.get(pid, {})
        targets = list(target_groups)
        if not targets:
            replace = len(idxs) < self.k_intervals
            picks = rng.choice(idxs, size=self.k_intervals, replace=replace)
            return [int(x) for x in picks]

        n_unique = min(self.k_intervals, len(targets))
        chosen_targets = rng.choice(targets, size=n_unique, replace=False)
        picks = [int(rng.choice(target_groups[int(t)])) for t in chosen_targets]

        # If K exceeds the number of distinct targets for this patient, fill the
        # remaining slots from all intervals. This only happens when the CLI asks
        # for K larger than the patient's candidate count.
        while len(picks) < self.k_intervals:
            picks.append(int(rng.choice(idxs)))
        return picks
