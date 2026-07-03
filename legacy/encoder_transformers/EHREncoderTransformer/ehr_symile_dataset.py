"""EHRNextStepDataset with Symile-style percentile + presence-indicator preprocessing."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ehr_nextstep_dataset import EHRNextStepDataset
from ehr_symile_preprocess import SymilePreprocessState, fit_symile_preprocessors, transform_symile_rows


class EHRNextStepDatasetSymile(EHRNextStepDataset):
    """
    Symile-MIMIC style row features: train-only ECDF percentiles concatenated with
    per-feature presence indicators (1=observed, 0=missing). Input dim = 2 * n_features.

    Call ``fit_preprocess(train_anchor_indices)`` after stratified split and before training.
    """

    def __init__(
        self,
        anchor_source_csv: str,
        history_csv: str,
        schema_csv: str,
        enriched_csv: Optional[str] = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        include_anchor_row: bool = False,
        fit_preprocess: bool = False,
        train_anchor_indices: Optional[np.ndarray] = None,
    ):
        super().__init__(
            anchor_source_csv=anchor_source_csv,
            history_csv=history_csv,
            schema_csv=schema_csv,
            enriched_csv=enriched_csv,
            lookback_min_hours=lookback_min_hours,
            lookback_max_hours=lookback_max_hours,
            include_anchor_row=include_anchor_row,
        )
        self._base_feature_dim = len(self.feature_cols)
        self.input_dim = 2 * self._base_feature_dim
        self._symile_state: Optional[SymilePreprocessState] = None
        self._symile_fitted = False

        # Discard legacy percentile arrays from parent; require Symile transform.
        self.anchor_pct = None
        self.history_pct = None

        if fit_preprocess:
            if train_anchor_indices is None:
                raise ValueError("train_anchor_indices required when fit_preprocess=True")
            self.fit_preprocess(train_anchor_indices)

    def fit_preprocess(self, train_anchor_indices: np.ndarray) -> None:
        """Fit ECDF on train-group history rows and transform all anchor/history rows."""
        train_anchor_indices = np.asarray(train_anchor_indices, dtype=np.int64)
        if train_anchor_indices.size == 0:
            raise ValueError("train_anchor_indices must be non-empty")

        train_group_ids = np.unique(self.anchor_group[train_anchor_indices])
        train_history_mask = np.isin(self.history_group, train_group_ids)

        n_train_hist = int(train_history_mask.sum())
        if n_train_hist == 0:
            raise ValueError("No history rows found for train anchor groups")

        self._symile_state = fit_symile_preprocessors(
            self.history_num,
            train_history_mask,
            self.feature_cols,
        )
        self.history_pct = transform_symile_rows(self.history_num, self._symile_state)
        self.anchor_pct = transform_symile_rows(self.anchor_num, self._symile_state)
        self._symile_fitted = True

        print(
            f"  Symile preprocess: train_history_rows={n_train_hist:,}, "
            f"features={self._base_feature_dim}, input_dim={self.input_dim} (pct+indicator)"
        )

    def _ensure_fitted(self) -> None:
        if not self._symile_fitted or self.history_pct is None or self.anchor_pct is None:
            raise RuntimeError(
                "EHRNextStepDatasetSymile requires fit_preprocess(train_anchor_indices) "
                "before use"
            )

    def __getitem__(self, idx: int):
        self._ensure_fitted()
        return super().__getitem__(idx)
