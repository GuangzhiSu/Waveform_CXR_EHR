"""EHRNextStepDatasetSymile plus anchor_ehr at t for embed-target loss (not added to input sequence)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_EXP = Path(__file__).resolve().parent
_TRANSFORMER_EXP = _EXP.parent / "EHREncoderTransformer"
for _p in (_EXP.parent / "EHRTrend", _TRANSFORMER_EXP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ehr_symile_dataset import EHRNextStepDatasetSymile  # noqa: E402


class EHRAnchorEmbedDataset(EHRNextStepDatasetSymile):
    """Lookback [t-24h, t-12h] input sequence; separate anchor_ehr@t for embed target only."""

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
            fit_preprocess=fit_preprocess,
            train_anchor_indices=train_anchor_indices,
        )

    def __getitem__(self, idx: int):
        out = super().__getitem__(idx)
        out["anchor_ehr"] = torch.from_numpy(self.anchor_pct[idx]).float()
        return out
