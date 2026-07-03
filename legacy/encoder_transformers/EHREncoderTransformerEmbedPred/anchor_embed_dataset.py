"""EHRNextStepDataset plus anchor row features at time t for embed-target loss."""
from __future__ import annotations

from typing import Optional

import torch

from ehr_nextstep_dataset import EHRNextStepDataset


class EHRAnchorEmbedDataset(EHRNextStepDataset):
    """Lookback [t-24h, t-12h] + anchor@t in sequence; separate anchor_ehr for embed target."""

    def __init__(
        self,
        anchor_source_csv: str,
        history_csv: str,
        schema_csv: str,
        enriched_csv: Optional[str] = None,
        lookback_min_hours: int = 12,
        lookback_max_hours: int = 24,
        include_anchor_row: bool = True,
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

    def __getitem__(self, idx: int):
        out = super().__getitem__(idx)
        out["anchor_ehr"] = torch.from_numpy(self.anchor_pct[idx]).float()
        return out
