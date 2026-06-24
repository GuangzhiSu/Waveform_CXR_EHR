"""Masked InfoNCE losses for patient-temporal contrastive learning.

Both losses operate on the similarity matrix S = q @ c_t2^T / temperature, of
shape (B, B). The positive for query i is always the diagonal (its own CXR_t2).
Invalid columns are masked to -inf, then per-row cross-entropy is taken with the
diagonal index as target.

cross_patient_loss : valid columns = {j : j == i OR patient_j != patient_i}
                     (same patient's other intervals are ignored, not negatives)
temporal_loss      : valid columns = {j : patient_j == patient_i}
                     (different patients ignored; rows with no same-patient other
                      interval are skipped)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG_INF = float("-inf")


def _masked_row_ce(logits: torch.Tensor, valid: torch.Tensor,
                   include: torch.Tensor | None = None):
    """Per-row CE with diagonal target over -inf-masked invalid columns.

    Returns (loss, n_rows_used).
    """
    B = logits.size(0)
    masked = logits.masked_fill(~valid, _NEG_INF)
    logp = F.log_softmax(masked, dim=1)
    diag = logp[torch.arange(B, device=logits.device), torch.arange(B, device=logits.device)]
    loss_per_row = -diag  # (B,)
    if include is None:
        return loss_per_row.mean(), B
    n = int(include.sum().item())
    if n == 0:
        return logits.new_zeros(()), 0
    return loss_per_row[include].mean(), n


def cross_patient_loss(logits: torch.Tensor, patient_ids: torch.Tensor):
    B = logits.size(0)
    same = patient_ids.view(-1, 1) == patient_ids.view(1, -1)
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    valid = (~same) | eye  # different patient, or self (positive)
    return _masked_row_ce(logits, valid, include=None)


def temporal_loss(logits: torch.Tensor, patient_ids: torch.Tensor,
                  c2_rows: torch.Tensor | None = None):
    B = logits.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    same = patient_ids.view(-1, 1) == patient_ids.view(1, -1)
    valid = same  # same patient only (self = positive, other intervals = negatives)
    if c2_rows is not None:
        # Identical target CXR (e.g. duplicate-sampled single-interval patients, or two
        # intervals sharing the same t2) must be ignored, not used as a negative.
        same_target = c2_rows.view(-1, 1) == c2_rows.view(1, -1)
        valid = valid & ((~same_target) | eye)
    include = (valid & ~eye).sum(dim=1) > 0  # need >= 1 genuine same-patient negative
    return _masked_row_ce(logits, valid, include=include)


def total_loss(logits: torch.Tensor, patient_ids: torch.Tensor,
               w_cross: float, w_temporal: float, c2_rows: torch.Tensor | None = None):
    cross, n_cross = cross_patient_loss(logits, patient_ids)
    temp, n_temp = temporal_loss(logits, patient_ids, c2_rows=c2_rows)
    loss = w_cross * cross + w_temporal * temp
    return loss, {
        "loss": float(loss.detach().item()),
        "cross_patient_loss": float(cross.detach().item()),
        "temporal_loss": float(temp.detach().item()),
        "n_cross_rows": n_cross,
        "n_temporal_rows": n_temp,
    }
