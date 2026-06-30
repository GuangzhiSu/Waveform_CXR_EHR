"""Masked InfoNCE losses for patient-temporal contrastive learning.

Both losses operate on the similarity matrix S = q @ c_t2^T / temperature, of
shape (B, B). Invalid columns are masked to -inf. The cross-patient loss uses
the diagonal target. The temporal loss is window-aware when ECG/CXR times are
provided: same-patient CXR columns inside the ECG->future-CXR horizon are
multi-positives; same-patient CXR columns outside the horizon are negatives.

cross_patient_loss : valid columns = {j : j == i OR patient_j != patient_i}
                     (same patient's other intervals are ignored, not negatives)
temporal_loss      : valid columns = {j : patient_j == patient_i}
                     positive columns = same-patient CXR targets where at least
                     one ECG in query i predicts target j within [min_h, max_h].
                     Rows with no same-patient temporal negative are skipped.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG_INF = float("-inf")


def _cross_patient_masks(logits: torch.Tensor, patient_ids: torch.Tensor):
    B = logits.size(0)
    same = patient_ids.view(-1, 1) == patient_ids.view(1, -1)
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    valid = (~same) | eye  # different patient, or self (positive)
    positive = eye
    include = torch.ones(B, dtype=torch.bool, device=logits.device)
    return valid, positive, include


def _temporal_masks(
    logits: torch.Tensor,
    patient_ids: torch.Tensor,
    c2_rows: torch.Tensor | None = None,
    c2_times_h: torch.Tensor | None = None,
    ecg_times_h: torch.Tensor | None = None,
    ecg_mask: torch.Tensor | None = None,
    min_horizon_hours: float | None = None,
    max_horizon_hours: float | None = None,
):
    B = logits.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    same = patient_ids.view(-1, 1) == patient_ids.view(1, -1)

    if (c2_times_h is not None and ecg_times_h is not None and ecg_mask is not None
            and min_horizon_hours is not None and max_horizon_hours is not None):
        # Query i may be a single ECG or an ECG sequence. Candidate CXR j is a
        # temporal positive if any valid ECG in query i predicts CXR_j inside
        # the desired future horizon.
        horizons = c2_times_h.view(1, B, 1) - ecg_times_h.unsqueeze(1)  # (B, B, L)
        in_window = ((horizons >= float(min_horizon_hours))
                     & (horizons <= float(max_horizon_hours))
                     & ecg_mask.unsqueeze(1))
        positive = same & in_window.any(dim=-1)
        if c2_rows is not None:
            # Identical target rows are always positives for a query that has
            # this target in-window, never temporal negatives.
            same_target = c2_rows.view(-1, 1) == c2_rows.view(1, -1)
            positive = positive | (same & same_target & positive.any(dim=1, keepdim=True))
        valid = same
        has_positive = positive.any(dim=1)
        has_negative = (valid & ~positive).any(dim=1)
        include = has_positive & has_negative
        return valid, positive, include

    valid = same  # same patient only (self = positive, other intervals = negatives)
    if c2_rows is not None:
        # Identical target CXR (e.g. duplicate-sampled single-interval patients, or two
        # intervals sharing the same t2) must be ignored, not used as a negative.
        same_target = c2_rows.view(-1, 1) == c2_rows.view(1, -1)
        valid = valid & ((~same_target) | eye)
    positive = eye
    include = (valid & ~eye).sum(dim=1) > 0  # need >= 1 genuine same-patient negative
    return valid, positive, include


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


def _masked_multi_positive_ce(logits: torch.Tensor, valid: torch.Tensor,
                              positive: torch.Tensor, include: torch.Tensor):
    """Per-row InfoNCE with one or more positive columns per row."""
    n = int(include.sum().item())
    if n == 0:
        return logits.new_zeros(()), 0
    masked = logits.masked_fill(~valid, _NEG_INF)
    pos_masked = logits.masked_fill(~positive, _NEG_INF)
    log_den = torch.logsumexp(masked, dim=1)
    log_num = torch.logsumexp(pos_masked, dim=1)
    loss_per_row = -(log_num - log_den)
    return loss_per_row[include].mean(), n


def cross_patient_loss(logits: torch.Tensor, patient_ids: torch.Tensor):
    valid, _, _ = _cross_patient_masks(logits, patient_ids)
    return _masked_row_ce(logits, valid, include=None)


def temporal_loss(
    logits: torch.Tensor,
    patient_ids: torch.Tensor,
    c2_rows: torch.Tensor | None = None,
    c2_times_h: torch.Tensor | None = None,
    ecg_times_h: torch.Tensor | None = None,
    ecg_mask: torch.Tensor | None = None,
    min_horizon_hours: float | None = None,
    max_horizon_hours: float | None = None,
):
    valid, positive, include = _temporal_masks(
        logits,
        patient_ids,
        c2_rows=c2_rows,
        c2_times_h=c2_times_h,
        ecg_times_h=ecg_times_h,
        ecg_mask=ecg_mask,
        min_horizon_hours=min_horizon_hours,
        max_horizon_hours=max_horizon_hours,
    )
    if (c2_times_h is not None and ecg_times_h is not None and ecg_mask is not None
            and min_horizon_hours is not None and max_horizon_hours is not None):
        return _masked_multi_positive_ce(logits, valid, positive, include)

    return _masked_row_ce(logits, valid, include=include)


def _masked_topk_counts(logits: torch.Tensor, valid: torch.Tensor,
                        positive: torch.Tensor, include: torch.Tensor, k: int):
    n = int(include.sum().item())
    if n == 0:
        return 0, 0
    k = min(int(k), logits.size(1))
    masked = logits.masked_fill(~valid, _NEG_INF)
    topk = masked.topk(k, dim=1).indices
    hit = positive.gather(1, topk).any(dim=1) & include
    return int(hit.sum().item()), n


def batch_retrieval_metrics(
    logits: torch.Tensor,
    patient_ids: torch.Tensor,
    c2_rows: torch.Tensor | None = None,
    c2_times_h: torch.Tensor | None = None,
    ecg_times_h: torch.Tensor | None = None,
    ecg_mask: torch.Tensor | None = None,
    temporal_min_horizon_hours: float | None = None,
    temporal_max_horizon_hours: float | None = None,
) -> dict:
    """Top-k retrieval counts inside the current contrastive batch.

    Counts use the same masks as the losses above, so same-patient ignored
    columns and multi-positive temporal rows are handled consistently.
    """
    xv, xp, xi = _cross_patient_masks(logits, patient_ids)
    tv, tp, ti = _temporal_masks(
        logits,
        patient_ids,
        c2_rows=c2_rows,
        c2_times_h=c2_times_h,
        ecg_times_h=ecg_times_h,
        ecg_mask=ecg_mask,
        min_horizon_hours=temporal_min_horizon_hours,
        max_horizon_hours=temporal_max_horizon_hours,
    )
    x1, xn = _masked_topk_counts(logits, xv, xp, xi, 1)
    x5, _ = _masked_topk_counts(logits, xv, xp, xi, 5)
    t1, tn = _masked_topk_counts(logits, tv, tp, ti, 1)
    t5, _ = _masked_topk_counts(logits, tv, tp, ti, 5)
    return {
        "cross_patient_top1_correct": x1,
        "cross_patient_top5_correct": x5,
        "cross_patient_rows": xn,
        "temporal_top1_correct": t1,
        "temporal_top5_correct": t5,
        "temporal_rows": tn,
    }


def total_loss(logits: torch.Tensor, patient_ids: torch.Tensor,
               w_cross: float, w_temporal: float, c2_rows: torch.Tensor | None = None,
               c2_times_h: torch.Tensor | None = None,
               ecg_times_h: torch.Tensor | None = None,
               ecg_mask: torch.Tensor | None = None,
               temporal_min_horizon_hours: float | None = None,
               temporal_max_horizon_hours: float | None = None):
    cross, n_cross = cross_patient_loss(logits, patient_ids)
    temp, n_temp = temporal_loss(
        logits, patient_ids, c2_rows=c2_rows, c2_times_h=c2_times_h,
        ecg_times_h=ecg_times_h, ecg_mask=ecg_mask,
        min_horizon_hours=temporal_min_horizon_hours,
        max_horizon_hours=temporal_max_horizon_hours,
    )
    loss = w_cross * cross + w_temporal * temp
    return loss, {
        "loss": float(loss.detach().item()),
        "cross_patient_loss": float(cross.detach().item()),
        "temporal_loss": float(temp.detach().item()),
        "n_cross_rows": n_cross,
        "n_temporal_rows": n_temp,
    }
