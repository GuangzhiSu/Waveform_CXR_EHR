"""Symile-style EHR preprocessing: train-only NaN-aware ECDF percentiles + presence indicators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class SymilePreprocessState:
    """Per-feature ECDF support and train mean percentiles for imputation."""

    feature_names: List[str]
    sorted_train_vals: Dict[str, np.ndarray]
    mean_percentiles: Dict[str, float]


def fit_symile_ecdf(train_values: np.ndarray) -> np.ndarray:
    """Return sorted non-NaN train values used as ECDF support."""
    vals = np.asarray(train_values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    return np.sort(finite)


def apply_symile_ecdf(values: np.ndarray, sorted_train_vals: np.ndarray) -> np.ndarray:
    """
    Map values to percentiles in [1/n, 1] using right-sided ECDF (Symile NaNAwareECDF).
    NaN inputs remain NaN.
    """
    vals = np.asarray(values, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return out

    nobs = len(sorted_train_vals)
    if nobs == 0:
        return out

    ranks = np.searchsorted(sorted_train_vals, vals[finite_mask], side="right")
    ranks = np.clip(ranks, 1, nobs)
    out[finite_mask] = ranks / nobs
    return out


def fit_symile_preprocessors(
    history_num: pd.DataFrame,
    train_history_mask: np.ndarray,
    feature_names: Sequence[str],
) -> SymilePreprocessState:
    """Fit ECDF per feature on train history rows; compute train mean percentiles."""
    sorted_train_vals: Dict[str, np.ndarray] = {}
    mean_percentiles: Dict[str, float] = {}

    for name in feature_names:
        col = history_num[name].to_numpy(dtype=np.float64)
        train_col = col[train_history_mask]
        support = fit_symile_ecdf(train_col)
        sorted_train_vals[name] = support

        train_pct = apply_symile_ecdf(train_col, support)
        if np.isfinite(train_pct).any():
            mean_percentiles[name] = float(np.nanmean(train_pct))
        else:
            mean_percentiles[name] = 0.5

    return SymilePreprocessState(
        feature_names=list(feature_names),
        sorted_train_vals=sorted_train_vals,
        mean_percentiles=mean_percentiles,
    )


def transform_symile_rows(num_df: pd.DataFrame, state: SymilePreprocessState) -> np.ndarray:
    """
    Transform numeric rows to [percentiles | indicators] with shape (N, 2F).

    Presence indicator: 1 = observed, 0 = missing (Symile convention).
    Missing percentiles are imputed with train mean percentile for that feature.
    """
    n = len(num_df)
    f = len(state.feature_names)
    pct = np.zeros((n, f), dtype=np.float32)
    ind = np.zeros((n, f), dtype=np.float32)

    for j, name in enumerate(state.feature_names):
        raw = num_df[name].to_numpy(dtype=np.float64)
        support = state.sorted_train_vals[name]
        mean_p = state.mean_percentiles[name]
        observed = np.isfinite(raw)

        col_pct = apply_symile_ecdf(raw, support)
        col_pct = np.where(observed, col_pct, mean_p)
        col_pct = np.where(np.isfinite(col_pct), col_pct, mean_p).astype(np.float32)

        pct[:, j] = col_pct
        ind[:, j] = observed.astype(np.float32)

    return np.concatenate([pct, ind], axis=1)
