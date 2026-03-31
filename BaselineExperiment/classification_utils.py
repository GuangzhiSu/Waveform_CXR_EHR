"""Stratified splits and class-weighted loss for 3-class ARDS severity baselines."""
import warnings
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def stratified_train_val_test_indices(
    y: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return index arrays for train / val / test with stratification by label.
    Falls back to random indices if stratify is not possible (e.g. too few per class).
    """
    y = np.asarray(y).astype(np.int64)
    n = len(y)
    assert abs(train_split + val_split + test_split - 1.0) < 1e-5, "splits must sum to 1"
    idx = np.arange(n, dtype=np.int64)
    rest_frac = val_split + test_split

    try:
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            idx, y, test_size=rest_frac, stratify=y, random_state=seed
        )
        frac_test_in_rest = test_split / rest_frac
        idx_val, idx_test, _, _ = train_test_split(
            idx_rest, y_rest, test_size=frac_test_in_rest, stratify=y_rest, random_state=seed
        )
        return idx_train, idx_val, idx_test
    except ValueError as e:
        warnings.warn(f"Stratified split failed ({e}); using random_split-equivalent indices.")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        n_test = n - n_train - n_val
        idx_train = perm[:n_train]
        idx_val = perm[n_train : n_train + n_val]
        idx_test = perm[n_train + n_val :]
        return idx_train, idx_val, idx_test


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights, mean-normalized; for CrossEntropyLoss(weight=...)."""
    y_train = np.asarray(y_train, dtype=np.int64)
    counts = np.bincount(y_train, minlength=num_classes)
    counts = np.maximum(counts.astype(np.float64), 1.0)
    n = y_train.size
    w = n / (num_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_subset(dataset, indices: np.ndarray) -> Subset:
    return Subset(dataset, indices.tolist())
