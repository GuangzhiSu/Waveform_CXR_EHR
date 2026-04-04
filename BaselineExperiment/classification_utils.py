"""Stratified splits and class-weighted loss for 3-class ARDS severity baselines."""
import math
import os
import warnings
from typing import Optional, Sequence, Tuple

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


def total_grad_l2_norm(model: torch.nn.Module) -> Tuple[float, int]:
    """Global L2 norm of all parameter gradients; count of tensors with non-None .grad."""
    sq = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        sq += float(p.grad.detach().data.pow(2).sum().item())
        n += 1
    return (math.sqrt(sq), n)


def print_trainable_param_counts(model: torch.nn.Module, tag: str = "model") -> None:
    """Total trainable vs frozen parameters; optional per-prefix breakdown for CXR-style models."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  [{tag}] trainable_params={trainable:,}  frozen_params={frozen:,}")
    prefixes = (
        ("cxr_encoder.vit", "cxr_encoder.vit"),
        ("cxr_encoder.proj", "cxr_encoder.proj"),
        ("head", "head"),
    )
    parts = []
    for label, pfx in prefixes:
        t = f = 0
        for name, p in model.named_parameters():
            if not name.startswith(pfx):
                continue
            if p.requires_grad:
                t += p.numel()
            else:
                f += p.numel()
        if t or f:
            parts.append(f"{label}: trainable={t:,} frozen={f:,}")
    if parts:
        print(f"  [{tag}] " + " | ".join(parts))


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


def print_tensor_batch_diagnostics(modality: str, x: torch.Tensor, dead_std_threshold: float = 1e-5) -> None:
    """Print min/max/mean/std; count samples with ~zero variance (empty / constant tensor)."""
    x = x.detach().float()
    B = x.size(0)
    flat = x.view(B, -1)
    per_std = flat.std(dim=1)
    n_dead = int((per_std < dead_std_threshold).sum().item())
    spread = flat.max(dim=1).values - flat.min(dim=1).values
    n_const = int((spread < dead_std_threshold).sum().item())
    tag = modality.upper()
    print(
        f"  [diag {tag} tensor] shape={tuple(x.shape)}  "
        f"global min/max/mean/std: {x.min().item():.6f} / {x.max().item():.6f} / "
        f"{x.mean().item():.6f} / {x.std().item():.6f}"
    )
    print(
        f"  [diag {tag} tensor] dead samples (per-sample std<{dead_std_threshold}): {n_dead}/{B}  "
        f"constant (max≈min): {n_const}/{B}"
    )


def scan_ecg_train_files(train_ds, max_scan: int = 512, seed: int = 0) -> None:
    """Random subset: WFDB path on disk + wfdb.rdsamp success + non-zero tensor after dataset load."""
    try:
        import wfdb
    except ImportError:
        wfdb = None
    rng = np.random.default_rng(seed)
    n = min(len(train_ds), max_scan)
    idxs = rng.choice(len(train_ds), n, replace=False)
    path_hit = 0
    wfdb_nonzero = 0
    sig_nonzero = 0
    for i in idxs:
        item = train_ds[int(i)]
        p = item.get("wf_File_Path")
        if p is not None and str(p).strip():
            ps = str(p).strip()
            if os.path.exists(ps) or os.path.exists(ps + ".hea"):
                path_hit += 1
            if wfdb is not None:
                try:
                    rec = wfdb.rdsamp(ps)
                    if rec[0].size and float(np.abs(rec[0]).mean()) > 1e-12:
                        wfdb_nonzero += 1
                except Exception:
                    pass
        sig = item["signal"]
        if float(sig.abs().mean()) > 1e-8:
            sig_nonzero += 1
    print(
        f"  [diag ECG files] random {n} train idx: path exists (.hea ok): {path_hit}/{n}  "
        f"wfdb rdsamp non-empty: {wfdb_nonzero}/{n}  "
        f"after norm |signal|>1e-8: {sig_nonzero}/{n}"
    )
    if path_hit > 0 and wfdb_nonzero == 0:
        print(
            "  [diag ECG files] WARNING: paths exist but wfdb never returned non-empty samples — "
            "check record path format, permissions, or corrupt segments."
        )
    if wfdb_nonzero > 0 and sig_nonzero == 0:
        print(
            "  [diag ECG files] WARNING: wfdb rdsamp OK but dataset |signal|~0 after norm — "
            "often caused by __getitem__ skipping load_ecg when os.path.exists(wf_path) is False "
            "(WFDB only guarantees record.hea). Use dataset that always calls load_ecg when path is non-empty."
        )


def scan_cxr_train_files(train_ds, max_scan: int = 512, seed: int = 0) -> None:
    """Random subset: JPG on disk + non-zero tensor after load (same as __getitem__)."""
    from cxr_classification.dataset import _first_non_empty_study_id, get_cxr_path

    rng = np.random.default_rng(seed)
    n = min(len(train_ds), max_scan)
    idxs = rng.choice(len(train_ds), n, replace=False)
    root = train_ds.cxr_root
    jpg_hit = 0
    tensor_nonzero = 0
    for i in idxs:
        row_idx = int(train_ds._indices[i])
        row = train_ds.df.iloc[row_idx]
        p = get_cxr_path(
            row["dicom_id"],
            row["subject_id"],
            _first_non_empty_study_id(row),
            root,
        )
        if p and os.path.isfile(p):
            jpg_hit += 1
        cxr = train_ds[int(i)]["cxr"]
        if float(cxr.abs().mean()) > 1e-8:
            tensor_nonzero += 1
    print(
        f"  [diag CXR files] random {n} train idx: jpg exists: {jpg_hit}/{n}  "
        f"|cxr| mean>1e-8: {tensor_nonzero}/{n}"
    )
    if jpg_hit == 0 and n > 0:
        print(
            f"  [diag CXR files] WARNING: no JPG found under cxr_root={root!r}. "
            "This host may not mount MIMIC-CXR-JPG — use a compute node with /hpc/group/... visible, "
            "or pass --cxr_root to your local mirror. Training on all-black tensors is invalid."
        )


def print_model_forward_spread(
    model,
    train_loader,
    device: torch.device,
    modality: str,
    batch=None,
) -> None:
    """One train batch: if logits identical across samples, inputs/features likely constant."""
    model.eval()
    if batch is None:
        batch = next(iter(train_loader))
    with torch.no_grad():
        if modality == "ecg":
            x = batch["signal"].to(device)
            logits = model(x)
        else:
            x = batch["cxr"].to(device)
            logits = model(x)
    tag = modality.upper()
    if logits.size(0) > 1:
        spread = (logits - logits[0:1]).abs().max().item()
        print(
            f"  [diag {tag} forward] logits max |Δ| vs batch[0]: {spread:.6f}  "
            f"(~0 => identical logits, often empty/constant input or collapsed head)"
        )
    model.train()


def _grad_l2_norm_for_prefix(model, prefix: str) -> Tuple[float, int, int]:
    """Total L2 norm of gradients for params whose name starts with ``prefix``."""
    sq = 0.0
    n_with_grad = 0
    n_missing = 0
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if p.grad is None:
            n_missing += 1
            continue
        sq += float(p.grad.detach().data.pow(2).sum().item())
        n_with_grad += 1
    return math.sqrt(sq), n_with_grad, n_missing


def print_grad_norm_groups(
    model,
    groups: Sequence[Tuple[str, str]],
    tag: str = "collapse grad",
) -> None:
    """
    Print L2 grad norms per module prefix (e.g. encoder vs head).
    Missing grads (frozen or unused) are counted — all ``None`` suggests wrong prefix or frozen block.
    """
    for prefix, label in groups:
        norm, n_ok, n_miss = _grad_l2_norm_for_prefix(model, prefix)
        print(
            f"  [{tag}] {label} (prefix={prefix!r}): ||g||_2={norm:.6f}  "
            f"params_with_grad={n_ok}  grad_is_none={n_miss}"
        )


def print_head_last_linear_stats(model, tag: str = "collapse head") -> None:
    """Norms and bias of the last Linear in ``model.head`` (typical classifier layer)."""
    head = getattr(model, "head", None)
    if head is None:
        return
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    if not linears:
        return
    last = linears[-1]
    wn = float(last.weight.detach().norm().item())
    bn = float(last.bias.detach().norm().item()) if last.bias is not None else 0.0
    b = last.bias.detach().cpu().numpy().tolist() if last.bias is not None else []
    print(
        f"  [{tag}] last Linear: ||W||={wn:.6f}  ||b||={bn:.6f}  bias_per_class={b}"
    )


def print_collapse_batch_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    tag: str = "collapse batch",
) -> None:
    """
    One training batch: logits/softmax shape, prediction vs label histograms, entropy.
    Low mean entropy + argmax mostly one class => collapsed predictions; compare to label counts.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()
    B = logits.size(0)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    probs = torch.softmax(logits, dim=1)
    mean_probs = probs.mean(dim=0)
    ent_batch = (
        -(mean_probs * (mean_probs.clamp_min(1e-12)).log()).sum()
    ).item()
    ent_per_sample = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
    mean_sample_ent = float(ent_per_sample.mean().item())
    pred = logits.argmax(dim=1)
    pred_c = torch.bincount(pred, minlength=num_classes).cpu().numpy().tolist()
    lab_c = torch.bincount(labels, minlength=num_classes).cpu().numpy().tolist()
    logit_mean = logits.mean(dim=0).cpu().numpy().tolist()
    logit_std = logits.std(dim=0).cpu().numpy().tolist()
    print(f"  [{tag}] batch_size={B}")
    print(
        f"  [{tag}] logits mean per class [{', '.join(class_names)}]: "
        f"{[round(x, 4) for x in logit_mean]}"
    )
    print(
        f"  [{tag}] logits std per class: {[round(x, 4) for x in logit_std]}"
    )
    print(
        f"  [{tag}] softmax mean per class: {[round(float(x), 4) for x in mean_probs.cpu().numpy()]}"
    )
    print(
        f"  [{tag}] entropy of batch-mean softmax: {ent_batch:.4f} "
        f"(ln {num_classes}≈{math.log(num_classes):.3f} ≈ uniform; →0 => one class)"
    )
    print(
        f"  [{tag}] mean per-sample entropy: {mean_sample_ent:.4f}"
    )
    print(
        f"  [{tag}] argmax counts [{', '.join(class_names)}]: {pred_c}  "
        f"label counts: {lab_c}"
    )
