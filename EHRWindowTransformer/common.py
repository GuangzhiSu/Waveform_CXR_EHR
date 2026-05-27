"""Shared training utilities for EHR/CXR/ECG window transformers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def masked_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.new_tensor(0.0)
    y_m = y.clone()
    y_m[~valid] = -100
    return F.cross_entropy(logits, y_m, ignore_index=-100)


def stratify_labels_from_anchor(
    anchor_has_p2f: np.ndarray,
    anchor_p2f_cls: np.ndarray,
    anchor_has_s2f: np.ndarray,
    anchor_s2f_cls: np.ndarray,
) -> np.ndarray:
    n = len(anchor_has_p2f)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if anchor_has_p2f[i] and anchor_p2f_cls[i] >= 0:
            y[i] = int(anchor_p2f_cls[i])
        elif anchor_has_s2f[i] and anchor_s2f_cls[i] >= 0:
            y[i] = 3 + int(anchor_s2f_cls[i])
        else:
            y[i] = 0
    return y


def collate_cxr_window_batch(batch):
    lengths = [b["cxr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    seq = torch.zeros(bsz, max_len, 3, 224, 224, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    anchor_s2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_p2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_has_s2f = torch.zeros(bsz, dtype=torch.bool)
    anchor_has_p2f = torch.zeros(bsz, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["cxr_seq"].shape[0]
        seq[i, :t] = b["cxr_seq"]
        mask[i, :t] = b["cxr_mask"][:t]
        c = b["anchor_s2f_cls"]
        anchor_s2f[i] = c if c >= 0 else -1
        c2 = b["anchor_p2f_cls"]
        anchor_p2f[i] = c2 if c2 >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])
    return {
        "cxr_seq": seq,
        "cxr_mask": mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
    }


def collate_ecg_window_batch(batch):
    lengths = [b["ecg_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    c, L = batch[0]["ecg_seq"].shape[1], batch[0]["ecg_seq"].shape[2]
    seq = torch.zeros(bsz, max_len, c, L, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    anchor_s2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_p2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_has_s2f = torch.zeros(bsz, dtype=torch.bool)
    anchor_has_p2f = torch.zeros(bsz, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["ecg_seq"].shape[0]
        seq[i, :t] = b["ecg_seq"]
        mask[i, :t] = b["ecg_mask"][:t]
        c = b["anchor_s2f_cls"]
        anchor_s2f[i] = c if c >= 0 else -1
        c2 = b["anchor_p2f_cls"]
        anchor_p2f[i] = c2 if c2 >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])
    return {
        "ecg_seq": seq,
        "ecg_mask": mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
    }


def forward_loss_from_logits(batch: dict, log_s: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    device = log_s.device
    s_tgt = batch["anchor_s2f"].to(device)
    p_tgt = batch["anchor_p2f"].to(device)
    s_ok = batch["anchor_has_s2f"].to(device) & (s_tgt >= 0)
    p_ok = batch["anchor_has_p2f"].to(device) & (p_tgt >= 0)
    return masked_ce(log_s, s_tgt, s_ok) + masked_ce(log_p, p_tgt, p_ok)


@torch.no_grad()
def eval_loader(model, loader, device, seq_key: str, mask_key: str) -> dict:
    model.eval()
    tot = 0.0
    n_batches = 0
    acc_s_n = acc_s_d = acc_p_n = acc_p_d = 0.0
    ce_s_sum = ce_p_sum = 0.0
    n_ce_s = n_ce_p = 0
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_s, log_p = model(b[seq_key], b[mask_key])
        ls = masked_ce(log_s, b["anchor_s2f"], b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0))
        lp = masked_ce(log_p, b["anchor_p2f"], b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0))
        tot += float(ls + lp)
        n_batches += 1
        m = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        if m.any():
            ce_s_sum += float(F.cross_entropy(log_s[m], b["anchor_s2f"][m]))
            n_ce_s += 1
            acc_s_n += (log_s[m].argmax(1) == b["anchor_s2f"][m]).float().sum().item()
            acc_s_d += int(m.sum())
        m = b["anchor_has_p2f"] & (b["anchor_p2f"] >= 0)
        if m.any():
            ce_p_sum += float(F.cross_entropy(log_p[m], b["anchor_p2f"][m]))
            n_ce_p += 1
            acc_p_n += (log_p[m].argmax(1) == b["anchor_p2f"][m]).float().sum().item()
            acc_p_d += int(m.sum())
    return {
        "loss": tot / max(n_batches, 1),
        "ce_s2f": ce_s_sum / max(n_ce_s, 1),
        "ce_p2f": ce_p_sum / max(n_ce_p, 1),
        "acc_s2f": acc_s_n / max(acc_s_d, 1),
        "acc_p2f": acc_p_n / max(acc_p_d, 1),
    }
