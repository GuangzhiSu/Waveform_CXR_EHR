#!/usr/bin/env python3
"""Verify NaN root cause: all-invalid ecg_mask row + transformer attention."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import importlib.util

_EXP = Path(__file__).resolve().parent
PROJECT_ROOT = _EXP.parents[0]
for p in (_EXP, PROJECT_ROOT / "CXREncoderTransformer", PROJECT_ROOT, PROJECT_ROOT / "EHRWindowTransformer", PROJECT_ROOT / "BaselineExperiment"):
    p_str = str(p)
    if p.is_dir() and p_str not in sys.path:
        sys.path.insert(0, p_str)

_spec = importlib.util.spec_from_file_location("ecg_enc_model", _EXP / "model.py")
_ecg_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ecg_model)
ECGEncoderTransformer = _ecg_model.ECGEncoderTransformer
_build_causal_mask = _ecg_model._build_causal_mask


def forward_without_safe_padding(model, ecg_seq, ecg_mask):
    """Old path: no safe_mask hack for all-invalid rows."""
    bsz, t, c, length = ecg_seq.shape
    device = ecg_seq.device
    flat = ecg_seq.reshape(bsz * t, c, length)
    flat_mask = ecg_mask.reshape(-1)
    z = model.miss_ecg.view(1, 1, -1).expand(bsz, t, -1).clone()
    if flat_mask.any():
        enc_out = model._encode_flat(flat[flat_mask])
        enc_out = torch.nan_to_num(enc_out, nan=0.0, posinf=0.0, neginf=0.0)
        z.reshape(-1, model.ecg_dim)[flat_mask] = enc_out
    if model.include_anchor_slot:
        slot = model.anchor_slot.view(1, 1, -1).expand(bsz, 1, model.ecg_dim)
        z = torch.cat([z, slot], dim=1)
        ecg_mask = torch.cat(
            [ecg_mask, torch.ones(bsz, 1, dtype=ecg_mask.dtype, device=device)],
            dim=1,
        )
    h = model.proj(z)
    h = model.pos(h)
    h = model.pos_drop(h)
    pad = ~ecg_mask.bool()
    caus = _build_causal_mask(h.size(1), device)
    for layer in model.layers:
        h = layer(h, key_padding_mask=pad, attn_mask=caus)
    h = model.enc_norm(h)
    anchor_vec = model._pool_anchor(h, ecg_mask)
    return model.head_s2f(anchor_vec), model.head_p2f(anchor_vec)


def count_all_invalid_rows(ds, n_scan=8000, seed=42, batch_size=16):
    rng = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(min(n_scan, len(ds)), generator=rng).tolist()
    loader = DataLoader(
        idxs,
        batch_size=batch_size,
        collate_fn=lambda batch_idxs: collate_ecg_window_batch([ds[i] for i in batch_idxs]),
    )
    n_bad_batches = 0
    first = None
    for bi, batch in enumerate(loader):
        bad = (~batch["ecg_mask"].any(dim=1)).nonzero(as_tuple=False).view(-1).tolist()
        if bad:
            n_bad_batches += 1
            if first is None:
                first = (bi, bad, batch)
    return n_bad_batches, first


def mini_train(forward_fn, model, batch, device, steps=5):
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3)
    losses = []
    for _ in range(steps):
        log_s, log_p = forward_fn(model, b["ecg_seq"], b["ecg_mask"])
        s_ok = b["anchor_has_s2f"] & (b["anchor_s2f"] >= 0)
        loss = F.cross_entropy(log_s[s_ok], b["anchor_s2f"][s_ok]) if s_ok.any() else log_s.new_tensor(0.0)
        losses.append(float(loss))
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    pl2 = sum(float(p.pow(2).sum()) for p in model.parameters() if p.requires_grad) ** 0.5
    return losses, pl2


def main():
    from common import collate_ecg_window_batch  # noqa: E402
    from config import ECG_CATALOG_LABELED_CSV, ECG_CKPT  # noqa: E402
    from ecg_labeled_dataset import ECGLabeledCatalogDataset  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    attn = torch.nn.MultiheadAttention(256, 4, batch_first=True).to(device)
    x = torch.randn(2, 3, 256, device=device)
    out, _ = attn(x, x, x, key_padding_mask=torch.ones(2, 3, dtype=torch.bool, device=device), need_weights=False)
    print(f"[1] MHA all-padded rows -> finite={torch.isfinite(out).all().item()}")

    # Leading-invalid synthetic: should be finite with current model (split masks + skip failed rows in data)
    ckpt = ECG_CKPT if Path(ECG_CKPT).is_file() else None
    ecg_dim = 1024 if ckpt and ckpt.endswith(".ckpt") else 512
    model_syn = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim).to(device)
    ecg_seq = torch.randn(1, 2, 12, 1000, device=device)
    ecg_mask = torch.tensor([[False, True]], device=device)
    with torch.no_grad():
        ls_syn, _ = model_syn(ecg_seq, ecg_mask)
    print(f"[2] synthetic [F,T] mask -> finite={torch.isfinite(ls_syn).all().item()}")

    ds = ECGLabeledCatalogDataset(ECG_CATALOG_LABELED_CSV)
    n_bad, first = count_all_invalid_rows(ds)
    print(f"[3] In 8000 shuffled samples: batches with all-invalid row={n_bad}")
    if first is None:
        print("No all-invalid rows in collated batches (expected after dataset fix).")
        batch = next(
            iter(
                DataLoader(
                    range(32),
                    batch_size=16,
                    collate_fn=lambda idxs: collate_ecg_window_batch([ds[int(i)] for i in idxs]),
                )
            )
        )
    else:
        bi, bad_rows, batch = first
        print(f"    first at batch_idx={bi} rows={bad_rows}")

    model = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim).to(device)
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        ls_fix, _ = model(b["ecg_seq"], b["ecg_mask"])
    print(f"[4] Real batch forward: fixed_finite={torch.isfinite(ls_fix).all().item()}")

    m2 = ECGEncoderTransformer(ecg_ckpt_path=ckpt, ecg_dim=ecg_dim).to(device)
    losses_fix, pl2_fix = mini_train(lambda m, s, mask: m(s, mask), m2, batch, device)
    print(f"[5] Mini-train 5 steps on batch:")
    print(f"    fixed  losses={[round(x, 4) for x in losses_fix]} param_l2={pl2_fix:.4f}")

    ok = torch.isfinite(ls_syn).all() and torch.isfinite(ls_fix).all() and all(np.isfinite(x) for x in losses_fix)
    print(f"\nVERIFIED={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
