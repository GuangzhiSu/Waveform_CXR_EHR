"""EHR + CXR + ECG encoders -> fused latent -> causal Transformer -> next fused latent + concat-MLP heads."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_exp_old = _PROJECT_ROOT / "experiment1(old)"
if _exp_old.is_dir() and str(_exp_old) not in sys.path:
    sys.path.insert(0, str(_exp_old))

from models.encoders.cxr import CXREncoder
from models.encoders.ehr import EHRMLPEncoder
from models.encoders.ecg import build_ecg_encoder

from blocks_cxrgen import EncoderBlock, PositionalEmbedding
from model_nextstep import StepDiscMLP


def _build_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    m = torch.triu(torch.ones(t, t, device=device, dtype=torch.float32), diagonal=1)
    return m.masked_fill(m.bool(), float("-inf"))


class MultimodalNextStepModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ehr_embed_dim: int = 256,
        cxr_dim: int = 512,
        ecg_dim: int = 512,
        fuse_dim: int = 256,
        d_model: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_length: int = 512,
        anchor_pool: str = "last",
        vit_path: str = "google/vit-base-patch16-224-in21k",
        freeze_cxr: bool = True,
        ecg_ckpt_path: Optional[str] = None,
        freeze_ecg: bool = True,
        ecg_sig_len: int = 5000,
    ):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.anchor_pool = anchor_pool
        if anchor_pool not in ("last", "mean"):
            raise ValueError("anchor_pool must be 'last' or 'mean'")

        self.ehr_enc = EHRMLPEncoder(input_dim=input_dim, embed_dim=ehr_embed_dim)
        self.cxr_enc = CXREncoder(vit_path=vit_path, hidden_dim=cxr_dim, freeze=freeze_cxr)
        self.ecg_enc = build_ecg_encoder(
            "cnn",
            hidden_dim=ecg_dim,
            ckpt_path=ecg_ckpt_path,
            freeze=freeze_ecg,
            sig_len=ecg_sig_len,
        )

        self.miss_cxr = nn.Parameter(torch.zeros(cxr_dim))
        self.miss_ecg = nn.Parameter(torch.zeros(ecg_dim))

        self.proj_e = nn.Linear(ehr_embed_dim, fuse_dim)
        self.proj_x = nn.Linear(cxr_dim, fuse_dim)
        self.proj_s = nn.Linear(ecg_dim, fuse_dim)
        self.fuse_in = nn.Linear(3 * fuse_dim, d_model)
        self.ln_fuse = nn.LayerNorm(d_model)

        self.pos = PositionalEmbedding(d_model, max_len=max_seq_length)
        self.pos_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.head_next = nn.Linear(d_model, d_model)

        concat_dim = 3 * fuse_dim
        self.anchor_s2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)
        self.anchor_p2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)
        self.disc_s2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)
        self.disc_p2f = StepDiscMLP(concat_dim, num_classes, hidden=(512, 256), dropout=dropout)

    def _pool_concat_last(self, fe: torch.Tensor, fx: torch.Tensor, fs: torch.Tensor, mask: torch.Tensor):
        """Return concat [B, 3*fuse_dim] at last valid timestep."""
        bsz = fe.size(0)
        if self.anchor_pool == "last":
            lengths = mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(bsz, device=fe.device, dtype=torch.long)
            return torch.cat([fe[batch_idx, last_idx], fx[batch_idx, last_idx], fs[batch_idx, last_idx]], dim=-1)
        m = mask.float().unsqueeze(-1)
        num = (fe * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        fe_p = num / denom
        num = (fx * m).sum(dim=1)
        fx_p = num / denom
        num = (fs * m).sum(dim=1)
        fs_p = num / denom
        return torch.cat([fe_p, fx_p, fs_p], dim=-1)

    def forward(
        self,
        ehr_seq: torch.Tensor,
        cxr_seq: torch.Tensor,
        ecg_seq: torch.Tensor,
        ehr_mask: torch.Tensor,
        cxr_mask: torch.Tensor,
        ecg_mask: torch.Tensor,
        return_embeddings: bool = False,
    ):
        bsz, t, _ = ehr_seq.shape
        device = ehr_seq.device

        ze = self.ehr_enc(ehr_seq.reshape(bsz * t, -1)).reshape(bsz, t, -1)

        cxr_flat = cxr_seq.reshape(bsz * t, 3, 224, 224)
        zx_raw = self.cxr_enc(cxr_flat).reshape(bsz, t, -1)
        m_x = cxr_mask.float().unsqueeze(-1)
        zx = zx_raw * m_x + self.miss_cxr * (1.0 - m_x)

        ecg_flat = ecg_seq.reshape(bsz * t, ecg_seq.size(2), ecg_seq.size(3))
        zs_raw = self.ecg_enc(ecg_flat).reshape(bsz, t, -1)
        m_s = ecg_mask.float().unsqueeze(-1)
        zs = zs_raw * m_s + self.miss_ecg * (1.0 - m_s)

        fe = self.proj_e(ze)
        fx = self.proj_x(zx)
        fs = self.proj_s(zs)
        fused_pre = torch.cat([fe, fx, fs], dim=-1)
        fused = self.ln_fuse(self.fuse_in(fused_pre))

        h = self.pos(fused)
        h = self.pos_drop(h)
        pad = ~ehr_mask.bool()
        caus = _build_causal_mask(t, device)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad, attn_mask=caus)
        h = self.enc_norm(h)

        if t > 1:
            pred_next = self.head_next(h[:, :-1, :])
            next_padded = pred_next.new_zeros(bsz, t, self.d_model)
            next_padded[:, : t - 1, :] = pred_next
        else:
            next_padded = h.new_zeros(bsz, 1, self.d_model)

        cat_anchor = self._pool_concat_last(fe, fx, fs, ehr_mask)
        log_s2f = self.anchor_s2f(cat_anchor)
        log_p2f = self.anchor_p2f(cat_anchor)

        flat_cat = fused_pre.reshape(bsz * t, -1)
        s2f_step = self.disc_s2f(flat_cat).reshape(bsz, t, self.num_classes)
        p2f_step = self.disc_p2f(flat_cat).reshape(bsz, t, self.num_classes)

        if return_embeddings:
            return log_s2f, log_p2f, next_padded, s2f_step, p2f_step, fused, h, fused_pre
        return log_s2f, log_p2f, next_padded, s2f_step, p2f_step
