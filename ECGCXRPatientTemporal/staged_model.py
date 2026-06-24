"""Unified, configurable model for the staged ECG->CXR experiments.

A single module covers Experiments 1-4 and the Experiment-4 shortcut controls,
driven by an :class:`~experiments.ExperimentSpec`:

  Exp 1   single ECG          -> p_ecg                       -> q        (no g)
  Exp 2   single ECG + dt     -> g(concat(p_ecg, t_emb(dt))) -> q
  Exp 3A  ECG seq             -> Transformer -> mean/cls pool -> g -> q
  Exp 3B  ECG seq + Q_future  -> Transformer -> query pool    -> g -> q
  Exp 4   CXR_t1 + ECG seq    -> g(concat(c_t1, pool(Tx)))    -> q

The target is always ``c_t2 = cxr_proj(E_cxr(CXR_t2))``. Everything is L2
normalized, ``S = q @ c_t2^T * exp(logit_scale)``.

Frozen Bio-ViL-T / ECG-CoCa features are precomputed; only the projections,
ECG temporal Transformer, time embeddings, learnable query and predictor ``g``
are trained.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CXRProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TimeEmbedding(nn.Module):
    """Scalar time (hours) -> dense embedding via a small MLP (times normalized by ``scale``)."""

    def __init__(self, out_dim: int, scale: float = 24.0):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(1, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t_hours: torch.Tensor) -> torch.Tensor:
        x = (t_hours / self.scale).unsqueeze(-1)
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class StagedModel(nn.Module):
    def __init__(self, spec, cxr_dim: int, ecg_dim: int, proj_dim: int = 256,
                 cxr_proj_hidden: int = 512, d_model: int = 256, ecg_tx_layers: int = 3,
                 ecg_tx_heads: int = 4, ecg_tx_mlp_ratio: float = 4.0,
                 fusion_hidden: int = 512, time_emb_dim: int = 64, dropout: float = 0.1,
                 temperature: float = 0.07, learnable_temperature: bool = False):
        super().__init__()
        self.spec = spec
        self.proj_dim = proj_dim
        self.d_model = d_model

        # Shared CXR projection (used for c_t1, c_t2 and the retrieval gallery).
        self.cxr_proj = CXRProjection(cxr_dim, cxr_proj_hidden, proj_dim, dropout)

        # ---- ECG branch ----------------------------------------------------
        self.ecg_single_proj = None
        self.single_time_emb = None
        self.ecg_in_proj = None
        self.seq_time_emb = None
        self.encoder = None
        self.enc_norm = None
        self.cls_token = None
        self.future_query = None
        self.future_time_emb = None

        ecg_out_dim = 0
        if spec.use_ecg:
            if spec.ecg_mode == "single":
                if spec.ecg_proj_kind == "mlp":
                    self.ecg_single_proj = MLP(ecg_dim, fusion_hidden, proj_dim, dropout)
                else:
                    self.ecg_single_proj = nn.Linear(ecg_dim, proj_dim)
                ecg_out_dim = proj_dim
                if spec.use_predictor_g and spec.use_time_embedding:
                    self.single_time_emb = TimeEmbedding(time_emb_dim)
                    ecg_out_dim += time_emb_dim
            else:  # sequence
                self.ecg_in_proj = nn.Linear(ecg_dim, d_model)
                if spec.use_time_embedding:
                    self.seq_time_emb = TimeEmbedding(d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=ecg_tx_heads,
                    dim_feedforward=int(d_model * ecg_tx_mlp_ratio), dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=ecg_tx_layers)
                self.enc_norm = nn.LayerNorm(d_model)
                if spec.ecg_pool == "cls":
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                    nn.init.normal_(self.cls_token, std=0.02)
                if spec.use_future_query or spec.ecg_pool == "query":
                    self.future_query = nn.Parameter(torch.zeros(1, 1, d_model))
                    nn.init.normal_(self.future_query, std=0.02)
                    if spec.use_time_embedding:
                        self.future_time_emb = TimeEmbedding(d_model)
                ecg_out_dim = d_model

        # ---- query head ----------------------------------------------------
        g_in = (proj_dim if spec.use_cxr_t1 else 0) + ecg_out_dim
        self.g_in = g_in
        if spec.use_predictor_g:
            assert g_in > 0, "predictor g needs at least one input component"
            self.g = MLP(g_in, fusion_hidden, proj_dim, dropout)
        else:
            # No predictor: query must already be proj_dim and a single component.
            assert g_in == proj_dim, (
                f"without predictor g the query dim must equal proj_dim ({proj_dim}), got {g_in}")
            self.g = None

        # ---- temperature ---------------------------------------------------
        self.learnable_temperature = learnable_temperature
        init_log = math.log(1.0 / temperature)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_log, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    def temperature_value(self) -> float:
        return float(torch.exp(-self.logit_scale).item())

    def _encode_sequence(self, batch) -> torch.Tensor:
        feats = batch["ecg_feats"]
        if self.spec.ecg_perturb == "zero":
            feats = feats * 0.0
        B = feats.size(0)
        h = self.ecg_in_proj(feats)
        if self.seq_time_emb is not None:
            h = h + self.seq_time_emb(batch["ecg_t2t"])  # (B, L, d_model)
        mask = batch["ecg_mask"]  # True = valid

        use_query = self.future_query is not None
        if use_query:
            q_tok = self.future_query.expand(B, 1, -1)
            if self.future_time_emb is not None:
                q_tok = q_tok + self.future_time_emb(batch["delta_t"]).unsqueeze(1)
            h = torch.cat([q_tok, h], dim=1)
            pad = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)
        elif self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)
            pad = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)

        h = self.encoder(h, src_key_padding_mask=~mask.bool())
        h = self.enc_norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        if use_query or self.cls_token is not None:
            return h[:, 0]
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _ecg_vector(self, batch):
        if not self.spec.use_ecg:
            return None
        if self.spec.ecg_mode == "single":
            feats = batch["ecg_feats"][:, 0, :]  # (B, D_ecg), L == 1
            if self.spec.ecg_perturb == "zero":
                feats = feats * 0.0
            z = self.ecg_single_proj(feats)
            if self.single_time_emb is not None:
                z = torch.cat([z, self.single_time_emb(batch["delta_t"])], dim=-1)
            return z
        return self._encode_sequence(batch)

    def encode(self, batch):
        comps = []
        c1 = None
        if self.spec.use_cxr_t1:
            c1 = self.cxr_proj(batch["c1"])
            comps.append(c1)
        ecg_vec = self._ecg_vector(batch)
        if ecg_vec is not None:
            comps.append(ecg_vec)
        fused = torch.cat(comps, dim=-1)
        q = self.g(fused) if self.g is not None else fused
        q = F.normalize(q, dim=-1)
        c2 = self.cxr_proj(batch["c2"])
        return q, c2, c1

    def forward(self, batch):
        q, c2, c1 = self.encode(batch)
        scale = torch.exp(self.logit_scale).clamp(max=100.0)
        logits = (q @ c2.t()) * scale
        return {"q": q, "c2": c2, "c1": c1, "logits": logits, "logit_scale": scale}
