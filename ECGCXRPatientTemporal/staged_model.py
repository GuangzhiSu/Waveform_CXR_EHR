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
        self.fusion_mode = getattr(spec, "fusion_mode", "mlp_concat")

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
        self.ecg_out_dim = ecg_out_dim

        # ---- query head ----------------------------------------------------
        g_in = (proj_dim if spec.use_cxr_t1 else 0) + ecg_out_dim
        self.g_in = g_in
        self.g = None

        # Extra fusion heads for the requested Exp4 follow-ups. The default
        # ``mlp_concat`` path below is the original implementation.
        self.cxr_token_proj = None
        self.cross_attn = None
        self.cross_attn_norm = None
        self.cross_ff = None
        self.cross_ff_norm = None
        self.cross_out = None
        self.add_norm = None
        self.add_out = None
        self.pool_score = None
        self.pool_norm = None
        self.pool_out = None
        self.residual_score = None
        self.residual_delta = None
        self.residual_scale = None

        if self.fusion_mode == "mlp_concat":
            if spec.use_predictor_g:
                assert g_in > 0, "predictor g needs at least one input component"
                self.g = MLP(g_in, fusion_hidden, proj_dim, dropout)
            else:
                # No predictor: query must already be proj_dim and a single component.
                assert g_in == proj_dim, (
                    f"without predictor g the query dim must equal proj_dim ({proj_dim}), got {g_in}")
        elif self.fusion_mode in {"cross_attention_norm", "add_norm", "weighted_attn_pool"}:
            assert spec.use_cxr_t1 and spec.use_ecg and spec.ecg_mode == "sequence", (
                f"fusion_mode={self.fusion_mode!r} expects CXR_t1 + ECG sequence inputs")
            assert self.encoder is not None and self.ecg_in_proj is not None
            self.cxr_token_proj = nn.Identity() if proj_dim == d_model else nn.Linear(proj_dim, d_model)
            if self.fusion_mode == "cross_attention_norm":
                self.cross_attn = nn.MultiheadAttention(
                    d_model, ecg_tx_heads, dropout=dropout, batch_first=True)
                self.cross_attn_norm = nn.LayerNorm(d_model)
                self.cross_ff = MLP(d_model, int(d_model * ecg_tx_mlp_ratio), d_model, dropout)
                self.cross_ff_norm = nn.LayerNorm(d_model)
                self.cross_out = nn.Linear(d_model, proj_dim)
            elif self.fusion_mode == "add_norm":
                self.add_norm = nn.LayerNorm(d_model)
                self.add_out = MLP(d_model, fusion_hidden, proj_dim, dropout)
            else:  # weighted_attn_pool
                self.pool_score = nn.Linear(d_model, 1)
                self.pool_norm = nn.LayerNorm(d_model)
                self.pool_out = nn.Linear(d_model, proj_dim)
        elif self.fusion_mode == "cxr_residual_ecg":
            assert spec.use_cxr_t1 and spec.use_ecg and spec.ecg_mode == "sequence", (
                "cxr_residual_ecg expects CXR_t1 + ECG sequence inputs")
            assert self.encoder is not None and self.ecg_in_proj is not None
            self.cxr_token_proj = nn.Identity() if proj_dim == d_model else nn.Linear(proj_dim, d_model)
            # ``g`` intentionally matches the CXR-only model so it can be warm-started.
            self.g = MLP(proj_dim, fusion_hidden, proj_dim, dropout)
            self.residual_score = nn.Linear(d_model, 1)
            self.residual_delta = MLP(proj_dim + d_model, fusion_hidden, proj_dim, dropout)
            self.residual_scale = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown fusion_mode={self.fusion_mode!r}")

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

    def _encode_sequence_tokens(self, batch, add_pool_token: bool = False):
        """Return Transformer ECG tokens, their mask, and whether token 0 is a pool token."""
        feats = batch["ecg_feats"]
        mask = batch["ecg_mask"].bool()
        if self.spec.ecg_perturb == "zero":
            h = feats.new_zeros(feats.size(0), feats.size(1), self.d_model)
            return h, mask, False

        B = feats.size(0)
        h = self.ecg_in_proj(feats)
        if self.seq_time_emb is not None:
            h = h + self.seq_time_emb(batch["ecg_t2t"])
        has_pool_token = False
        if add_pool_token and self.future_query is not None:
            q_tok = self.future_query.expand(B, 1, -1)
            if self.future_time_emb is not None:
                q_tok = q_tok + self.future_time_emb(batch["delta_t"]).unsqueeze(1)
            h = torch.cat([q_tok, h], dim=1)
            pad = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)
            has_pool_token = True
        elif add_pool_token and self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)
            pad = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)
            has_pool_token = True

        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.enc_norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h, mask, has_pool_token

    def _encode_sequence(self, batch) -> torch.Tensor:
        feats = batch["ecg_feats"]
        if self.spec.ecg_perturb == "zero":
            return feats.new_zeros(feats.size(0), self.ecg_out_dim)

        h, mask, has_pool_token = self._encode_sequence_tokens(batch, add_pool_token=True)
        if has_pool_token:
            return h[:, 0]
        return self._masked_mean(h, mask)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _ecg_vector(self, batch):
        if not self.spec.use_ecg:
            return None
        if self.spec.ecg_mode == "single":
            feats = batch["ecg_feats"][:, 0, :]  # (B, D_ecg), L == 1
            if self.spec.ecg_perturb == "zero":
                return feats.new_zeros(feats.size(0), self.ecg_out_dim)
            z = self.ecg_single_proj(feats)
            if self.single_time_emb is not None:
                z = torch.cat([z, self.single_time_emb(batch["delta_t"])], dim=-1)
            return z
        return self._encode_sequence(batch)

    def _special_fusion_query(self, batch, c1: torch.Tensor) -> torch.Tensor:
        """Fusion modes for the three requested Exp4 follow-up architectures."""
        ecg_tokens, mask, _ = self._encode_sequence_tokens(batch)
        c1_tok = self.cxr_token_proj(c1).unsqueeze(1)

        if self.fusion_mode == "cross_attention_norm":
            attn, _ = self.cross_attn(
                c1_tok, ecg_tokens, ecg_tokens, key_padding_mask=~mask, need_weights=False)
            h = self.cross_attn_norm(c1_tok + attn)
            h = self.cross_ff_norm(h + self.cross_ff(h))
            return F.normalize(self.cross_out(h.squeeze(1)), dim=-1)

        if self.fusion_mode == "add_norm":
            ecg_pool = self._masked_mean(ecg_tokens, mask)
            h = self.add_norm(c1_tok.squeeze(1) + ecg_pool)
            return F.normalize(self.add_out(h), dim=-1)

        if self.fusion_mode == "weighted_attn_pool":
            tokens = torch.cat([c1_tok, ecg_tokens], dim=1)
            c1_mask = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([c1_mask, mask], dim=1)
            scores = self.pool_score(tokens).squeeze(-1).masked_fill(~full_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            h = (tokens * weights.unsqueeze(-1)).sum(dim=1)
            h = self.pool_norm(h)
            return F.normalize(self.pool_out(h), dim=-1)

        if self.fusion_mode == "cxr_residual_ecg":
            base = F.normalize(self.g(c1), dim=-1)
            scores = self.residual_score(ecg_tokens).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            ecg_pool = (ecg_tokens * weights.unsqueeze(-1)).sum(dim=1)
            delta = F.normalize(self.residual_delta(torch.cat([c1, ecg_pool], dim=-1)), dim=-1)
            scale = torch.sigmoid(self.residual_scale)
            return F.normalize(base + scale * delta, dim=-1)

        raise RuntimeError(f"_special_fusion_query called for fusion_mode={self.fusion_mode!r}")

    def encode(self, batch):
        if self.fusion_mode != "mlp_concat":
            c1 = self.cxr_proj(batch["c1"])
            q = self._special_fusion_query(batch, c1)
            c2 = self.cxr_proj(batch["c2"])
            return q, c2, c1

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
