"""Staged experiment registry: Exp1 -> Exp4 + shortcut controls.

Each :class:`ExperimentSpec` fully describes one run: which pairs file to use
(single-ECG vs sequence), the model configuration (whether to use CXR_t1, a
single ECG vs an ECG sequence, a predictor ``g``, a time embedding, a temporal
Transformer, a learnable future query), the ECG perturbation (none / zero /
shuffle) used by the Experiment-4 shortcut controls, and the loss weights.

All experiments share the same two contrastive losses:

    loss = cross_patient_loss + lambda_temporal * temporal_loss

The framework deliberately stages from ECG-only (no CXR_t1) to multimodal
fusion so we can tell whether the model truly uses the ECG signal rather than
leaning on a CXR_t1 shortcut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ExperimentSpec:
    name: str                      # unique run name / output subdir
    description: str               # human readable
    pairs_kind: str                # "single" | "seq_target" (Exp3) | "seq_t1" (Exp4)
    target_window: str             # human-readable horizon, e.g. "9-15h"

    # ---- model configuration ------------------------------------------------
    use_cxr_t1: bool = False       # include c_t1 in the query (multimodal fusion)
    use_ecg: bool = True           # use the ECG branch at all (False = CXR-only)
    ecg_mode: str = "single"       # "single" | "sequence"
    use_predictor_g: bool = False  # MLP predictor g on top of the fused vector
    use_time_embedding: bool = False
    use_transformer: bool = False  # temporal Transformer over the ECG sequence
    use_future_query: bool = False # learnable future-time query pooling
    ecg_pool: str = "mean"         # "mean" | "cls" | "query"
    ecg_proj_kind: str = "linear"  # single-ECG projector: "linear" | "mlp"
    ecg_perturb: str = "none"      # "none" | "zero" | "shuffle"  (shortcut controls)
    fusion_mode: str = "mlp_concat"  # "mlp_concat" | "cross_attention_norm" | "add_norm" | "weighted_attn_pool"

    # ---- loss ---------------------------------------------------------------
    loss_mode: str = "combined"    # "cross" | "temporal" | "combined"
    lambda_temporal: float = 0.2
    temporal_min_horizon_hours: Optional[float] = None
    temporal_max_horizon_hours: Optional[float] = None

    # -------------------------------------------------------------------------
    def data_kind(self) -> str:
        """Dataset behaviour: single ECG vs ECG sequence."""
        return "single" if self.pairs_kind == "single" else "sequence"

    def table_row_meta(self) -> dict:
        """Descriptive columns for the unified results table."""
        return {
            "experiment_name": self.name,
            "input_type": self._input_type(),
            "target_window": self.target_window,
            "uses_cxr_t1": self.use_cxr_t1,
            "uses_single_ecg": self.use_ecg and self.ecg_mode == "single",
            "uses_ecg_sequence": self.use_ecg and self.ecg_mode == "sequence",
            "uses_predictor_g": self.use_predictor_g,
            "uses_transformer": self.use_transformer,
            "uses_future_query": self.use_future_query,
            "ecg_perturb": self.ecg_perturb,
            "fusion_mode": self.fusion_mode,
            "loss_type": self.loss_mode,
            "lambda_temporal": self.lambda_temporal if self.loss_mode != "cross" else 0.0,
        }

    def _input_type(self) -> str:
        if not self.use_ecg:
            return "cxr_t1_only"
        ecg = "single_ecg" if self.ecg_mode == "single" else "ecg_sequence"
        if self.ecg_perturb == "shuffle":
            ecg += "_shuffled"
        elif self.ecg_perturb == "zero":
            ecg += "_zeroed"
        if self.use_cxr_t1:
            return f"cxr_t1+{ecg}"
        return ecg

    def asdict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Registry: recommended run order (Step 1 .. Step 6).
# --------------------------------------------------------------------------- #
def default_registry() -> "list[ExperimentSpec]":
    specs: list[ExperimentSpec] = []

    # Step 1 -- Experiment 1A: single ECG -> future CXR, cross-patient only.
    specs.append(ExperimentSpec(
        name="exp1a_single_ecg_cross",
        description="Single ECG -> future CXR alignment (9-15h), cross-patient loss only.",
        pairs_kind="single", target_window="9-15h",
        use_cxr_t1=False, use_ecg=True, ecg_mode="single",
        use_predictor_g=False, use_time_embedding=False,
        use_transformer=False, use_future_query=False, ecg_proj_kind="mlp",
        loss_mode="cross", lambda_temporal=0.0,
        temporal_min_horizon_hours=9.0, temporal_max_horizon_hours=15.0,
    ))

    # Step 2 -- Experiment 1B: single ECG -> future CXR, cross + 0.2 temporal.
    specs.append(ExperimentSpec(
        name="exp1b_single_ecg_combined",
        description="Single ECG -> future CXR alignment (9-15h), cross + 0.2*temporal.",
        pairs_kind="single", target_window="9-15h",
        use_cxr_t1=False, use_ecg=True, ecg_mode="single",
        use_predictor_g=False, use_time_embedding=False,
        use_transformer=False, use_future_query=False, ecg_proj_kind="mlp",
        loss_mode="combined", lambda_temporal=0.2,
        temporal_min_horizon_hours=9.0, temporal_max_horizon_hours=15.0,
    ))

    # Step 3 -- Experiment 2: single ECG + predictor g(., delta_t) -> future CXR.
    specs.append(ExperimentSpec(
        name="exp2_single_ecg_predictor",
        description="Single ECG + delta-time predictor g -> future CXR (9-15h), cross + 0.2*temporal.",
        pairs_kind="single", target_window="9-15h",
        use_cxr_t1=False, use_ecg=True, ecg_mode="single",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=False, use_future_query=False, ecg_proj_kind="linear",
        loss_mode="combined", lambda_temporal=0.2,
        temporal_min_horizon_hours=9.0, temporal_max_horizon_hours=15.0,
    ))

    # Step 4 -- Experiment 3A: sequential ECG -> future CXR, mean pooling.
    specs.append(ExperimentSpec(
        name="exp3a_seq_ecg_meanpool",
        description="ECG sequence (12-24h before CXR_t2, no CXR_t1) -> Transformer -> mean pool -> g -> CXR_t2.",
        pairs_kind="seq_target", target_window="12-24h before CXR_t2",
        use_cxr_t1=False, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        loss_mode="combined", lambda_temporal=0.2,
        temporal_min_horizon_hours=12.0, temporal_max_horizon_hours=24.0,
    ))

    # Step 5 -- Experiment 3B: sequential ECG + learnable future query.
    specs.append(ExperimentSpec(
        name="exp3b_seq_ecg_future_query",
        description="ECG sequence (12-24h before CXR_t2, no CXR_t1) + learnable future-time query -> CXR_t2.",
        pairs_kind="seq_target", target_window="12-24h before CXR_t2",
        use_cxr_t1=False, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=True, ecg_pool="query",
        loss_mode="combined", lambda_temporal=0.2,
        temporal_min_horizon_hours=12.0, temporal_max_horizon_hours=24.0,
    ))

    # Step 6 -- Experiment 4: CXR_t1 + ECG sequence fusion, with shortcut controls.
    # C: full fusion
    specs.append(ExperimentSpec(
        name="exp4c_fusion_cxr1_ecgseq",
        description="Fusion: CXR_t1 + ECG sequence -> CXR_t2 (the target model).",
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    # A: ECG-only (no CXR_t1) -- same architecture as 3A, kept under exp4 for comparison.
    specs.append(ExperimentSpec(
        name="exp4a_ecg_only",
        description="Shortcut control A: ECG sequence only -> CXR_t2 (no CXR_t1), Exp4 sample set.",
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=False, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    # B: CXR-only (no ECG)
    specs.append(ExperimentSpec(
        name="exp4b_cxr_only",
        description="Shortcut control B: CXR_t1 only -> CXR_t2 (no ECG).",
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=False, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=False,
        use_transformer=False, use_future_query=False,
        loss_mode="combined", lambda_temporal=0.2,
    ))
    # D: fusion with shuffled ECG (ECG from another patient)
    specs.append(ExperimentSpec(
        name="exp4d_fusion_shuffled_ecg",
        description="Shortcut control D: CXR_t1 + shuffled (other-patient) ECG sequence -> CXR_t2.",
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        ecg_perturb="shuffle",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    # E: fusion with zeroed ECG
    specs.append(ExperimentSpec(
        name="exp4e_fusion_zeroed_ecg",
        description="Shortcut control E: CXR_t1 + zeroed ECG sequence -> CXR_t2.",
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        ecg_perturb="zero",
        loss_mode="combined", lambda_temporal=0.2,
    ))

    # Follow-up fusion schemes requested after Exp4. All keep the same Exp4
    # sample set and the same contrastive objective against CXR_t2.
    specs.append(ExperimentSpec(
        name="exp5a_proj_tx_crossattn_norm",
        description=("Scheme 1: project CXR_t1 and ECG tokens to one dimension, "
                     "ECG Transformer, CXR query cross-attends to ECG, normalized projection."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="cross_attention_norm",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp5b_proj_add_norm",
        description=("Scheme 2: project CXR_t1 and pooled ECG sequence to one dimension, "
                     "add them, then normalize projection."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="add_norm",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp5c_weighted_attn_pool",
        description=("Scheme 3: single-linear weighted attentive pooling over CXR_t1 "
                     "and ECG Transformer tokens, then normalized projection."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="weighted_attn_pool",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp5c_weighted_attn_pool_shuffled",
        description=("Scheme 3 control: weighted attentive pooling over CXR_t1 "
                     "and shuffled other-patient ECG tokens."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="weighted_attn_pool", ecg_perturb="shuffle",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp5c_weighted_attn_pool_zeroed",
        description=("Scheme 3 control: weighted attentive pooling over CXR_t1 "
                     "and zeroed ECG tokens."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="weighted_attn_pool", ecg_perturb="zero",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp6a_cxr_residual_ecg",
        description=("CXR_t1 base predictor plus a gated ECG residual. Intended to "
                     "warm-start from CXR-only and test whether ECG improves the CXR_t1 shortcut."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="cxr_residual_ecg",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp6b_ecg_only_residual",
        description=("Frozen CXR_t1 base predictor plus a gated residual whose delta "
                     "is computed from ECG tokens only."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="cxr_ecg_only_residual",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp6b_ecg_only_residual_shuffled",
        description=("Control for exp6b: frozen CXR_t1 base plus ECG-only residual "
                     "using shuffled other-patient ECG tokens."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="cxr_ecg_only_residual", ecg_perturb="shuffle",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    specs.append(ExperimentSpec(
        name="exp6b_ecg_only_residual_zeroed",
        description=("Control for exp6b: frozen CXR_t1 base plus ECG-only residual "
                     "using zeroed ECG tokens."),
        pairs_kind="seq_t1", target_window="0-24h",
        use_cxr_t1=True, use_ecg=True, ecg_mode="sequence",
        use_predictor_g=True, use_time_embedding=True,
        use_transformer=True, use_future_query=False, ecg_pool="mean",
        fusion_mode="cxr_ecg_only_residual", ecg_perturb="zero",
        loss_mode="combined", lambda_temporal=0.2,
    ))
    return specs


REGISTRY = {s.name: s for s in default_registry()}

# Ordered groups for convenient CLI selection.
STEP_GROUPS = {
    "step1": ["exp1a_single_ecg_cross", "exp1b_single_ecg_combined"],
    "step2": ["exp2_single_ecg_predictor"],
    "step3": ["exp3a_seq_ecg_meanpool", "exp3b_seq_ecg_future_query"],
    "step4": ["exp4c_fusion_cxr1_ecgseq", "exp4a_ecg_only", "exp4b_cxr_only",
              "exp4d_fusion_shuffled_ecg", "exp4e_fusion_zeroed_ecg"],
    "fusion_schemes": ["exp5a_proj_tx_crossattn_norm", "exp5b_proj_add_norm",
                       "exp5c_weighted_attn_pool"],
    "weighted_controls": ["exp5c_weighted_attn_pool",
                          "exp5c_weighted_attn_pool_shuffled",
                          "exp5c_weighted_attn_pool_zeroed"],
    "ecg_residual_controls": ["exp6b_ecg_only_residual",
                              "exp6b_ecg_only_residual_shuffled",
                              "exp6b_ecg_only_residual_zeroed"],
    "improve": ["exp5c_weighted_attn_pool", "exp6a_cxr_residual_ecg"],
}
ALL_IN_ORDER = (STEP_GROUPS["step1"] + STEP_GROUPS["step2"]
                + STEP_GROUPS["step3"] + STEP_GROUPS["step4"])
