"""Config for the ECG-CXR patient-temporal contrastive learning baseline."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent

# ---------------------------------------------------------------------------
# Data sources (same modality catalogs used by CXR/ECG encoder experiments).
# ---------------------------------------------------------------------------
# CXR catalog: subject_id, dicom_id, hadm_id, supertable_datetime
CXR_CATALOG_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_cxr_catalog.csv")
# ECG catalog: subject_id, ..., wf_Base_Time, wf_File_Path, wf_File_Name
ECG_CATALOG_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_ecg_catalog.csv")
# Full non-EHR-restricted catalogs produced by build_full_catalogs.py.  Use these
# via --cxr_csv/--ecg_csv when rebuilding pairs for the contrastive-only study.
FULL_CXR_CATALOG_CSV = str(PROJECT_ROOT / "data" / "ecg_cxr_full_cxr_catalog.csv")
FULL_ECG_CATALOG_CSV = str(PROJECT_ROOT / "data" / "ecg_cxr_full_ecg_catalog.csv")

# MIMIC-CXR-JPG image root + metadata (dicom_id -> study_id for path building)
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
CXR_METADATA_PATH = (
    "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
)

# ---------------------------------------------------------------------------
# Runtime artifacts.
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = EXP_DIR / "artifacts"
CACHE_ROOT = ARTIFACTS_DIR / "cache"
CACHE_DIR = CACHE_ROOT / "default"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
OUTPUT_DIR = str(OUTPUTS_DIR / "default")
STAGED_OUTPUT_DIR = str(OUTPUTS_DIR / "staged")
LOG_DIR = ARTIFACTS_DIR / "logs"
PYLIBS_DIR = ARTIFACTS_DIR / "pylibs"

# ---------------------------------------------------------------------------
# Frozen encoder checkpoints.
# ---------------------------------------------------------------------------
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
# Bio-ViL-T image encoder (downloaded from HuggingFace microsoft/BiomedVLP-BioViL-T)
BIOVIL_T_CKPT = str(CKPT_DIR / "biovil_t_image_model_proj_size_128.pt")
# ECG-CoCa encoder (from PKUDigitalHealth/ECG-R1 -> ECG-Chat Google Drive)
ECG_COCA_CKPT = str(CKPT_DIR / "cpt_wfep_epoch_20.pt")
ECG_COCA_CONFIG = str(
    EXP_DIR / "external" / "ECG-R1" / "ecg_coca" / "open_clip" / "model_configs" / "coca_ViT-B-32.json"
)

# Frozen encoder output dims
CXR_FEAT_DIM = 512   # Bio-ViL-T img_embedding (global pooled ResNet50 feature)
ECG_FEAT_DIM = 512   # ECG-CoCa pooled latent
ECG_SIG_LEN = 5000   # ECG-CoCa expects 12 x 5000 (500 Hz x 10 s)

# ---------------------------------------------------------------------------
# Embedding cache paths.
# ---------------------------------------------------------------------------
# Exp 4 (+ controls): CXR_t1 + ECG sequence -> CXR_t2 (has cxr_t1).
PAIRS_JSON = str(CACHE_DIR / "patient_temporal_pairs.json")
# Exp 3: ECG sequence -> CXR_t2 (no cxr_t1; larger, less constrained set).
SEQ_TARGET_PAIRS_JSON = str(CACHE_DIR / "seq_target_pairs.json")
# Exp 1 & 2: single ECG -> future CXR.
SINGLE_ECG_PAIRS_JSON = str(CACHE_DIR / "single_ecg_pairs.json")
CXR_EMB_NPY = str(CACHE_DIR / "cxr_emb.npy")
CXR_IDS_JSON = str(CACHE_DIR / "cxr_ids.json")
ECG_EMB_NPY = str(CACHE_DIR / "ecg_emb.npy")
ECG_IDS_JSON = str(CACHE_DIR / "ecg_ids.json")

# ---------------------------------------------------------------------------
# Sequence pair-building parameters (Experiments 3 & 4).
# ---------------------------------------------------------------------------
# Exp 3 (no CXR_t1): ECGs in [t2 - SEQ_LOOKBACK_HOURS, t2 - SEQ_MIN_HORIZON_HOURS]
# -> CXR_t2.
SEQ_MIN_HORIZON_HOURS = 12.0
SEQ_LOOKBACK_HOURS = 24.0
# Exp 4 (with CXR_t1): CXR_t1 in [t2 - 24h, t2], ECGs in [max(t2 - 12h, t1), t2].
MIN_INTERVAL_HOURS = 0.0
MAX_INTERVAL_HOURS = 24.0
MAX_SKIP = 60                     # pair a target t2 with up to this many earlier CXRs as t1
MIN_ECGS_PER_INTERVAL = 1         # require >= this many ECGs in the window
MAX_ECGS_PER_INTERVAL = 32        # cap ECGs per sample (keep most recent before t2)
ECG_LOOKBACK_HOURS = 12.0
# Legacy t1-anchored ECG window kept only for older callers that still pass
# explicit --ecg_before_hours / --ecg_after_hours.
ECG_WINDOW_BEFORE_HOURS = 12.0
ECG_WINDOW_AFTER_HOURS = 3.0

# ---------------------------------------------------------------------------
# Single-ECG -> future-CXR pair-building parameters (Experiments 1 & 2).
#   ECG at time t  ->  CXR at time t + [MIN, MAX] hours.
# ---------------------------------------------------------------------------
SINGLE_MIN_HORIZON_HOURS = 9.0
SINGLE_MAX_HORIZON_HOURS = 15.0
# Caps set high for the maximal sample set (0/large = effectively uncapped).
SINGLE_MAX_CXR_PER_ECG = 100000
SINGLE_MAX_PAIRS_PER_PATIENT = 100000

# ---------------------------------------------------------------------------
# Model hyper-parameters.
# ---------------------------------------------------------------------------
PROJ_DIM = 256                    # contrastive embedding dim (c_t1, c_t2, q)
CXR_PROJ_HIDDEN = 512
D_MODEL = 256                     # ECG temporal transformer width (== PROJ_DIM for clean concat)
ECG_TX_LAYERS = 3
ECG_TX_HEADS = 4
ECG_TX_MLP_RATIO = 4.0
FUSION_HIDDEN = 512               # predictor / fusion MLP hidden width
DROPOUT = 0.1
ECG_POOL = "mean"                 # 'mean' | 'cls' | 'query'
TIME_EMB_DIM = 64                 # delta-time embedding dim for the single-ECG predictor (Exp 2)

# Contrastive temperature
TEMPERATURE = 0.07
LEARNABLE_TEMPERATURE = False

# ---------------------------------------------------------------------------
# Loss weights / ablation.
# ---------------------------------------------------------------------------
LAMBDA_TEMPORAL = 0.2
# loss_mode in {"combined", "cross", "temporal"} sets the (w_cross, w_temporal) pair
LOSS_MODE = "combined"

# ---------------------------------------------------------------------------
# Sampler / training.
# ---------------------------------------------------------------------------
N_PATIENTS = 16                   # patients per batch (N)
K_INTERVALS = 2                   # intervals per patient per batch (K) -> batch = N*K
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0
SEED = 42
NUM_WORKERS = 4
STEPS_PER_EPOCH = None            # None -> derive from dataset size

# Patient-level split fractions
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4


def loss_weights(loss_mode: str, lambda_temporal: float) -> tuple[float, float]:
    """Return (w_cross, w_temporal) for the requested ablation mode."""
    m = loss_mode.lower()
    if m == "cross":
        return 1.0, 0.0
    if m == "temporal":
        return 0.0, 1.0
    if m == "combined":
        return 1.0, float(lambda_temporal)
    raise ValueError(f"Unknown loss_mode={loss_mode!r} (expected cross|temporal|combined)")
