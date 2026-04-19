"""Multimodal EHR + ECG: contrastive alignment + ARDS classification."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEDTVT_ROOT = PROJECT_ROOT / "MedTVT-R1"

# EHR (same as EHRUni temporal)
ANCHOR_CSV = str(DATA_DIR / "p2f_ehr_classified.csv")
HISTORY_CSV = str(DATA_DIR / "p2f_vent_fio2_enriched.csv")
SCHEMA_CSV = str(DATA_DIR / "supertable_columns_completed.csv")

# ECG temporal pool (must share subject_id + anchor time with EHR anchors where possible)
ECG_POOL_CSV = str(DATA_DIR / "p2f_ecg_all_classified.csv")

ECG_CKPT = (
    str(MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt")
    if MEDTVT_ROOT.exists()
    and (MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt").exists()
    else None
)

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
# EHR encoder (Symile-style trunk)
EHR_EMBED_DIM = 256
# ECG SignalEncoder output dim (MedTVT proj)
ECG_HIDDEN_DIM = 512
# Shared space for EHR–ECG contrastive (CLIP)
CONTRAST_DIM = 256
# Fusion classifier: concat(EHR_pool, ECG_pool) -> 3-layer MLP
FUSION_HIDDEN = 512
POOLING_STATS = ("mean",)  # lighter default; can use ("mean", "max") etc.

# Loss weights
LAMBDA_CONTRAST = 0.5
LAMBDA_TASK = 1.0
# Learnable temperature for CLIP (init ~ 1/0.07)
LOGIT_SCALE_INIT = 2.6592  # ln(1/0.07)

FREEZE_ECG_ENCODER = True

BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 2
