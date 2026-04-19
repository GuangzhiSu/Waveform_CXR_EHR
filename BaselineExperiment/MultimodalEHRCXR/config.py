"""Multimodal EHR + CXR: CLIP alignment + ARDS classification."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ANCHOR_CSV = str(DATA_DIR / "p2f_ehr_classified.csv")
HISTORY_CSV = str(DATA_DIR / "p2f_vent_fio2_enriched.csv")
SCHEMA_CSV = str(DATA_DIR / "supertable_columns_completed.csv")

# Rows with CXR times + paths (same windowing as EHR)
CXR_POOL_CSV = str(DATA_DIR / "p2f_cxr_classified.csv")
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"

VIT_PATH = "google/vit-base-patch16-224-in21k"

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
EHR_EMBED_DIM = 256
CXR_HIDDEN_DIM = 512
CONTRAST_DIM = 256
FUSION_HIDDEN = 512
POOLING_STATS = ("mean",)

LAMBDA_CONTRAST = 0.5
LAMBDA_TASK = 1.0
LOGIT_SCALE_INIT = 2.6592

FREEZE_CXR_ENCODER = True

BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 2
