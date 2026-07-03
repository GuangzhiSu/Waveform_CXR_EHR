"""Config for EHR ARDS classification (temporal percentile + pooled embeddings)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Anchor points (rows with p2f_class labels for classification)
DATA_CSV = str(DATA_DIR / "p2f_ehr_classified.csv")
# Historical rows searched in lookback window for EHR sequence construction
HISTORY_CSV = str(DATA_DIR / "p2f_vent_fio2_enriched.csv")
# Schema with feature inclusion + default imputation rules
SCHEMA_CSV = str(DATA_DIR / "supertable_columns_completed.csv")

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
EMBED_DIM = 256
# 5 pooled statistics -> mean/median/max/min/std
POOLING_STATS = ("mean", "median", "max", "min", "std")
HEAD_HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 0.001
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0

# Within lookback window: per-feature temporal nearest imputation on raw values, then percentiles
WINDOW_TEMPORAL_IMPUTE = True
