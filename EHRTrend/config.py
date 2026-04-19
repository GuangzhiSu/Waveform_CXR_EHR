"""Config for EHR trend classification (decrease/remain/increase)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent

SOURCE_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")
SCHEMA_CSV = str(PROJECT_ROOT / "supertable_columns_completed.csv")
ANCHOR_CSV = str(EXP_DIR / "data" / "ehr_trend_anchors.csv")

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

# trend labels: 0=decrease, 1=remain, 2=increase
NUM_CLASSES = 3
EMBED_DIM = 256
POOLING_STATS = ("mean", "median", "max", "min", "std")
HEAD_HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0
