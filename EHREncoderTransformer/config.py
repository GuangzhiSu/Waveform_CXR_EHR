"""Config for EHREncoderTransformer: MLP row encoder + causal transformer + dual cls heads."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent

P2F_OR_S2F_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv")
ENRICHED_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")
SCHEMA_CSV = str(PROJECT_ROOT / "supertable_columns_completed.csv")

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
EMBED_DIM = 256
D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
HEAD_DROPOUT = 0.2
ANCHOR_POOL = "last"
MAX_SEQ_LENGTH = 512

BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
OUTPUT_DIR = str(EXP_DIR / "output")

# Input sequence = lookback [t-24h, t-12h] only (no anchor@t row).
INCLUDE_ANCHOR_ROW = False
# Up-weight p2f CE vs s2f (train ~10:1 s2f:p2f anchors per batch).
P2F_LOSS_WEIGHT = 10.0
USE_CLASS_WEIGHTS = True
