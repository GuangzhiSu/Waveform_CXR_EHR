"""Config for EHREncoderTransformerEmbedPred: EHREncoderTransformer + anchor-embed prediction loss."""
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

# Third loss: predict t embedding from [t-24h, t-12h] window vs row_encoder(t) snapshot.
L_EMBED = 0.25

BATCH_SIZE = 64
# Two-phase schedule: embed-only pretrain, then cls-only finetune.
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 50
EPOCHS = FINETUNE_EPOCHS  # backward-compatible alias
PRETRAIN_EARLY_STOP_PATIENCE = 0
LR = 5e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
OUTPUT_DIR = str(EXP_DIR / "output_twophase")

# Input sequence = lookback [t-24h, t-12h] only; anchor_ehr@t used only as embed-loss target.
INCLUDE_ANCHOR_ROW = False
P2F_LOSS_WEIGHT = 10.0
USE_CLASS_WEIGHTS = True
GRAD_CLIP_NORM = 1.0
