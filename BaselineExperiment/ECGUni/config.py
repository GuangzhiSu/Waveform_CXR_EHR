"""Config for ECG temporal ARDS classification baseline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEDTVT_ROOT = PROJECT_ROOT / "MedTVT-R1"

# Use expanded ECG rows as history pool; anchors are deduped by (subject_id, index)
DATA_CSV = str(DATA_DIR / "p2f_ecg_all_classified.csv")
ECG_CKPT = (
    str(MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt")
    if MEDTVT_ROOT.exists() and (MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt").exists()
    else None
)

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
HIDDEN_DIM = 512
FREEZE_ENCODER = True
USE_LORA = False
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4
POOLING_STATS = ("mean",)
