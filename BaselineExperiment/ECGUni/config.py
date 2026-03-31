"""Config for ECG ARDS classification baseline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEDTVT_ROOT = PROJECT_ROOT / "MedTVT-R1"

DATA_CSV = str(DATA_DIR / "p2f_ecg_all_classified.csv")
ECG_CKPT = str(MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt") if MEDTVT_ROOT.exists() and (MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt").exists() else None

NUM_CLASSES = 3
HIDDEN_DIM = 512
FREEZE_ENCODER = True
# LoRA on xresnet1d + proj (train LoRA + classification head only)
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16.0
BATCH_SIZE = 16
EPOCHS = 50
# Head often tolerates a slightly higher LR than LoRA adapters on a frozen backbone
LR = 3e-4
LORA_LR = 1e-4
WEIGHT_DECAY = 0.01
LORA_WEIGHT_DECAY = 0.0
# Reduces overconfident collapse toward one class when combined with class weights
LABEL_SMOOTHING = 0.05
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4
