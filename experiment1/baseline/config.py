"""
Default configuration for baseline training.
"""
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDTVT_ROOT = PROJECT_ROOT.parent / "MedTVT-R1"

DATA_CSV = str(PROJECT_ROOT / "cxr_supertable_waveform_matched.csv")
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"

# Encoder checkpoints (MedTVT-R1)
ECG_CKPT = str(MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt")
VIT_PATH = "google/vit-base-patch16-224-in21k"

# Model
HIDDEN_DIM = 512
FREEZE_ENCODERS = True

# Training (encoders frozen; these apply to head only)
BATCH_SIZE = 16
EPOCHS = 50
LR = 5e-4          # Lower LR for head to avoid collapse
WEIGHT_DECAY = 0.001  # Less regularization so head can use encoder features
TRAIN_SPLIT = 0.7   # 70% train
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% test (held out for final eval)
SEED = 42
NUM_WORKERS = 4

# Task: predict oxygenation from ECG + CXR; EHR oxygenation = ground truth
TARGET_COL = "p2f_vent_fio2"  # P/F ratio; or: spo2, partial_pressure_of_oxygen_(pao2), s2f_vent_fio2
