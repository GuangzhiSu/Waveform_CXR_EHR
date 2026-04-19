"""Config: multimodal ECG + CXR ARDS severity classification (concat features + MLP head)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEDTVT_ROOT = PROJECT_ROOT / "MedTVT-R1"

MULTIMODAL_CSV = str(DATA_DIR / "p2f_ecg_cxr_multimodal.csv")
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH = str(MEDTVT_ROOT / "CKPTS" / "vit-base-patch16-224") if MEDTVT_ROOT.exists() else "google/vit-base-patch16-224-in21k"
ECG_CKPT = (
    str(MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt")
    if MEDTVT_ROOT.exists()
    and (MEDTVT_ROOT / "CKPTS" / "best_valid_all_increase_with_augment_epoch_3.pt").exists()
    else None
)

NUM_CLASSES = 3
HIDDEN_DIM = 512
FREEZE_ENCODER = True
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
BACKBONE_LR = 1e-5
BACKBONE_WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4
