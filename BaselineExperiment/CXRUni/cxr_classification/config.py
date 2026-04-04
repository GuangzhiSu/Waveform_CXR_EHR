"""Config for CXR ARDS classification baseline."""
from pathlib import Path

# cxr_classification/ -> CXRUni/ -> BaselineExperiment/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MEDTVT_ROOT = PROJECT_ROOT / "MedTVT-R1"

DATA_CSV = str(DATA_DIR / "p2f_cxr_classified.csv")
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH = str(MEDTVT_ROOT / "CKPTS" / "vit-base-patch16-224") if MEDTVT_ROOT.exists() else "google/vit-base-patch16-224-in21k"

NUM_CLASSES = 3
HIDDEN_DIM = 512
FREEZE_ENCODER = True
BATCH_SIZE = 16
EPOCHS = 50
# Head + projection (when encoder frozen): aligned with ECG baseline tuning
LR = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
# When --no_freeze: smaller LR for ViT backbone vs head/proj
BACKBONE_LR = 1e-5
BACKBONE_WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4
