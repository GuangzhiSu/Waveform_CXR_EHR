"""Config for baseline3: CXR-only."""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
from medtvt_paths import resolve_medtvt_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDTVT_ROOT = Path(resolve_medtvt_root())

DATA_CSV = str(PROJECT_ROOT / "cxr_supertable_waveform_matched.csv")
CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH = str(MEDTVT_ROOT / "CKPTS" / "vit-base-patch16-224")

HIDDEN_DIM = 512
FREEZE_ENCODERS = True
BATCH_SIZE = 16
EPOCHS = 50
LR = 5e-4          # Lower LR for head to avoid collapse
WEIGHT_DECAY = 0.001  # Less regularization so head can use encoder features
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4
TARGET_COL = "p2f_vent_fio2"
