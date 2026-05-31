"""Config for CXREncoderTransformer: frozen ViT + causal transformer + dual MLP heads."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent

# Anchors + s2f/p2f change labels
P2F_OR_S2F_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv")
# In-admission CXR catalog (subject_id, dicom_id, hadm_id, supertable_datetime) for [t-24h, t-12h] windows
CXR_CATALOG_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_cxr_catalog.csv")
CXR_CATALOG_LABELED_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_cxr_catalog_labeled.csv")
# Optional: map anchor (hadm_id, index) -> subject_id when enriched join is available
ENRICHED_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")

CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH = "google/vit-base-patch16-224-in21k"

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24
CXR_SPLIT = "train"

NUM_CLASSES = 3
CXR_DIM = 512
D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
HEAD_DROPOUT = 0.2
ANCHOR_POOL = "last"
MAX_SEQ_LENGTH = 512
FREEZE_CXR = True

BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
OUTPUT_DIR = str(EXP_DIR / "output")
