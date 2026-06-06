"""Config for CXREncoderTransformer: frozen ViT + causal transformer + dual MLP heads."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent

# Same ViT resolution as BaselineExperiment/CXRUni/cxr_classification/config.py
_MEDTVT_CANDIDATES = (
    PROJECT_ROOT / "MedTVT-R1",
    PROJECT_ROOT.parent / "MedTVT-R1",
)
MEDTVT_ROOT = next((p for p in _MEDTVT_CANDIDATES if p.is_dir()), None)
_VIT_LOCAL = MEDTVT_ROOT / "CKPTS" / "vit-base-patch16-224" if MEDTVT_ROOT else None

# Anchors + s2f/p2f change labels
P2F_OR_S2F_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv")
# In-admission CXR catalog (subject_id, dicom_id, hadm_id, supertable_datetime) for [t-24h, t-12h] windows
CXR_CATALOG_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_cxr_catalog.csv")
CXR_CATALOG_LABELED_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_cxr_catalog_labeled.csv")
# Optional: map anchor (hadm_id, index) -> subject_id when enriched join is available
ENRICHED_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")

CXR_ROOT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH = (
    str(_VIT_LOCAL)
    if _VIT_LOCAL is not None and _VIT_LOCAL.is_dir()
    else "google/vit-base-patch16-224-in21k"
)

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24
# Train split uses RandomCrop; val/test loaders override to "val"/"test" (CenterCrop).
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
LR = 1e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
OUTPUT_DIR = str(EXP_DIR / "output")

# Learnable anchor slot can decouple from CXR pixels; verify_collapse_fix compares True/False.
# Default False: pool last valid CXR token (better image sensitivity in diagnostics).
INCLUDE_ANCHOR_SLOT = False
# Balanced with s2f; large p2f weights destabilize sparse p2f mini-batches.
P2F_LOSS_WEIGHT = 1.0
USE_CLASS_WEIGHTS = True
MAX_GRAD_NORM = 1.0
GRAD_CLIP = MAX_GRAD_NORM  # alias for --grad_clip
