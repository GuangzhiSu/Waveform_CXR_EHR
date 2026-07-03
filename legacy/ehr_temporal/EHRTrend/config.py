"""Config for EHR trend classification (decrease/remain/increase)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent

SOURCE_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")
SCHEMA_CSV = str(PROJECT_ROOT / "supertable_columns_completed.csv")
ANCHOR_CSV = str(EXP_DIR / "data" / "ehr_trend_anchors.csv")

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

# trend labels: 0=decrease, 1=remain, 2=increase
NUM_CLASSES = 3
EMBED_DIM = 256
POOLING_STATS = ("mean", "median", "max", "min", "std")
HEAD_HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0

# --- EHR next-step + dual-head training (see train_nextstep.py) ---
P2F_OR_S2F_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv")
NEXTSTEP_ENRICHED_CSV = str(PROJECT_ROOT / "data" / "p2f_vent_fio2_enriched.csv")
NEXTSTEP_D_MODEL = 256
NEXTSTEP_LAMBDA_NEXT = 1.0
NEXTSTEP_LAMBDA_ANCHOR = 1.0
NEXTSTEP_LAMBDA_DISC = 0.5
NEXTSTEP_NUM_TRANSFORMER_LAYERS = 4
NEXTSTEP_NUM_HEADS = 4
NEXTSTEP_DROPOUT = 0.1
NEXTSTEP_ANCHOR_POOL = "last"
# Early stopping on val loss (train_nextstep.py, train_multimodal_nextstep.py)
NEXTSTEP_EARLY_STOP_PATIENCE = 10
NEXTSTEP_EARLY_STOP_MIN_DELTA = 1e-4

# --- Forward row MLP: predict s2f/p2f change in [t+12h, t+24h] (train_forward_mlp.py, train_multimodal_forward_mlp.py) ---
FORWARD_MIN_HOURS = 12
FORWARD_MAX_HOURS = 24
FORWARD_MLP_OUTPUT_DIR = str(EXP_DIR / "output_forward_mlp")
FORWARD_EARLY_STOP_PATIENCE = 10
FORWARD_EARLY_STOP_MIN_DELTA = 0.0
# Example path after train_forward_mlp.py: EHRTrend/output_forward_mlp/best.pt (pass via CLI)

# --- Multimodal next-step (train_multimodal_nextstep.py): EHR+CXR+ECG ---
MULTIMODAL_HISTORY_CSV = NEXTSTEP_ENRICHED_CSV
CXR_ROOT_DEFAULT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH_DEFAULT = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH_DEFAULT = "google/vit-base-patch16-224-in21k"
ECG_CKPT_DEFAULT = ""
CXR_DIM_DEFAULT = 512
ECG_DIM_DEFAULT = 512
FUSE_DIM_DEFAULT = 256
ECG_TARGET_LEN_DEFAULT = 5000
# After train_multimodal_forward_mlp.py: EHRTrend/output_mm_forward_mlp/best.pt
MM_FORWARD_MLP_OUTPUT_DIR = str(EXP_DIR / "output_mm_forward_mlp")
