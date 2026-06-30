"""Config for ECGEncoderTransformer: frozen baseline2 SignalEncoder + causal transformer + dual MLP heads."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent
_EXP_OLD = PROJECT_ROOT / "experiment1(old)"
if _EXP_OLD.is_dir() and str(_EXP_OLD) not in sys.path:
    sys.path.insert(0, str(_EXP_OLD))

from medtvt_paths import resolve_medtvt_root  # noqa: E402

P2F_OR_S2F_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_vent_fio2_valid_rows.csv")
ECG_CATALOG_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_ecg_catalog.csv")
ECG_CATALOG_LABELED_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_ecg_catalog_labeled.csv")
ANCHOR_MODALITY_CSV = str(PROJECT_ROOT / "data" / "p2f_or_s2f_anchor_modality_window.csv")

MEDTVT_ROOT = Path(resolve_medtvt_root())
_CKPTS_DIR = MEDTVT_ROOT / "CKPTS"


def resolve_ecg_ckpt_path() -> str | None:
    """MedTVT xresnet .pt or Symile PyTorch-Lightning ``*.ckpt`` under CKPTS/."""
    legacy = _CKPTS_DIR / "best_valid_all_increase_with_augment_epoch_3.pt"
    if legacy.is_file():
        return str(legacy)
    if _CKPTS_DIR.is_dir():
        pl = sorted(_CKPTS_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pl:
            return str(pl[0])
    return None


ECG_CKPT = resolve_ecg_ckpt_path()

LOOKBACK_MIN_HOURS = 12
LOOKBACK_MAX_HOURS = 24

NUM_CLASSES = 3
# Symile PL ckpt uses 1024-d; xresnet SignalEncoder uses 512 (set in train from ckpt kind)
ECG_DIM = 1024 if ECG_CKPT and ECG_CKPT.endswith(".ckpt") else 512
ECG_TARGET_LEN = 1000
INPUT_CHANNELS = 12
D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
HEAD_DROPOUT = 0.2
ANCHOR_POOL = "last"
MAX_SEQ_LENGTH = 512
FREEZE_ECG = True

BATCH_SIZE = 16
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
MAX_GRAD_NORM = 1.0
# Append learnable anchor slot after lookback ECGs so last-pool = query at anchor t.
INCLUDE_ANCHOR_SLOT = True
P2F_LOSS_WEIGHT = 1.0
USE_CLASS_WEIGHTS = True
OUTPUT_DIR = str(EXP_DIR / "output")
