"""Config for EHR ARDS classification baseline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DATA_CSV = str(DATA_DIR / "p2f_ehr_classified.csv")

# EHR feature columns: numeric + encodable. Exclude p2f, s2f (target/leak), IDs, text.
EHR_FEATURE_COLS = [
    "age", "gender", "cci9", "cci10",
    "anion_gap", "base_excess", "bicarb_(hco3)", "blood_urea_nitrogen_(bun)",
    "calcium", "calcium_ionized", "chloride", "creatinine", "gfr", "glucose",
    "magnesium", "osmolarity", "phosphorus", "potassium", "sodium",
    "hematocrit", "hemoglobin", "platelets", "white_blood_cell_count",
    "alanine_aminotransferase_(alt)", "albumin", "aspartate_aminotransferase_(ast)",
    "bilirubin_total", "inr", "lactic_acid", "ph",
    "fio2", "partial_pressure_of_carbon_dioxide_(paco2)", "partial_pressure_of_oxygen_(pao2)",
    "saturation_of_oxygen_(sao2)", "lymphocyte", "neutrophils", "crp_high_sens", "d_dimer", "d_dimer",
    "temperature", "daily_weight_kg", "height_cm",
    "sbp_line", "dbp_line", "map_line", "sbp_cuff", "dbp_cuff", "map_cuff",
    "pulse", "unassisted_resp_rate", "spo2", "end_tidal_co2", "cvp",
    "gcs_total_score", "vent_fio2", "peep",
    "vent_tidal_rate_set", "vent_rate_set",
    "icu", "imc", "ed", "procedure", "on_dialysis", "history_of_dialysis",
    "norepinephrine", "epinephrine", "dobutamine", "dopamine", "phenylephrine", "vasopressin",
    "best_map", "pulse_pressure", "n_to_l",
    "SOFA_coag", "SOFA_renal", "SOFA_hep", "SOFA_neuro", "SOFA_cardio",
    "SOFA_resp", "sofa_total",
    "sirs_total", "meld_score", "aki_score", "infection", "sepsis",
    "elapsed_icu", "elapsed_hosp",
]
# Add gender/race as simple numeric encoding
EHR_CATEGORICAL = {"gender": {"Male": 0, "Female": 1}, "race": "onehot_or_drop"}

NUM_CLASSES = 3
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 0.001
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 0  # EHR is small, no need for multi-worker
