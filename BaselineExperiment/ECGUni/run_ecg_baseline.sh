#!/bin/bash
#SBATCH -J ecg-ards-baseline
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
# #SBATCH -p gpu-common   # Partition empty in assoc; use default
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# -x excludes nodes; remove if "Requested node configuration is not available"
# #SBATCH -x dcc-core-gpu-ferc-s-p15-20

# ECG baseline: extract all ECGs in window + p2f classification -> train ECG-only ARDS severity
set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/BaselineExperiment/ECGUni"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DATA_DIR="${PROJECT_DIR}/data"
ENRICHED_CSV="${DATA_DIR}/p2f_vent_fio2_enriched.csv"
CLASSIFIED_CSV="${DATA_DIR}/p2f_ecg_all_classified.csv"
EXTRACT_SCRIPT="${DATA_DIR}/extract_ecg_all_p2f_classified.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/MedTVT-R1" 2>/dev/null && pwd || cd "${PROJECT_DIR}/../MedTVT-R1" 2>/dev/null && pwd || true)"
if [[ -n "${MEDTVT_ROOT}" && -f "${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt" ]]; then
  ECG_CKPT="${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt"
else
  ECG_CKPT=""
fi
OUTPUT_DIR="${SCRIPT_DIR}/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

# Step 1: Extract all ECGs in window with p2f classification (skip if already done)
if [[ -f "${CLASSIFIED_CSV}" ]]; then
  echo "=== Step 1: SKIP (${CLASSIFIED_CSV} exists) ==="
else
  echo "=== Step 1: Extract all ECGs in window and add ARDS severity class ==="
  python "${EXTRACT_SCRIPT}" --input "${ENRICHED_CSV}" --output "${CLASSIFIED_CSV}"
  if [[ ! -f "${CLASSIFIED_CSV}" ]]; then
    echo "ERROR: Classified CSV not created"
    exit 1
  fi
fi

# Step 2: Train ECG classification model
echo "=== Step 2: Train ECG ARDS severity classification model (LoRA on encoder+proj by default; use --no_lora to disable) ==="
python "${TRAIN_SCRIPT}" \
  --csv_path "${CLASSIFIED_CSV}" \
  --ecg_ckpt "${ECG_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
