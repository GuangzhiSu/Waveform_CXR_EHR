#!/bin/bash
#SBATCH -J ehr-ards-baseline
#SBATCH -t 12:00:00
#SBATCH -A kamaleswaranlab
# #SBATCH -p gpu-common   # Partition empty in assoc; use default
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EHR baseline: extract EHR+classified data -> train EHR-only ARDS severity
set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/BaselineExperiment/EHRUni"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DATA_DIR="${PROJECT_DIR}/data"
ENRICHED_CSV="${DATA_DIR}/p2f_vent_fio2_enriched.csv"
CLASSIFIED_CSV="${DATA_DIR}/p2f_ehr_classified.csv"
EXTRACT_SCRIPT="${DATA_DIR}/extract_ehr_p2f_classified.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

# Step 1: Extract EHR rows with p2f classification (skip if already done)
if [[ -f "${CLASSIFIED_CSV}" ]]; then
  echo "=== Step 1: SKIP (${CLASSIFIED_CSV} exists) ==="
else
  echo "=== Step 1: Extract EHR rows and add ARDS severity class ==="
  python "${EXTRACT_SCRIPT}" --input "${ENRICHED_CSV}" --output "${CLASSIFIED_CSV}"
  if [[ ! -f "${CLASSIFIED_CSV}" ]]; then
    echo "ERROR: Classified CSV not created"
    exit 1
  fi
fi

# Step 2: Train EHR classification model
echo "=== Step 2: Train EHR ARDS severity classification model ==="
python "${TRAIN_SCRIPT}" \
  --csv_path "${CLASSIFIED_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
