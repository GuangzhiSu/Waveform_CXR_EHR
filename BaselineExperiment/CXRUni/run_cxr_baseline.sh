#!/bin/bash
#SBATCH -J cxr-ards-baseline
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

# CXR baseline: extract CXR+classified data -> train CXR-only ARDS severity classification
#
# Usage:
#   sbatch BaselineExperiment/CXRUni/run_cxr_baseline.sh
#   sbatch .../run_cxr_baseline.sh --epochs 50 --output_dir BaselineExperiment/CXRUni/cxr_classification/output
# Override image root on hosts where /hpc/group/... is not mounted:
#   export CXR_ROOT=/your/mimic_cxr_jpg
#   export METADATA_PATH=/your/mimic-cxr-2.0.0-metadata.csv.gz   # optional
set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/BaselineExperiment/CXRUni"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DATA_DIR="${PROJECT_DIR}/data"
ENRICHED_CSV="${DATA_DIR}/p2f_vent_fio2_enriched.csv"
CLASSIFIED_CSV="${DATA_DIR}/p2f_cxr_classified.csv"
EXTRACT_SCRIPT="${DATA_DIR}/extract_cxr_p2f_classified.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/cxr_classification/train.py"
CXR_ROOT="${CXR_ROOT:-/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg}"
METADATA_PATH="${METADATA_PATH:-/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz}"
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/MedTVT-R1" 2>/dev/null && pwd || cd "${PROJECT_DIR}/../MedTVT-R1" 2>/dev/null && pwd || true)"
if [[ -n "${MEDTVT_ROOT}" && -d "${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224" ]]; then
  VIT_PATH="${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224"
else
  VIT_PATH="google/vit-base-patch16-224-in21k"
fi
OUTPUT_DIR="${SCRIPT_DIR}/cxr_classification/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

# Step 1: Extract CXR rows with p2f classification (skip if already done)
if [[ -f "${CLASSIFIED_CSV}" ]]; then
  echo "=== Step 1: SKIP (${CLASSIFIED_CSV} exists) ==="
else
  echo "=== Step 1: Extract CXR rows and add ARDS severity class ==="
  python "${EXTRACT_SCRIPT}" --input "${ENRICHED_CSV}" --output "${CLASSIFIED_CSV}"
  if [[ ! -f "${CLASSIFIED_CSV}" ]]; then
    echo "ERROR: Classified CSV not created"
    exit 1
  fi
fi

# Step 2: Train CXR classification model
echo "=== Step 2: Train CXR ARDS severity classification model ==="
# Add --train_diag for per-epoch val pred vs label counts + first-batch grad norm (collapse debugging).
python "${TRAIN_SCRIPT}" \
  --csv_path "${CLASSIFIED_CSV}" \
  --cxr_root "${CXR_ROOT}" \
  --metadata_path "${METADATA_PATH}" \
  --vit_path "${VIT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_diag \
  "$@"

echo "=== Done ==="
