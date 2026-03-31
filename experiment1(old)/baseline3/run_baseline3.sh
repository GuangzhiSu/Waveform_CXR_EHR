#!/bin/bash
#SBATCH -J baseline3-cxr
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -x dcc-core-gpu-ferc-s-p15-20

# Baseline3: CXR-only → predict oxygenation
set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/baseline3"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/../MedTVT-R1" && pwd)"

CSV_PATH="${PROJECT_DIR}/cxr_supertable_waveform_matched.csv"
CXR_ROOT="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
VIT_PATH="${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224"
OUTPUT_DIR="${SCRIPT_DIR}/output"

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

cd "${SCRIPT_DIR}"
python train.py --csv_path "${CSV_PATH}" --cxr_root "${CXR_ROOT}" --metadata_path "${METADATA_PATH}" --vit_path "${VIT_PATH}" --output_dir "${OUTPUT_DIR}" "$@"
