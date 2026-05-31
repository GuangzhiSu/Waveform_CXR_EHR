#!/bin/bash
#SBATCH -J ehr-enc-tr
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EHREncoderTransformer: 3-layer row MLP -> causal transformer -> dual s2f/p2f change heads.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHREncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

echo "Slurm partition=${SLURM_JOB_PARTITION:-unknown}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "WARNING: nvidia-smi not found on compute node"
fi

python -u "${TRAIN_SCRIPT}" \
  --anchor_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --history_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --schema_csv "${PROJECT_DIR}/supertable_columns_completed.csv" \
  --enriched_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
