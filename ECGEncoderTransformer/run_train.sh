#!/bin/bash
#SBATCH -J ecg-enc-tr
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

# ECGEncoderTransformer: frozen baseline2 xresnet1d -> causal transformer -> dual MLP heads (s2f/p2f change).

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/ECGEncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRWindowTransformer:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/experiment1(old)"

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
  --ecg_labeled_csv "${PROJECT_DIR}/data/p2f_or_s2f_ecg_catalog_labeled.csv" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
