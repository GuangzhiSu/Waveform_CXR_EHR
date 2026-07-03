#!/bin/bash
#SBATCH -J cxr-enc-tr
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

# CXREncoderTransformer: frozen ViT -> causal transformer -> dual MLP heads (s2f/p2f change).

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/CXREncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/MedTVT-R1" 2>/dev/null && pwd || cd "${PROJECT_DIR}/../MedTVT-R1" 2>/dev/null && pwd || true)"
if [[ -n "${MEDTVT_ROOT}" && -d "${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224" ]]; then
  VIT_PATH="${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224"
else
  VIT_PATH="google/vit-base-patch16-224-in21k"
fi
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRWindowTransformer:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment"

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

echo "  ViT encoder path: ${VIT_PATH}"

python -u "${TRAIN_SCRIPT}" \
  --cxr_labeled_csv "${PROJECT_DIR}/data/p2f_or_s2f_cxr_catalog_labeled.csv" \
  --output_dir "${OUTPUT_DIR}" \
  --vit_path "${VIT_PATH}" \
  "$@"

echo "=== Done ==="
