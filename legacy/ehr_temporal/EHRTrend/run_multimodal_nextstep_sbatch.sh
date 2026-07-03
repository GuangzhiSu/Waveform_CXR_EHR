#!/bin/bash
#SBATCH -J mm-nextstep
#SBATCH -t 48:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EHR + CXR + ECG next-step (train_multimodal_nextstep.py). Submit from repo root.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHRTrend"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

_exp_old="${PROJECT_DIR}/experiment1(old)"
export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/BaselineExperiment:${_exp_old}"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

PRETRAINED_MM="${SCRIPT_DIR}/output_mm_forward_mlp/best.pt"
MM_NEXT_ARGS=(
  --output_dir "${SCRIPT_DIR}/output_multimodal_nextstep"
)
if [[ -f "${PRETRAINED_MM}" ]]; then
  MM_NEXT_ARGS+=( --pretrained_mm_forward_mlp_ckpt "${PRETRAINED_MM}" )
fi

python -u "${SCRIPT_DIR}/train_multimodal_nextstep.py" \
  "${MM_NEXT_ARGS[@]}" \
  "$@"

echo "=== Done ==="
