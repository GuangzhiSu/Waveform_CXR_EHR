#!/bin/bash
#SBATCH -J cxr-diag
#SBATCH -t 2:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/CXREncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1
export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRWindowTransformer:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -u "${SCRIPT_DIR}/diagnose_collapse.py" \
  --max_samples 3000 \
  --epochs 5 \
  --output_dir "${SCRIPT_DIR}/output/diagnose"

echo "=== Diagnose done ==="
