#!/bin/bash
#SBATCH -J cxr-verify
#SBATCH -t 02:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
cd "${PROJECT_DIR}" || exit 1
mkdir -p "${PROJECT_DIR}/logs"

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRWindowTransformer:${PROJECT_DIR}/BaselineExperiment"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

echo "=== test_pool_anchor_index ==="
python "${SCRIPT_DIR}/test_pool_anchor_index.py"

echo "=== verify_collapse_fix ==="
python -u "${SCRIPT_DIR}/verify_collapse_fix.py" --max_samples 2000 --mini_steps 80 --batch_size 16

echo "=== Done verify ==="
