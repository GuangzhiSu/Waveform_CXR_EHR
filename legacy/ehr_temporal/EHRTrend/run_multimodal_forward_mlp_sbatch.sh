#!/bin/bash
#SBATCH -J mm-fwd-mlp
# Prior 48h runs hit Slurm TIME LIMIT before finishing epochs (e.g. ~22/50); extend for full train + eval.
# If submit fails, lower to your partition max (e.g. 72:00:00) or add checkpoint resume in the trainer.
#SBATCH -t 120:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EHR+CXR+ECG forward-window change MLP (train_multimodal_forward_mlp.py). Submit from repo root.

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

# Early stopping: val loss = val_ce_s2f + val_ce_p2f (same as train_multimodal_forward_mlp.py / config FORWARD_EARLY_STOP_*).
# Override without duplicating CLI flags, e.g. MM_FWD_EARLY_STOP_PATIENCE=15 sbatch EHRTrend/run_multimodal_forward_mlp_sbatch.sh
: "${MM_FWD_EARLY_STOP_PATIENCE:=10}"
: "${MM_FWD_EARLY_STOP_MIN_DELTA:=0}"

python -u "${SCRIPT_DIR}/train_multimodal_forward_mlp.py" \
  --output_dir "${SCRIPT_DIR}/output_mm_forward_mlp" \
  --early_stop_patience "${MM_FWD_EARLY_STOP_PATIENCE}" \
  --early_stop_min_delta "${MM_FWD_EARLY_STOP_MIN_DELTA}" \
  "$@"

echo "=== Done ==="
