#!/bin/bash
#SBATCH -J cxr-window-transformer
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

# Train CXRWindowTransformer on CXR images in [anchor_t - 24h, anchor_t - 12h] only.
# Usage:
#   ./EHRWindowTransformer/run_cxr_window_transformer.sh
#   sbatch EHRWindowTransformer/run_cxr_window_transformer.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${SCRIPT_DIR}/train_cxr.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/EHRWindowTransformer/train_cxr.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/EHRWindowTransformer"
  elif [[ -f "${SLURM_SUBMIT_DIR}/train_cxr.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/EHRTrend:${SCRIPT_DIR}"

python -u "${SCRIPT_DIR}/train_cxr.py" \
  --anchor_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --history_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv" \
  --label_lookup_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --enriched_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv" \
  --output_dir "${SCRIPT_DIR}/output_cxr_window" \
  "$@"

echo "=== Done ==="
