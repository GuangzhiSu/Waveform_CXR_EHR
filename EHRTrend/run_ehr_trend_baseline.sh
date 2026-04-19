#!/bin/bash
#SBATCH -J ehr-trend-baseline
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EHR trend experiment:
# 1) Build trend anchors (decrease/remain/increase)
# 2) Train temporal EHR model on [t-24h, t-12h] sequence

set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHRTrend"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

BUILD_SCRIPT="${SCRIPT_DIR}/build_trend_dataset.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
SOURCE_CSV="${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv"
SCHEMA_CSV="${PROJECT_DIR}/supertable_columns_completed.csv"
ANCHOR_CSV="${SCRIPT_DIR}/data/ehr_trend_anchors.csv"
OUTPUT_DIR="${SCRIPT_DIR}/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

if [[ -f "${ANCHOR_CSV}" ]]; then
  echo "=== Step 1: SKIP (${ANCHOR_CSV} exists) ==="
else
  echo "=== Step 1: Build EHR trend anchors ==="
  python "${BUILD_SCRIPT}" \
    --source_csv "${SOURCE_CSV}" \
    --output_csv "${ANCHOR_CSV}" \
    --lookback_min_hours 12 \
    --lookback_max_hours 24
fi

echo "=== Step 2: Train EHR trend classification model ==="
python -u "${TRAIN_SCRIPT}" \
  --anchor_csv "${ANCHOR_CSV}" \
  --history_csv "${SOURCE_CSV}" \
  --schema_csv "${SCHEMA_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
