#!/bin/bash
# Submit Fix-A/B/C ablation jobs for EHREncoderTransformer training fixes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RUN="${SCRIPT_DIR}/run_train.sh"

echo "Submitting Fix-A (no class weights only)..."
JOB_A=$(sbatch --parsable --job-name=ehr-tr-fixA "${RUN}" \
  --output_dir "${SCRIPT_DIR}/output_fixA" \
  --no-use_class_weights)
echo "  Fix-A job ${JOB_A}"

echo "Submitting Fix-B (+ grad_clip + label_smoothing)..."
JOB_B=$(sbatch --parsable --job-name=ehr-tr-fixB "${RUN}" \
  --output_dir "${SCRIPT_DIR}/output_fixB" \
  --no-use_class_weights \
  --grad_clip 1.0 \
  --label_smoothing 0.05)
echo "  Fix-B job ${JOB_B}"

echo "Submitting Fix-C (+ lr 1e-4 + p2f_weight 5)..."
JOB_C=$(sbatch --parsable --job-name=ehr-tr-fixC "${RUN}" \
  --output_dir "${SCRIPT_DIR}/output_fixC" \
  --no-use_class_weights \
  --grad_clip 1.0 \
  --label_smoothing 0.05 \
  --lr 1e-4 \
  --p2f_loss_weight 5.0)
echo "  Fix-C job ${JOB_C}"

echo "All ablation jobs submitted."
