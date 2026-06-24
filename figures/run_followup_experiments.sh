#!/bin/bash
# Submit follow-up training jobs (EHR / CXR / ECG fixes).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

FIX_BUNDLE=(
  --no-use_class_weights
  --grad_clip 1.0
  --label_smoothing 0.05
  --lr 1e-4
  --p2f_loss_weight 5.0
)

echo "=== EHR EHREncoderTransformer Fix-D ==="
sbatch --job-name=ehr-tr-fixD EHREncoderTransformer/run_train.sh \
  --output_dir EHREncoderTransformer/output_fixD \
  "${FIX_BUNDLE[@]}" \
  --epochs 80 \
  --early_stop_patience 20

echo "=== EHR EHREncoderTransformer Fix-E ==="
sbatch --job-name=ehr-tr-fixE EHREncoderTransformer/run_train.sh \
  --output_dir EHREncoderTransformer/output_fixE \
  --no-use_class_weights \
  --grad_clip 1.0 \
  --label_smoothing 0.05 \
  --lr 5e-5 \
  --p2f_loss_weight 5.0 \
  --epochs 80 \
  --early_stop_patience 20

echo "=== EmbedPred Exp-D ==="
sbatch --job-name=ehr-embed-expD EHREncoderTransformerEmbedPred/run_train.sh \
  --output_dir EHREncoderTransformerEmbedPred/output_twophase_expD \
  --no-use_class_weights \
  --grad_clip 1.0 \
  --label_smoothing 0.05 \
  --lr 1e-4 \
  --p2f_loss_weight 5.0 \
  --finetune_epochs 100 \
  --early_stop_patience 20

echo "=== EmbedPred Exp-E (skip pretrain) ==="
sbatch --job-name=ehr-embed-expE EHREncoderTransformerEmbedPred/run_train.sh \
  --output_dir EHREncoderTransformerEmbedPred/output_twophase_expE \
  --skip_pretrain \
  --no-use_class_weights \
  --grad_clip 1.0 \
  --label_smoothing 0.05 \
  --lr 1e-4 \
  --p2f_loss_weight 5.0 \
  --finetune_epochs 100 \
  --early_stop_patience 20

echo "=== EHRWindowTransformer Window-Fix ==="
sbatch --job-name=ehr-window-fix EHRWindowTransformer/run_ehr_window_transformer.sh \
  --output_dir EHRWindowTransformer/output_direct_window_fix \
  --lr 1e-4 \
  --epochs 80 \
  --early_stop_patience 20

echo "=== CXR-Fix-A (5k subset, no class weights) ==="
sbatch --job-name=cxr-fixA CXREncoderTransformer/run_train.sh \
  --output_dir CXREncoderTransformer/output_fixA \
  --no-use_class_weights \
  --label_smoothing 0.05 \
  --max_samples 5000

echo "=== CXR-Fix-B (full data) ==="
sbatch --job-name=cxr-fixB CXREncoderTransformer/run_train.sh \
  --output_dir CXREncoderTransformer/output_fixB \
  --no-use_class_weights \
  --label_smoothing 0.05

echo "=== ECG-Fix-A ==="
sbatch --job-name=ecg-fixA ECGEncoderTransformer/run_train.sh \
  --output_dir ECGEncoderTransformer/output_fixA \
  --no-use_class_weights

echo "=== ECG-Fix-B ==="
sbatch --job-name=ecg-fixB ECGEncoderTransformer/run_train.sh \
  --output_dir ECGEncoderTransformer/output_fixB \
  --no-use_class_weights \
  --p2f_loss_weight 5.0 \
  --early_stop_patience 20

echo "=== All follow-up jobs submitted ==="
