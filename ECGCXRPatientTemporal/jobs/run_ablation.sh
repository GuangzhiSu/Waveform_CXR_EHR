#!/bin/bash
#SBATCH -J ptc-ablation
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o ECGCXRPatientTemporal/artifacts/logs/%x-%j.out
#SBATCH -e ECGCXRPatientTemporal/artifacts/logs/%x-%j.err
# Ablation: only cross_patient_loss, only temporal_loss, cross + 0.2*temporal.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

for mode in cross temporal combined; do
  echo "########## ABLATION: ${mode} ##########"
  python -u "${EXP_DIR}/train.py" --loss_mode "${mode}" --tag "ablation_${mode}" "$@"
done
echo "=== Done ==="
