#!/bin/bash
#SBATCH -J ptc-train
#SBATCH -t 12:00:00
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
# Train the patient-temporal contrastive baseline. Pass-through args, e.g.:
#   sbatch ECGCXRPatientTemporal/jobs/run_train.sh --loss_mode combined
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
python -u "${EXP_DIR}/train.py" "$@"
echo "=== Done ==="
