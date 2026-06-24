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
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Train the patient-temporal contrastive baseline. Pass-through args, e.g.:
#   sbatch ECGCXRPatientTemporal/run_train.sh --loss_mode combined
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"; SCRIPT_DIR="${PROJECT_DIR}/ECGCXRPatientTemporal"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"
source "${SCRIPT_DIR}/setup_env.sh"
python -u "${SCRIPT_DIR}/train.py" "$@"
echo "=== Done ==="
