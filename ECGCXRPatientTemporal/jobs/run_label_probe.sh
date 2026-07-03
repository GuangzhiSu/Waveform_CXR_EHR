#!/bin/bash
#SBATCH -J ptc-label-probe
#SBATCH -t 08:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -G 1
#SBATCH -o ECGCXRPatientTemporal/artifacts/logs/%x-%j.out
#SBATCH -e ECGCXRPatientTemporal/artifacts/logs/%x-%j.err
# Frozen contrastive embedding -> CXR annotation label prediction.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  EXP_DIR="${PROJECT_DIR}/ECGCXRPatientTemporal"
else
  JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  EXP_DIR="$(cd "${JOB_DIR}/.." && pwd)"
  PROJECT_DIR="$(cd "${EXP_DIR}/.." && pwd)"
fi
mkdir -p "${EXP_DIR}/artifacts/logs"
cd "${PROJECT_DIR}"
source "${EXP_DIR}/setup_env.sh"
python -u "${EXP_DIR}/label_probe.py" "$@"
echo "=== Done ==="
