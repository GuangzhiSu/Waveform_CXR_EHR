#!/bin/bash
#SBATCH -J ptc-precompute
#SBATCH -t 12:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o ECGCXRPatientTemporal/artifacts/logs/%x-%j.out
#SBATCH -e ECGCXRPatientTemporal/artifacts/logs/%x-%j.err
# Precompute frozen Bio-ViL-T (CXR) + ECG-CoCa (ECG) embeddings.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
python -u "${EXP_DIR}/precompute_embeddings.py" "$@"
echo "=== Done ==="
