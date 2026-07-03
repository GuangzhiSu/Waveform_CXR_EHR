#!/bin/bash
#SBATCH -J ehr-enc-analyze
#SBATCH -t 2:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHREncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/EHREncoderTransformerEmbedPred"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -u "${SCRIPT_DIR}/analyze_runs.py" \
  --log_tr logs/ehr-enc-tr-47410738.out \
  --log_embed logs/ehr-enc-embed-47414328.out \
  --ckpt_tr EHREncoderTransformer/output/best.pt \
  --ckpt_embed EHREncoderTransformerEmbedPred/output/best.pt \
  --out_dir figures/ehr_enc_47410738_47414328 \
  --tsne_n 3000 \
  --seed 42 \
  "$@"

echo "=== Done ==="
