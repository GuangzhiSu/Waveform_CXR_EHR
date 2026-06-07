#!/bin/bash
#SBATCH -J ehr-tsne-expC
#SBATCH -t 2:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${PROJECT_DIR}"
mkdir -p logs

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/EHREncoderTransformerEmbedPred:${PROJECT_DIR}/EHREncoderTransformer:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/BaselineExperiment"

python -u figures/plot_tsne_embedpred.py \
  --compare_baseline \
  --tsne_n 3000 \
  --out_dir figures/ehr_embedpred_exp_runs

echo "=== Done ==="
