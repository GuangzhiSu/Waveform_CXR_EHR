#!/bin/bash
#SBATCH -J ehr-fwd-mlp
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Step-1 pipeline: single-row EHR (percentile) -> EHRMLPEncoder -> dual MLP heads
# predicting forward severity change in [t+12h, t+24h] for s2f and p2f (3-class each).

set -euo pipefail

# When Slurm copies this script under /var/spool/slurmd/..., BASH_SOURCE would make
# PROJECT_DIR=/var/spool/slurmd and "mkdir -p ${PROJECT_DIR}/logs" hits Permission denied.
# Prefer the directory from which sbatch was invoked (repo root).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHRTrend"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

python -u "${SCRIPT_DIR}/train_forward_mlp.py" \
  --source_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --schema_csv "${PROJECT_DIR}/supertable_columns_completed.csv" \
  --enriched_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv" \
  --output_dir "${SCRIPT_DIR}/output_forward_mlp" \
  "$@"

echo "=== Done ==="
