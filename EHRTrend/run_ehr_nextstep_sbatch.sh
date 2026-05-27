#!/bin/bash
#SBATCH -J ehr-nextstep
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

# Extended EHR experiment: causal Transformer + row MLP encoder + next-step MSE
# + anchor / per-step severity-change heads (train_nextstep.py).

set -euo pipefail

# See run_ehr_forward_mlp_sbatch.sh: under Slurm, BASH_SOURCE may live under /var/spool/slurmd;
# use SLURM_SUBMIT_DIR so PROJECT_DIR is the repo root where sbatch was run.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/EHRTrend"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train_nextstep.py"
OUTPUT_DIR="${SCRIPT_DIR}/output_nextstep"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

PRETRAINED_FWD="${SCRIPT_DIR}/output_forward_mlp/best.pt"
NEXTSTEP_ARGS=(
  --anchor_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv"
  --history_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv"
  --schema_csv "${PROJECT_DIR}/supertable_columns_completed.csv"
  --enriched_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -f "${PRETRAINED_FWD}" ]]; then
  NEXTSTEP_ARGS+=( --pretrained_forward_mlp_ckpt "${PRETRAINED_FWD}" )
fi

python -u "${TRAIN_SCRIPT}" "${NEXTSTEP_ARGS[@]}" "$@"

echo "=== Done ==="
