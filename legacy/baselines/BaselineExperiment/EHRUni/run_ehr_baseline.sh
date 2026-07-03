#!/bin/bash
#SBATCH -J ehr-ards-baseline
#SBATCH -t 12:00:00
#SBATCH -A kamaleswaranlab
# #SBATCH -p gpu-common   # Partition empty in assoc; use default
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -G 1
# Write to submit-directory (no logs/ subdir required; avoids sbatch -o relative-path failures)
#SBATCH -o ehr-ards-baseline-%j.out
#SBATCH -e ehr-ards-baseline-%j.err

# EHR baseline: extract EHR+classified data -> train EHR-only ARDS severity
set -e

# Resolve Waveform_CXR_EHR repo root (directory that contains data/supertable_columns_completed.csv).
# Important: under sbatch, BASH_SOURCE may point at a Slurm spool copy — do not rely on it for PROJECT_DIR.
_find_project_root_from() {
  local dir="$1"
  [[ -z "$dir" ]] && return 1
  dir="$(cd "$dir" && pwd)"
  local guard=0
  while [[ "$dir" != "/" && guard -lt 64 ]]; do
    if [[ -f "$dir/data/supertable_columns_completed.csv" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
    guard=$((guard + 1))
  done
  return 1
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if ! PROJECT_DIR="$(_find_project_root_from "${SLURM_SUBMIT_DIR}")"; then
    echo "ERROR: Could not find project root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} (expected .../data/supertable_columns_completed.csv)." >&2
    echo "       Submit from anywhere under the repo, or set WAVEFORM_CXR_ROOT to the repo path." >&2
    if [[ -n "${WAVEFORM_CXR_ROOT:-}" ]] && [[ -f "${WAVEFORM_CXR_ROOT}/data/supertable_columns_completed.csv" ]]; then
      PROJECT_DIR="$(cd "${WAVEFORM_CXR_ROOT}" && pwd)"
      echo "       Using WAVEFORM_CXR_ROOT=${PROJECT_DIR}" >&2
    else
      exit 1
    fi
  fi
else
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
  if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
    _SCRIPT_PATH="$(readlink -f "${_SCRIPT_PATH}")"
  else
    _SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  fi
  SCRIPT_DIR_LOCAL="$(dirname "${_SCRIPT_PATH}")"
  PROJECT_DIR="$(cd "${SCRIPT_DIR_LOCAL}/../.." && pwd)"
fi

SCRIPT_DIR="${PROJECT_DIR}/BaselineExperiment/EHRUni"

DATA_DIR="${PROJECT_DIR}/data"
ENRICHED_CSV="${DATA_DIR}/p2f_vent_fio2_enriched.csv"
CLASSIFIED_CSV="${DATA_DIR}/p2f_ehr_classified.csv"
EXTRACT_SCRIPT="${DATA_DIR}/extract_ehr_p2f_classified.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

# Step 1: Extract EHR rows with p2f classification (skip if already done)
if [[ -f "${CLASSIFIED_CSV}" ]]; then
  echo "=== Step 1: SKIP (${CLASSIFIED_CSV} exists) ==="
else
  echo "=== Step 1: Extract EHR rows and add ARDS severity class ==="
  python "${EXTRACT_SCRIPT}" --input "${ENRICHED_CSV}" --output "${CLASSIFIED_CSV}"
  if [[ ! -f "${CLASSIFIED_CSV}" ]]; then
    echo "ERROR: Classified CSV not created"
    exit 1
  fi
fi

# Step 2: Train EHR classification model
echo "=== Step 2: Train EHR ARDS severity classification model ==="
python "${TRAIN_SCRIPT}" \
  --csv_path "${CLASSIFIED_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo "=== Done ==="
