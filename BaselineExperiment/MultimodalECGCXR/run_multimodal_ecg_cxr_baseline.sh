#!/bin/bash
#SBATCH -J mm-ecg-cxr-ards
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Multimodal ECG + CXR ARDS severity (concat 512+512 -> MLP head). No LoRA.
#
# Usage:
#   sbatch BaselineExperiment/MultimodalECGCXR/run_multimodal_ecg_cxr_baseline.sh
#
# Requires:
#   - data/p2f_ecg_cxr_multimodal.csv (run data/extract_ecg_cxr_multimodal_classified.py)
#   - MedTVT-R1 CKPTS for ViT + ECG (optional ECG ckpt if path missing)
set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/BaselineExperiment/MultimodalECGCXR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DATA_DIR="${PROJECT_DIR}/data"
CXR_CSV="${DATA_DIR}/p2f_cxr_classified.csv"
ECG_CSV="${DATA_DIR}/p2f_ecg_all_classified.csv"
MM_CSV="${DATA_DIR}/p2f_ecg_cxr_multimodal.csv"
EXTRACT_MM="${DATA_DIR}/extract_ecg_cxr_multimodal_classified.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
CXR_ROOT="${CXR_ROOT:-/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg}"
METADATA_PATH="${METADATA_PATH:-/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz}"
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/MedTVT-R1" 2>/dev/null && pwd || cd "${PROJECT_DIR}/../MedTVT-R1" 2>/dev/null && pwd || true)"
if [[ -n "${MEDTVT_ROOT}" && -d "${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224" ]]; then
  VIT_PATH="${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224"
else
  VIT_PATH="google/vit-base-patch16-224-in21k"
fi
if [[ -n "${MEDTVT_ROOT}" && -f "${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt" ]]; then
  ECG_CKPT="${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt"
else
  ECG_CKPT=""
fi

# Python imports ``llama`` before argparse; medtvt_paths uses env + path hints.
if [[ -n "${MEDTVT_ROOT}" && -d "${MEDTVT_ROOT}/llama" ]]; then
  export MEDTVT_ROOT
fi
if [[ -n "${VIT_PATH}" && ! "${VIT_PATH}" =~ ^google/ ]]; then
  export VIT_PATH
fi
if [[ -n "${ECG_CKPT}" ]]; then
  export ECG_CKPT
fi

OUTPUT_DIR="${SCRIPT_DIR}/output"

cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

# Step 1: Build multimodal CSV (inner join on supertable index)
if [[ -f "${MM_CSV}" ]]; then
  echo "=== Step 1: SKIP (${MM_CSV} exists) ==="
else
  echo "=== Step 1: Build multimodal CSV (CXR + ECG on same index) ==="
  if [[ ! -f "${CXR_CSV}" || ! -f "${ECG_CSV}" ]]; then
    echo "ERROR: Need ${CXR_CSV} and ${ECG_CSV}"
    exit 1
  fi
  python "${EXTRACT_MM}" --cxr_csv "${CXR_CSV}" --ecg_csv "${ECG_CSV}" --output "${MM_CSV}"
fi

echo "=== Step 2: Train multimodal ECG+CXR classifier ==="
# --train_diag: val prediction histograms each epoch (optional; remove to shorten logs)
python "${TRAIN_SCRIPT}" \
  --csv_path "${MM_CSV}" \
  --cxr_root "${CXR_ROOT}" \
  --metadata_path "${METADATA_PATH}" \
  --vit_path "${VIT_PATH}" \
  ${ECG_CKPT:+--ecg_ckpt "${ECG_CKPT}"} \
  --output_dir "${OUTPUT_DIR}" \
  --train_diag \
  "$@"

echo "=== Done ==="
