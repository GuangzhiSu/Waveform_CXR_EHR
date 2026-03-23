#!/bin/bash
#SBATCH -J baseline-train
#SBATCH -t 24:00:00
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
#SBATCH -x dcc-core-gpu-ferc-s-p15-20

# Run baseline: predict oxygenation from ECG + CXR (EHR = ground truth)
#
# Data paths:
#   - EHR/Matched CSV: CXR+ECG+EHR rows, oxygenation in target_col
#   - CXR images: MIMIC-CXR-JPG
#   - ECG waveforms: full path in wf_File_Path column of CSV
# Encoders: ViT (CXR), xresnet1d101 (ECG) from MedTVT-R1

set -e
# Under sbatch, BASH_SOURCE is the spool copy - use SLURM_SUBMIT_DIR instead
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/baseline"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
MEDTVT_ROOT="$(cd "${PROJECT_DIR}/../MedTVT-R1" && pwd)"

# Data paths
CSV_PATH="${PROJECT_DIR}/cxr_supertable_waveform_matched.csv"
CXR_ROOT="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg"
METADATA_PATH="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
# ECG paths are in wf_File_Path column (full path, e.g. /hpc/.../MIMIC_IV_ECG_Matched/files/...)

# Modality encoder checkpoints (MedTVT-R1)
ECG_CKPT="${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt"
VIT_PATH="${MEDTVT_ROOT}/CKPTS/vit-base-patch16-224"
# Alternative: VIT_PATH="google/vit-base-patch16-224-in21k" (downloads from HF)

# Output
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Sanity checks
for p in "${CSV_PATH}" "${CXR_ROOT}" "${METADATA_PATH}" "${ECG_CKPT}" "${VIT_PATH}"; do
  if [[ "$p" != *"google"* ]] && [[ ! -e "$p" ]]; then
    echo "WARNING: Path not found: $p"
  fi
done

echo "=== Baseline: Predict oxygenation from ECG + CXR ==="
echo "  CSV (EHR ground truth): ${CSV_PATH}"
echo "  CXR root:               ${CXR_ROOT}"
echo "  CXR metadata:           ${METADATA_PATH}"
echo "  ECG encoder:            ${ECG_CKPT}"
echo "  CXR encoder (ViT):      ${VIT_PATH}"
echo "  Output:                 ${OUTPUT_DIR}"
echo ""

cd "${SCRIPT_DIR}"

# Activate conda env (adjust for your setup)
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || source activate MedTVT-R1 2>/dev/null || true
fi

# NumPy 2.x breaks PyTorch/matplotlib compiled against NumPy 1.x - downgrade if needed
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

python train.py \
  --csv_path "${CSV_PATH}" \
  --cxr_root "${CXR_ROOT}" \
  --metadata_path "${METADATA_PATH}" \
  --ecg_ckpt "${ECG_CKPT}" \
  --vit_path "${VIT_PATH}" \
  --target_col p2f_vent_fio2 \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
