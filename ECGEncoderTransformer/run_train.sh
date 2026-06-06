#!/bin/bash
#SBATCH -J ecg-enc-tr
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ECGEncoderTransformer: frozen baseline2 xresnet1d -> causal transformer -> dual MLP heads (s2f/p2f change).

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${PROJECT_DIR}/ECGEncoderTransformer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}/CXREncoderTransformer:${PROJECT_DIR}:${PROJECT_DIR}/EHRWindowTransformer:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/EHRTrend:${PROJECT_DIR}/experiment1(old)"

MEDTVT_ROOT="$(cd "${PROJECT_DIR}/MedTVT-R1" 2>/dev/null && pwd || cd "${PROJECT_DIR}/../MedTVT-R1" 2>/dev/null && pwd || true)"
ECG_CKPT=""
if [[ -n "${MEDTVT_ROOT}" ]]; then
  if [[ -f "${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt" ]]; then
    ECG_CKPT="${MEDTVT_ROOT}/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt"
  else
    shopt -s nullglob
    _pl_ckpts=("${MEDTVT_ROOT}/CKPTS/"*.ckpt)
    shopt -u nullglob
    if [[ ${#_pl_ckpts[@]} -gt 0 ]]; then
      ECG_CKPT="${_pl_ckpts[0]}"
      for _f in "${_pl_ckpts[@]}"; do
        [[ "${_f}" -nt "${ECG_CKPT}" ]] && ECG_CKPT="${_f}"
      done
    fi
  fi
fi
if [[ -n "${ECG_CKPT}" ]]; then
  export ECG_CKPT
  ECG_CKPT_ARG=(--ecg_ckpt "${ECG_CKPT}")
  echo "Using ECG checkpoint: ${ECG_CKPT}"
else
  ECG_CKPT_ARG=()
  echo "WARNING: no ECG ckpt under ${MEDTVT_ROOT:-<unset>}/CKPTS/ (.pt or .ckpt) — random-init frozen encoder"
fi

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

echo "Slurm partition=${SLURM_JOB_PARTITION:-unknown}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "WARNING: nvidia-smi not found on compute node"
fi

python -u "${TRAIN_SCRIPT}" \
  --ecg_labeled_csv "${PROJECT_DIR}/data/p2f_or_s2f_ecg_catalog_labeled.csv" \
  --output_dir "${OUTPUT_DIR}" \
  "${ECG_CKPT_ARG[@]}" \
  "$@"

echo "=== Done ==="
