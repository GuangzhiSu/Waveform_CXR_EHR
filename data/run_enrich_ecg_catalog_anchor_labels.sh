#!/bin/bash
#SBATCH -J enrich-ecg-labels
#SBATCH -t 04:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o enrich-ecg-labels-%j.out
#SBATCH -e enrich-ecg-labels-%j.err

set -e

PROJECT_DIR="/work/gs285/Waveform_CXR_EHR"
cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate base 2>/dev/null || true
}

python data/enrich_ecg_catalog_anchor_labels.py "$@"
