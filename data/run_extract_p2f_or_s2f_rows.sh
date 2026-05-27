#!/bin/bash
#SBATCH -J extract-p2f-s2f
#SBATCH -t 12:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o extract-p2f-s2f-%j.out
#SBATCH -e extract-p2f-s2f-%j.err

set -e

PROJECT_DIR="/work/gs285/Waveform_CXR_EHR"
cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate base 2>/dev/null || true
}

python data/extract_p2f_or_s2f_rows.py
