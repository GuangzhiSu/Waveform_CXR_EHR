#!/bin/bash
#SBATCH -J enrich-or-s2f-modality
#SBATCH -t 24:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o enrich-or-s2f-modality-%j.out
#SBATCH -e enrich-or-s2f-modality-%j.err

set -e

PROJECT_DIR="/work/gs285/Waveform_CXR_EHR"
cd "${PROJECT_DIR}" || exit 1

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate base 2>/dev/null || true
}

# Full run: catalogs + anchor window summary for all p2f_or_s2f rows.
# Add --write-matches for long-format CXR/ECG match tables (much larger outputs).
python data/enrich_or_s2f_modality_window.py "$@"
