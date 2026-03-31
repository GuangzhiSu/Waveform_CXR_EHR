#!/bin/bash
# Diagnose prediction collapse: encoder vs head.
# Usage:
#   ./run_diagnose_collapse.sh              # all models
#   ./run_diagnose_collapse.sh baseline2   # single model
#   ./run_diagnose_collapse.sh --max_samples 50  # quick test

#SBATCH -J diagnose-collapse
#SBATCH -t 01:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

cd "${PROJECT_DIR}"
mkdir -p logs

# If first arg is baseline/baseline2/baseline3 (no --), add --model
if [[ -n "$1" && "$1" != --* && "$1" =~ ^(baseline|baseline2|baseline3)$ ]]; then
  python diagnose_collapse.py --model "$1" "${@:2}"
else
  python diagnose_collapse.py "$@"
fi
