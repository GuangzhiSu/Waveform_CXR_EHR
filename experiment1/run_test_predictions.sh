#!/bin/bash
#SBATCH -J test-predictions
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

# Run test set only, save per-sample predictions to CSV.
# Usage:
#   ./run_test_predictions.sh              # run all three models
#   ./run_test_predictions.sh baseline2    # run only baseline2
#   sbatch run_test_predictions.sh         # submit via SLURM (all models)



set -e
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

cd "${PROJECT_DIR}"
mkdir -p logs

MODELS="${1:-baseline baseline2 baseline3}"
for m in ${MODELS}; do
  echo "=== Running test predictions for ${m} ==="
  python run_test_predictions.py --model "${m}" --output "predictions_${m}.csv"
done

echo "Done. Predictions saved to predictions_*.csv"
