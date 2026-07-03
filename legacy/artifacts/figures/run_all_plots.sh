#!/bin/bash
# Generate all loss/acc figures on login node (matplotlib only, no GPU).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/EHRTrend"

[[ -n "$(command -v conda)" ]] && {
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
}

echo "=== plot_embedpred_exp_runs ==="
python -u figures/plot_embedpred_exp_runs.py

echo "=== plot_symile_runs ==="
python -u figures/plot_symile_runs.py

echo "=== plot_ehr_tr_fix_runs ==="
python -u figures/plot_ehr_tr_fix_runs.py

echo "=== plot_ehr_window_tr ==="
python -u figures/plot_ehr_window_tr.py

echo "=== plot_cxr_ecg_runs ==="
python -u figures/plot_cxr_ecg_runs.py

echo "=== Done — PNGs under figures/ ==="
