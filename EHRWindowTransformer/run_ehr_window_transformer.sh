#!/bin/bash
#SBATCH -J ehr-window-transformer
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

# Train DirectWindowTransformer on EHR rows in [anchor_t - 24h, anchor_t - 12h] only.
# Usage:
#   ./EHRWindowTransformer/run_ehr_window_transformer.sh
#   ./EHRWindowTransformer/run_ehr_window_transformer.sh --epochs 10 --max_samples 5000
#   sbatch EHRWindowTransformer/run_ehr_window_transformer.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# #region agent log
export _AGENT_DBG_SCRIPT_DIR="${SCRIPT_DIR}"
export _AGENT_DBG_BASH_SOURCE="${BASH_SOURCE[0]}"
export _AGENT_DBG_SLURM_SD="${SLURM_SUBMIT_DIR:-}"
python -c "import json,time,os;p=os.environ.get('_AGENT_DBG_SCRIPT_DIR','');open('/work/gs285/Waveform_CXR_EHR/.cursor/debug-d06dda.log','a').write(json.dumps({'sessionId':'d06dda','timestamp':int(time.time()*1000),'hypothesisId':'H1','location':'run_ehr_window_transformer.sh:pre_slurm_scriptdir','message':'BASH_SOURCE vs SCRIPT_DIR','data':{'SCRIPT_DIR':os.environ.get('_AGENT_DBG_SCRIPT_DIR'),'BASH_SOURCE':os.environ.get('_AGENT_DBG_BASH_SOURCE'),'SLURM_SUBMIT_DIR':os.environ.get('_AGENT_DBG_SLURM_SD'),'train_py_here':__import__('os').path.isfile(__import__('os').path.join(os.environ.get('_AGENT_DBG_SCRIPT_DIR',''),'train.py'))}})+'\n')" 2>/dev/null || true
unset _AGENT_DBG_SCRIPT_DIR _AGENT_DBG_BASH_SOURCE _AGENT_DBG_SLURM_SD
# #endregion

# sbatch copies this script under /var/spool/slurmd/job*/ ; BASH_SOURCE is not the repo path.
if [[ ! -f "${SCRIPT_DIR}/train.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/EHRWindowTransformer/train.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/EHRWindowTransformer"
  elif [[ -f "${SLURM_SUBMIT_DIR}/train.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi

# #region agent log
export _AGENT_DBG_SCRIPT_DIR="${SCRIPT_DIR}"
export _AGENT_DBG_SLURM_SD="${SLURM_SUBMIT_DIR:-}"
python -c "import json,time,os;p=os.environ.get('_AGENT_DBG_SCRIPT_DIR','');open('/work/gs285/Waveform_CXR_EHR/.cursor/debug-d06dda.log','a').write(json.dumps({'sessionId':'d06dda','timestamp':int(time.time()*1000),'hypothesisId':'H2','location':'run_ehr_window_transformer.sh:post_slurm_scriptdir','message':'SCRIPT_DIR after spool fallback','data':{'SCRIPT_DIR':os.environ.get('_AGENT_DBG_SCRIPT_DIR'),'SLURM_SUBMIT_DIR':os.environ.get('_AGENT_DBG_SLURM_SD'),'train_py_here':__import__('os').path.isfile(__import__('os').path.join(os.environ.get('_AGENT_DBG_SCRIPT_DIR',''),'train.py'))}})+'\n')" 2>/dev/null || true
unset _AGENT_DBG_SCRIPT_DIR _AGENT_DBG_SLURM_SD
# #endregion

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

[[ -n "$(command -v conda)" ]] && { eval "$(conda shell.bash hook 2>/dev/null)" || true; conda activate MedTVT-R1 2>/dev/null || true; }
python -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null || pip install "numpy<2" --quiet

cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/BaselineExperiment:${PROJECT_DIR}/EHRTrend:${SCRIPT_DIR}"

python -u "${SCRIPT_DIR}/train.py" \
  --anchor_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --history_csv "${PROJECT_DIR}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --schema_csv "${PROJECT_DIR}/supertable_columns_completed.csv" \
  --enriched_csv "${PROJECT_DIR}/data/p2f_vent_fio2_enriched.csv" \
  --output_dir "${SCRIPT_DIR}/output_direct_window" \
  "$@"

echo "=== Done ==="
