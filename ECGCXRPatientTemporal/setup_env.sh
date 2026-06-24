#!/bin/bash
# Source this to set up the environment for the ECG-CXR patient-temporal experiment.
#   source ECGCXRPatientTemporal/setup_env.sh
# Activates the MedTVT-R1 conda env and wires the workspace-local pip prefix
# (pylibs/, holds health_multimodal + gdown) and the vendored ECG-R1 repo.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_DIR="$(cd "${_SCRIPT_DIR}/.." && pwd)"

if [[ -n "$(command -v conda)" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate MedTVT-R1 2>/dev/null || true
fi

export PYTHONUSERBASE="${_SCRIPT_DIR}/pylibs"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
_PY_SP="${PYTHONUSERBASE}/lib/python3.9/site-packages"

export PYTHONPATH="${_SCRIPT_DIR}:${_PY_SP}:${_SCRIPT_DIR}/external/ECG-R1:${_PROJECT_DIR}:${_PROJECT_DIR}/BaselineExperiment:${_PROJECT_DIR}/BaselineExperiment/CXRUni:${PYTHONPATH:-}"

echo "env ready: $(python --version 2>&1)  PYTHONUSERBASE=${PYTHONUSERBASE}"
