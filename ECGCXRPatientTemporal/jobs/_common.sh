#!/bin/bash
# Shared setup for local and Slurm job launchers.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  EXP_DIR="${PROJECT_DIR}/ECGCXRPatientTemporal"
else
  JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  EXP_DIR="$(cd "${JOB_DIR}/.." && pwd)"
  PROJECT_DIR="$(cd "${EXP_DIR}/.." && pwd)"
fi

mkdir -p "${EXP_DIR}/artifacts/logs"
cd "${PROJECT_DIR}"
source "${EXP_DIR}/setup_env.sh"
