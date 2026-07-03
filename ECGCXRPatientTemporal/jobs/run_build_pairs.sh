#!/bin/bash
# Build staged sequence pairs for Exp3 and Exp4 (CPU; reads catalogs + CXR metadata).
# This emits both artifacts/cache/default/seq_target_pairs.json and
# artifacts/cache/default/patient_temporal_pairs.json.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${JOB_DIR}/.." && pwd)"
source "${EXP_DIR}/setup_env.sh"
python -u "${EXP_DIR}/build_seq_pairs.py" "$@"
