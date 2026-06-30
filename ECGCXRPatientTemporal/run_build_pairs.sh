#!/bin/bash
# Build staged sequence pairs for Exp3 and Exp4 (CPU; reads catalogs + CXR metadata).
# This emits both cache/seq_target_pairs.json and cache/patient_temporal_pairs.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"
python -u "${SCRIPT_DIR}/build_seq_pairs.py" "$@"
