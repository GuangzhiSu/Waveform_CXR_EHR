#!/bin/bash
# Build patient-temporal pairs (CPU; reads catalogs + CXR metadata). Run from repo root.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"
python -u "${SCRIPT_DIR}/build_pairs.py" "$@"
