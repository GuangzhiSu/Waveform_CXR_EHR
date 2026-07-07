#!/bin/bash
# Build single-ECG -> future-CXR pairs (Experiments 1 & 2) and merge-precompute
# any embeddings that are not already cached. CPU for pair building; the
# precompute step needs a GPU (submit run_precompute.sh after, or run here on a
# GPU node). Pass --restrict_to_cache to skip GPU recompute entirely.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
python -u "${EXP_DIR}/build_single_ecg_pairs.py" "$@"
echo "=== Done building single-ECG pairs ==="
