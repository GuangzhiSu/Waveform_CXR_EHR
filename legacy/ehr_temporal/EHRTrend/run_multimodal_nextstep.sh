#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_exp_old="${ROOT}/experiment1(old)"
export PYTHONPATH="${ROOT}:${ROOT}/EHRTrend:${ROOT}/BaselineExperiment:${_exp_old}"
cd "${ROOT}"
python "${ROOT}/EHRTrend/train_multimodal_nextstep.py" "$@"
