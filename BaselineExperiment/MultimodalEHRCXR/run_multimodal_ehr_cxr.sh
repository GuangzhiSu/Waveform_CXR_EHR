#!/bin/bash
# Multimodal EHR+CXR training (CLIP + CE). Run from BaselineExperiment:
#   bash MultimodalEHRCXR/run_multimodal_ehr_cxr.sh
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}:${PYTHONPATH}"
python MultimodalEHRCXR/train.py "$@"
