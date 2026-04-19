#!/bin/bash
# Train EHR+ECG with CLIP contrastive loss + classification (see MultimodalEHRECG/config.py).
# Run from repo: sbatch / path/to/BaselineExperiment/MultimodalEHRECG/run_multimodal_ehr_ecg.sh
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}:${PYTHONPATH}"
python MultimodalEHRECG/train.py "$@"
