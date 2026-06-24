#!/bin/bash
# Download frozen encoder checkpoints into checkpoints/.
#   - Bio-ViL-T image encoder  (HuggingFace microsoft/BiomedVLP-BioViL-T)
#   - ECG-CoCa encoder         (Google Drive, via gdown)
#
# Run from repo root after `source ECGCXRPatientTemporal/setup_env.sh`.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"
cd "${CKPT_DIR}"

BIOVIL="biovil_t_image_model_proj_size_128.pt"
if [[ -f "${BIOVIL}" ]]; then
  echo "Bio-ViL-T weights already present: ${BIOVIL}"
else
  echo "Downloading Bio-ViL-T image weights from HuggingFace..."
  python - <<'PY'
import urllib.request
u="https://huggingface.co/microsoft/BiomedVLP-BioViL-T/resolve/main/biovil_t_image_model_proj_size_128.pt"
urllib.request.urlretrieve(u, "biovil_t_image_model_proj_size_128.pt")
import os; print("  done", os.path.getsize("biovil_t_image_model_proj_size_128.pt"), "bytes")
PY
fi

ECGCOCA="cpt_wfep_epoch_20.pt"
if [[ -f "${ECGCOCA}" ]]; then
  echo "ECG-CoCa weights already present: ${ECGCOCA}"
else
  echo "Downloading ECG-CoCa weights from Google Drive (file id 1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7)..."
  echo "  NOTE: this Drive file is frequently rate-limited ('Too many users have viewed or"
  echo "        downloaded this file recently'). If it fails, retry later (resets within ~24h)."
  gdown "1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7" -O "${ECGCOCA}" || \
    echo "  ECG-CoCa download FAILED (quota). Retry later or download manually from:"
  [[ -f "${ECGCOCA}" ]] || echo "    https://drive.google.com/file/d/1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7/view"
fi

echo "=== checkpoints/ ==="
ls -la "${CKPT_DIR}"
