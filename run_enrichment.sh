#!/bin/bash
# Run CXR/ECG 6-12h lookback enrichment.
# Option 1: Use p2f_vent_fio2_valid_rows.csv (run extract_p2f_rows.py first)
# Option 2: Use cxr_supertable_matched or cxr_supertable_waveform_matched
set -e
BASE="/work/gs285/Waveform_CXR_EHR"
cd "$BASE"

INPUT="${1:-$BASE/p2f_vent_fio2_valid_rows.csv}"
OUTPUT="${2:-$BASE/p2f_vent_fio2_enriched.csv}"

if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT"
  echo "Run extract_p2f_rows.py first to generate p2f_vent_fio2_valid_rows.csv"
  echo "Or pass another CSV: ./run_enrichment.sh <input.csv> [output.csv]"
  exit 1
fi

echo "Input: $INPUT"
echo "Output: $OUTPUT"
python3 "$BASE/enrich_p2f_cxr_ecg_lookback.py" --input "$INPUT" --output "$OUTPUT"
