#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/EHRTrend"
cd "${ROOT}"
python "${ROOT}/EHRTrend/train_nextstep.py" \
  --anchor_csv "${ROOT}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --history_csv "${ROOT}/data/p2f_or_s2f_vent_fio2_valid_rows.csv" \
  --schema_csv "${ROOT}/supertable_columns_completed.csv" \
  --enriched_csv "${ROOT}/data/p2f_vent_fio2_enriched.csv" \
  --output_dir "${ROOT}/EHRTrend/output_nextstep" \
  "$@"
