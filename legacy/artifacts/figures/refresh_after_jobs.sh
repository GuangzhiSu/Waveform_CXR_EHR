#!/bin/bash
# Wait for follow-up Slurm jobs, regenerate plots, print result summary.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

JOBIDS="${FOLLOWUP_JOBIDS:-47837615,47837616,47837617,47837618,47837619,47837620,47837621,47837622,47837623}"
POLL_SEC="${POLL_SEC:-300}"

echo "Waiting for jobs: ${JOBIDS}"
while true; do
  pending=$(squeue -u "${USER}" -h -j "${JOBIDS}" 2>/dev/null | wc -l)
  if [[ "${pending}" -eq 0 ]]; then
    break
  fi
  echo "$(date -Iseconds)  ${pending} job(s) still running..."
  sleep "${POLL_SEC}"
done

echo "All follow-up jobs finished. Regenerating plots..."
bash figures/run_all_plots.sh

echo ""
echo "=== Follow-up test accuracy summary ==="
python -u figures/summarize_followup_results.py

echo "Done at $(date -Iseconds)"
