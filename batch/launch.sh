#!/usr/bin/env bash
# Launch 4DVarNet-FM experiments in background with live report updates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-fdv}"
REPORT="${SCRIPT_DIR}/reports/outputs/synthesis_report.pdf"
LOG="${SCRIPT_DIR}/experiments/pipeline.log"
POLL_INTERVAL="${POLL_INTERVAL:-120}"

mkdir -p outputs experiments reports/outputs

echo "============================================" | tee -a "$LOG"
echo " 4DVarNet-FM Pipeline  |  $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  Env:   ${CONDA_ENV}" | tee -a "$LOG"
echo "  Log:   ${LOG}" | tee -a "$LOG"
echo "  Poll:  ${POLL_INTERVAL}s" | tee -a "$LOG"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Initial report
conda run -n "$CONDA_ENV" python3 reports/generate_report.py --output "$REPORT" >> "$LOG" 2>&1
echo "  Initial report: ${REPORT}" | tee -a "$LOG"

# Launch pipeline in background
nohup conda run -n "$CONDA_ENV" python3 -u run_experiments.py "$@" >> "$LOG" 2>&1 &
PIPELINE_PID=$!
echo "  Pipeline PID: ${PIPELINE_PID}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Polling loop
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
    echo "  [$(date '+%H:%M:%S')] Regenerating report ..." | tee -a "$LOG"
    conda run -n "$CONDA_ENV" python3 reports/generate_report.py --output "$REPORT" >> "$LOG" 2>&1
    echo "  [$(date '+%H:%M:%S')] Report updated" | tee -a "$LOG"
done

# Final report
echo "" | tee -a "$LOG"
echo "  Pipeline finished at $(date)" | tee -a "$LOG"
conda run -n "$CONDA_ENV" python3 reports/generate_report.py --output "$REPORT" >> "$LOG" 2>&1
echo "  Final report: ${REPORT}" | tee -a "$LOG"
