#!/usr/bin/env bash
# =============================================================================
# 4DVarNet-FM: Scheduler with live-updating synthesis report
# =============================================================================
# Usage:
#   ./run_with_scheduler.sh                         # Run all experiments
#   ./run_with_scheduler.sh --experiment A1_baseline # Run specific experiment
#   POLL_INTERVAL=300 ./run_with_scheduler.sh        # Custom poll interval (s)
#
# The synthesis report in outputs/synthesis_report.pdf is regenerated every
# POLL_INTERVAL seconds (default: 120) and whenever an experiment finishes.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-fdv}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
REPORT="${SCRIPT_DIR}/reports/outputs/synthesis_report.pdf"
PID_FILE="/tmp/4dvarnet_scheduler.pid"
RUNNER_PID_FILE="/tmp/4dvarnet_runner.pid"

# ── helpers ─────────────────────────────────────────────────────────────────
cleanup() {
    echo ""; echo "=== Shutting down scheduler ==="
    rm -f "$PID_FILE" "$RUNNER_PID_FILE" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

generate_report() {
    local ts
    ts=$(date '+%H:%M:%S')
    echo "  [${ts}] Regenerating report ..."
    conda run -n "$CONDA_ENV" python3 reports/generate_report.py --output "$REPORT" 2>&1 | \
        sed 's/^/    /'
    echo "  [${ts}] Report updated: ${REPORT}"
}

# ── banner ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        4DVarNet-FM  —  Scheduler with Live Report              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Environment:  ${CONDA_ENV}"
echo "  Poll every:   ${POLL_INTERVAL}s"
echo "  Report:       ${REPORT}"
echo "  Start:        $(date)"
echo ""

echo "$$" > "$PID_FILE"

# ── initial report ──────────────────────────────────────────────────────────
generate_report

# ── launch experiment runner in background ──────────────────────────────────
echo "=== Launching experiments ..."
RUN_CMD="conda run -n ${CONDA_ENV} python3 run_experiments.py $*"
echo "  ${RUN_CMD}"
$RUN_CMD &
RUNNER_PID=$!
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"
echo "  Runner PID: ${RUNNER_PID}"

# ── polling loop ────────────────────────────────────────────────────────────
while kill -0 "$RUNNER_PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
    generate_report
done

# ── final report ────────────────────────────────────────────────────────────
echo ""
echo "=== Experiments finished ==="
generate_report
echo ""
echo "  Done at: $(date)"
echo "  Final report: ${REPORT}"
echo ""

# If user wants to watch the report live, suggest:
echo "  To watch report updates, run: watch -n 30 ls -la ${REPORT}"
echo ""
