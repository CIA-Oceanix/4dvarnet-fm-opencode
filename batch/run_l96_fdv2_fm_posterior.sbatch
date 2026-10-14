#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-fdv2-fmpost
#SBATCH --output=sbatch_logs/l96_fdv2_fm_posterior_%A_%a.out
# One row of the L96 FDV2 x FM posterior harness (eval_fdv_fm_posterior_l96.py,
# docs/plans/analysis/l96_fdv2_fm_posterior.md). Env vars:
#   METHOD (B1..B5, M*), ARGS (key=value,...), SPLIT (val|test), LAYOUT (regular|random),
#   CASE (s0|s1), FDV_SEED (1|2), N_MEMBERS (30), WINDOWS (subset, default all), SEED (0),
#   OUT_DIR (default experiments/l96_fdv2_fm_posterior), CONDA_ENV (fdv-monai-proto).
# Array use: put one "METHOD|ARGS|SPLIT|LAYOUT|CASE|FDV_SEED[|WINDOWS|NAME]" line per task in ROWS_FILE;
# task i runs lines i*K..i*K+K-1 (0-based, K = ROWS_PER_TASK, default 1) in one Python process;
# unset fields fall back to the env vars above; SKIP_EXISTING=1 skips rows already written. NAME (optional)
# writes OUT_DIR/NAME.json instead of the default per-split/layout path. Throttle arrays
# (e.g. --array=0-15%2) to keep the project within half of the Odyssey_GPU nodes.
# Only per-window scores are written to OUT_DIR; members live in memory (or node-local TMPDIR).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONDA_ENV="${CONDA_ENV:-fdv-monai-proto}"
PY=/Odyssey/private/rfablet/miniforge3/envs/$CONDA_ENV/bin/python
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p sbatch_logs

OUT_DIR="${OUT_DIR:-experiments/l96_fdv2_fm_posterior}"
ARGV_FILE="$NODE_TMP/argv.txt"
: > "$ARGV_FILE"
add_row() {
  local METHOD="${1:-${METHOD:-B3}}" ARGS="${2:-${ARGS:-}}" SPLIT="${3:-${SPLIT:-val}}" LAYOUT="${4:-${LAYOUT:-regular}}"
  local CASE="${5:-${CASE:-s0}}" FDV_SEED="${6:-${FDV_SEED:-2}}" WIN="${7:-${WINDOWS:-}}" NAME="${8:-}" OUT TAG
  TAG=$(echo "${ARGS:-default}" | tr ',=' '_-')
  OUT="$OUT_DIR/${SPLIT}_${LAYOUT}_${CASE}_fdvs${FDV_SEED}/${METHOD}_${TAG}.json"
  [ -n "$NAME" ] && OUT="$OUT_DIR/$NAME.json"
  if [ "${SKIP_EXISTING:-0}" = 1 ] && [ -s "$OUT" ]; then echo "skip existing $OUT"; return; fi
  mkdir -p "$(dirname "$OUT")"
  printf '%q ' --method "$METHOD" ${ARGS:+--args "$ARGS"} --split "$SPLIT" --layout "$LAYOUT" --case "$CASE" \
    --fdv-seed "$FDV_SEED" --n-members "${N_MEMBERS:-30}" --seed "${SEED:-0}" --out "$OUT" \
    ${WIN:+--windows "$WIN"} >> "$ARGV_FILE"
  echo >> "$ARGV_FILE"
  echo "row: $METHOD [$ARGS] $SPLIT/$LAYOUT/$CASE fdv_seed=$FDV_SEED windows=${WIN:-all} -> $OUT"
}
if [ -n "${ROWS_FILE:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  K="${ROWS_PER_TASK:-1}"
  for ((j = SLURM_ARRAY_TASK_ID * K + 1; j <= (SLURM_ARRAY_TASK_ID + 1) * K; j++)); do
    LINE=$(sed -n "${j}p" "$ROWS_FILE")
    [ -z "$LINE" ] && continue
    IFS='|' read -r M A S L C F W N <<< "$LINE"
    add_row "$M" "$A" "$S" "$L" "$C" "$F" "$W" "$N"
  done
else
  add_row
fi
echo "===== $(wc -l < "$ARGV_FILE") rows | $(date) | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
[ -s "$ARGV_FILE" ] && "$PY" eval_fdv_fm_posterior_l96.py --argv-file "$ARGV_FILE"
echo "===== done $(date) ====="
