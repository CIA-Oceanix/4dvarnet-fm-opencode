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
# Array use: put one "METHOD|ARGS|SPLIT|LAYOUT|CASE|FDV_SEED" line per task in ROWS_FILE;
# task i runs line i (0-based); unset fields fall back to the env vars above. Throttle arrays
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

METHOD="${METHOD:-B3}" ARGS="${ARGS:-}" SPLIT="${SPLIT:-val}" LAYOUT="${LAYOUT:-regular}"
CASE="${CASE:-s0}" FDV_SEED="${FDV_SEED:-2}"
if [ -n "${ROWS_FILE:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROWS_FILE")
  IFS='|' read -r M A S L C F <<< "$LINE"
  METHOD="${M:-$METHOD}" ARGS="${A:-$ARGS}" SPLIT="${S:-$SPLIT}" LAYOUT="${L:-$LAYOUT}"
  CASE="${C:-$CASE}" FDV_SEED="${F:-$FDV_SEED}"
fi
OUT_DIR="${OUT_DIR:-experiments/l96_fdv2_fm_posterior}"
TAG=$(echo "${ARGS:-default}" | tr ',=' '_-')
OUT="$OUT_DIR/${SPLIT}_${LAYOUT}_${CASE}_fdvs${FDV_SEED}/${METHOD}_${TAG}.json"
mkdir -p "$(dirname "$OUT")"
echo "===== $METHOD [$ARGS] $SPLIT/$LAYOUT/$CASE fdv_seed=$FDV_SEED | $(date) | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
EXTRA=()
[ -n "${WINDOWS:-}" ] && EXTRA+=(--windows "$WINDOWS")
"$PY" eval_fdv_fm_posterior_l96.py --method "$METHOD" ${ARGS:+--args "$ARGS"} --split "$SPLIT" \
  --layout "$LAYOUT" --case "$CASE" --fdv-seed "$FDV_SEED" --n-members "${N_MEMBERS:-30}" \
  --seed "${SEED:-0}" --out "$OUT" "${EXTRA[@]}"
echo "===== done $(date) -> $OUT ====="
