#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=qg-specwind-da
#SBATCH --output=sbatch_logs/qg_specwind_da_%A_%a.out
#SBATCH --error=sbatch_logs/qg_specwind_da_%A_%a.err
# S0 ETKF/EnKF on the spectral-wind QG datasets (evaluation/run_qg_specwind_da.py).
# One array task = one shard of the window range; the data are read from the staged
# copy under experiments/qg_datasets (not the generation server's /SCRATCH). Example:
#   SPEC=qg_specwind_gyrostat_v1 METHOD=etkf COLS=3 N_WINDOWS=100 N_SHARDS=10 \
#     sbatch --array=0-9 batch/run_qg_specwind_da.sbatch
# Tuning on val: add SPLIT=val and override LOC / RIDGE.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
: "${SPEC:?set SPEC}" "${METHOD:?set METHOD}" "${N_SHARDS:?set N_SHARDS}"
SPLIT="${SPLIT:-test}"; COLS="${COLS:-3}"; N_WINDOWS="${N_WINDOWS:-100}"; START="${START:-0}"
LOC="${LOC:-2.0}"; RIDGE="${RIDGE:-0.1}"; ENS="${ENS:-80}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
OUT="experiments/qg_specwind_da/${SPEC}/${SPLIT}/${METHOD}_c${COLS}_loc${LOC}_r${RIDGE}_s${START}_n${N_WINDOWS}/shard_${SHARD}_of_${N_SHARDS}.json"
mkdir -p "$(dirname "$OUT")"
"$PY" -m evaluation.run_qg_specwind_da --spec "$SPEC" --root experiments/qg_datasets \
  --split "$SPLIT" --start "$START" --n-windows "$N_WINDOWS" --shard "$SHARD" --n-shards "$N_SHARDS" \
  --method "$METHOD" --cols-per-day "$COLS" --N "$ENS" --loc-radius "$LOC" --etkf-ridge "$RIDGE" \
  --out "$OUT" --device cuda
