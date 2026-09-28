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
# Defaults are the DA-2 val-tuned settings (docs/results/qg_specwind_da2_val_tuning.md);
# override LOC / RIDGE / CROSS / INIT to reproduce other configurations. The EnKF's tuned
# radius is 6: METHOD=enkf LOC=6.0.
# KAPPA > 0 runs S1 at that intensity (evaluation/qg_specwind_s1.py reference levels);
# PRESET=realistic KAPPA=1 [VARIANT=base] runs the realism-anchored S1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
: "${SPEC:?set SPEC}" "${METHOD:?set METHOD}" "${N_SHARDS:?set N_SHARDS}"
SPLIT="${SPLIT:-test}"; COLS="${COLS:-3}"; N_WINDOWS="${N_WINDOWS:-100}"; START="${START:-0}"
LOC="${LOC:-8.0}"; RIDGE="${RIDGE:-1.0}"; ENS="${ENS:-80}"; CROSS="${CROSS:-1.0}"; INIT="${INIT:-bred}"; DISP="${DISP:-1.0}"; KAPPA="${KAPPA:-0}"; PRESET="${PRESET:-reference}"; VARIANT="${VARIANT:-base}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
OUT="experiments/qg_specwind_da/${SPEC}/${SPLIT}/${METHOD}_c${COLS}_loc${LOC}_r${RIDGE}_x${CROSS}_${INIT}_d${DISP}_k${KAPPA}$([ "$PRESET" = realistic ] && echo "_r${VARIANT}" || true)_s${START}_n${N_WINDOWS}/shard_${SHARD}_of_${N_SHARDS}.json"
mkdir -p "$(dirname "$OUT")"
"$PY" -m evaluation.run_qg_specwind_da --spec "$SPEC" --root experiments/qg_datasets \
  --split "$SPLIT" --start "$START" --n-windows "$N_WINDOWS" --shard "$SHARD" --n-shards "$N_SHARDS" \
  --method "$METHOD" --cols-per-day "$COLS" --N "$ENS" --loc-radius "$LOC" --etkf-ridge "$RIDGE" \
  --loc-cross-layer "$CROSS" --init-ensemble "$INIT" --disp-frac "$DISP" --s1-kappa "$KAPPA" \
  --s1-preset "$PRESET" --s1-variant "$VARIANT" \
  --out "$OUT" --device cuda
