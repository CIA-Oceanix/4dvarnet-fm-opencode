#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%4
#SBATCH --nodelist=sl-mee-br-209
#SBATCH --job-name=l96-w4dvar-sweep
#SBATCH --output=sbatch_logs/l96_w4dvar_sweep_%A_%a.out
#SBATCH --error=sbatch_logs/l96_w4dvar_sweep_%A_%a.err
# Weak-4DVar (L96Weak4DVar, LBFGS) model-error scale sweep on 8 L96 validation windows
# (regular grid), S0 and S1: q_var_scale {0.01, 0.03, 0.1, 0.3}; task 4 = the benchmark
# Strong-4DVar on the same windows (paired reference). Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments/weak4dvar_sweep
TASKS=("weak 0.01" "weak 0.03" "weak 0.1" "weak 0.3" "strong_ref 0")
read -r KIND Q <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $KIND q=$Q | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
python -u scripts/weak4dvar_sweep_l96.py val_regular 8 "$KIND" "$Q" "experiments/weak4dvar_sweep/${KIND}_q${Q}.json"
echo "Done at $(date)"
