#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-19%10
#SBATCH --exclude=sl-mee-br-207
#SBATCH --job-name=l96-etks100-infl
#SBATCH --output=sbatch_logs/l96_etks100_infl_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etks100_infl_%A_%a.err
# Inflation sweep for the 100-member ETKS (and its own filter pass) on the L96
# validation windows (50 per case), one case per task: S0 lambda in
# {1.0 ... 1.3}, S1 lambda in {2.0 ... 3.5}, regular grid and random layout.
# Selection as for #295: one value per case minimising the worse layout's
# RMSE loss. Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/sbatch_logs/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" experiments/etks100_inflation_sweep
TASKS=()
for V in val_regular val_rlayout; do
  for L in 1.0 1.05 1.1 1.15 1.2 1.3; do TASKS+=("$V s0 $L"); done
  for L in 2.0 2.5 3.0 3.5; do TASKS+=("$V s1 $L"); done
done
read -r SET CASE LAM <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $SET $CASE lambda=$LAM N=100 | commit $(git rev-parse --short HEAD) | $(hostname) $(date +%T)"
python -u scripts/etks_ensemble_check.py "$SET" 100 "experiments/etks100_inflation_sweep/etks100_${SET}_${CASE}_lam${LAM}.json" "$LAM" "$CASE"
rm -rf "$MPLCONFIGDIR"
echo "Done at $(date)"
