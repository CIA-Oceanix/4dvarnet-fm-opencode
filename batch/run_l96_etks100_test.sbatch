#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --array=0-3%2
#SBATCH --nodelist=sl-mee-br-211
#SBATCH --job-name=l96-etks100-test
#SBATCH --output=sbatch_logs/l96_etks100_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etks100_test_%A_%a.err
# 100-member ETKS (and its own filter pass) on the 200 L96 test windows at the validation-selected
# inflation (S0 1.05, S1 2.5; one value per case, worse-layout RMSE loss), regular grid and
# canonical random set: the ensemble-size sensitivity row. Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments/etks100_test
TASKS=("test_regular s0 1.05" "test_regular s1 2.5" "test_rlayout s0 1.05" "test_rlayout s1 2.5")
read -r SET CASE LAM <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $SET $CASE lambda=$LAM N=100 | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
python -u scripts/etks_ensemble_check.py "$SET" 100 "experiments/etks100_test/etks100_${SET}_${CASE}_lam${LAM}.json" "$LAM" "$CASE"
echo "Done at $(date)"
