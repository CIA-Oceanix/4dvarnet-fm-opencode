#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --exclude=sl-mee-br-207
#SBATCH --job-name=l96-etks-id
#SBATCH --output=sbatch_logs/l96_etks_id_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etks_id_%A_%a.err
# ETKS filter/smoother identity with the ETKS's own filter pass (one ensemble
# realisation), and its ensemble-size dependence: test sets at N=30, validation
# sets at N=30 and N=100. Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/sbatch_logs/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
TASKS=("test_regular 30" "test_rlayout 30" "val_regular 30" "val_rlayout 30" "val_regular 100" "val_rlayout 100")
read -r SET N <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $SET N=$N | commit $(git rev-parse --short HEAD) | $(hostname) $(date +%T)"
python -u scripts/etks_ensemble_check.py "$SET" "$N" "experiments/etks_identity/etks_${SET}_N${N}.json"
rm -rf "$MPLCONFIGDIR"
echo "Done at $(date)"
