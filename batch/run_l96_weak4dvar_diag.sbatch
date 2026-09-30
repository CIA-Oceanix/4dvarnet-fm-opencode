#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --nodelist=sl-mee-br-209
#SBATCH --job-name=l96-w4dvar-diag
#SBATCH --output=sbatch_logs/l96_w4dvar_diag_%A_%a.out
#SBATCH --error=sbatch_logs/l96_w4dvar_diag_%A_%a.err
# Why is L96Weak4DVar 33% worse than the benchmark Strong-4DVar at S0 even at q 0.01?
# Same 8 validation windows as run_l96_weak4dvar_sweep.sbatch: L96Weak4DVar in strong mode
# with 10 and 40 LBFGS iterations, and weak mode (q 0.1) with 40. Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments/weak4dvar_sweep
TASKS=("strong 0 10" "strong 0 40" "weak 0.1 40")
read -r KIND Q IT <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $KIND q=$Q max_iter=$IT | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
python -u scripts/weak4dvar_sweep_l96.py val_regular 8 "$KIND" "$Q" "experiments/weak4dvar_sweep/diag_${KIND}_q${Q}_it${IT}.json" "$IT"
echo "Done at $(date)"
