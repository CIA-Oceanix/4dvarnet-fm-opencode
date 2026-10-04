#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%3
#SBATCH --nodelist=sl-mee-br-204
#SBATCH --job-name=l96-w4dvar-it40
#SBATCH --output=sbatch_logs/l96_w4dvar_it40_%A_%a.out
#SBATCH --error=sbatch_logs/l96_w4dvar_it40_%A_%a.err
# Weak-4DVar (L96Weak4DVar, LBFGS 40 iterations per sub-window -- the budget at which its strong
# mode reproduces the benchmark Strong-4DVar) model-error scale re-tune on the same 8 validation
# windows as run_l96_weak4dvar_sweep.sbatch: q_var_scale {0.003, 0.01, 0.03, 0.3, 1.0} (0.1 is in
# run_l96_weak4dvar_diag.sbatch). Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments/weak4dvar_sweep
QS=(0.003 0.01 0.03 0.3 1.0)
Q=${QS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: weak q=$Q max_iter=40 | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
python -u scripts/weak4dvar_sweep_l96.py val_regular 8 weak "$Q" "experiments/weak4dvar_sweep/it40_weak_q${Q}.json" 40
echo "Done at $(date)"
