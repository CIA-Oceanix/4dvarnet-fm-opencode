#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-21%4
#SBATCH --job-name=l96-s1full-val
#SBATCH --output=sbatch_logs/l96_s1full_val_%A_%a.out
#SBATCH --error=sbatch_logs/l96_s1full_val_%A_%a.err
# Validation re-tuning of the L96 S1 DA schemes when the S1 DA model is the full 40-variable
# model (--s1-j 4) instead of the default truncated 24-variable one (s1_j = obs_j = 2).
#   tasks 0-7:   ETKF inflation sweep, 50 regular validation windows
#   tasks 8-15:  EnKF inflation sweep, same windows
#   tasks 16-20: Weak-4DVar model-error scale sweep (L-BFGS 40 it), 8 validation windows, S1 only
#   task 21:     Strong-4DVar reference on the same 8 windows, S1 only
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
OUT=experiments/l96_s1_full_model/val
mkdir -p "$MPLCONFIGDIR" sbatch_logs "$OUT"
T=$SLURM_ARRAY_TASK_ID
LAMS=(1.0 1.05 1.1 1.15 1.2 1.3 1.5 2.0)
QS=(0.01 0.03 0.1 0.3 1.0)
echo "task $T | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
if (( T < 8 )); then
  python -u scripts/sweep_l96_etkf_inflation.py --valset regular --method ETKF --inflation "${LAMS[$T]}" --s1-j 4 \
    --out "$OUT/etkf_lam${LAMS[$T]}.json"
elif (( T < 16 )); then
  python -u scripts/sweep_l96_etkf_inflation.py --valset regular --method EnKF --inflation "${LAMS[$((T - 8))]}" --s1-j 4 \
    --out "$OUT/enkf_lam${LAMS[$((T - 8))]}.json"
elif (( T < 21 )); then
  Q=${QS[$((T - 16))]}
  CASES=s1 S1_J=4 python -u scripts/weak4dvar_sweep_l96.py val_regular 8 weak "$Q" "$OUT/weak_q${Q}.json" 40
else
  CASES=s1 S1_J=4 python -u scripts/weak4dvar_sweep_l96.py val_regular 8 strong_ref 0 "$OUT/strong_ref.json" 40
fi
echo "Done at $(date)"
