#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-39%4
#SBATCH --job-name=l96-w4dvar-bench
#SBATCH --output=sbatch_logs/l96_w4dvar_bench_%A_%a.out
#SBATCH --error=sbatch_logs/l96_w4dvar_bench_%A_%a.err
# Weak-4DVar (L96Weak4DVar, LBFGS 40 iterations per sub-window) benchmark rows on the
# 200-window regular test set at the per-case model-error scales picked on validation
# (run_l96_weak4dvar_sweep_it40.sbatch): S0 q_var_scale 0.03, S1 0.3. 40 tasks = 20 chunks of
# 10 windows x 2 cases (even task -> S0, odd -> S1); each keeps its trajectories in
# experiments/weak4dvar_bench/ for assembling the rows. GPU model / node / index range are
# overridden at submission (e.g. --gres=gpu:a40:1 --nodelist=... --array=0-23%3).
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
OUT=experiments/weak4dvar_bench
mkdir -p "$MPLCONFIGDIR" sbatch_logs "$OUT"
T=$SLURM_ARRAY_TASK_ID
if (( T % 2 == 0 )); then CASE=s0; Q=0.03; else CASE=s1; Q=0.3; fi
W0=$(( (T / 2) * 10 ))
echo "task $T: $CASE q=$Q windows $W0-$((W0 + 9)) | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
export W_START=$W0 CASES=$CASE KEEP_DIR=$OUT
python -u scripts/weak4dvar_sweep_l96.py test_regular 10 weak "$Q" "$OUT/${CASE}_q${Q}_it40_w$(printf %03d $W0).json" 40
echo "Done at $(date)"
