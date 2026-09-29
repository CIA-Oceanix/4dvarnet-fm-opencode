#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96-da-post291
#SBATCH --output=sbatch_logs/l96_da_post291_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_post291_%A_%a.err
# batch/run_l96_da_crps.sbatch, unchanged, rerun on master after #291 (ETKF square
# root keeps the unobserved anomaly directions). Keeps the per-time ensemble
# variance in the trajectory npz, which pooled spread/RMSE needs: task 0 regular
# 30-obs grid (P1 cache, hash-guarded), task 1 canonical random layouts
# (regenerated from the same seeds; compare against the archived layouts file).
# Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=experiments/l96_datasets_obsj2_int100_nwin200.pt
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
echo "commit $(git rev-parse HEAD)"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  python -u evaluate_all_l96.py --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
    --obs-j 2 --obs-interval 100 --suffix _s0c_crps \
    --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}' \
    --skip-weak --skip-strong --da-fast-weights
else
  python -u eval_da_random_layout_l96.py --n-obs-range 10 100 --fast-range 4 16 --n-windows 200 --n-draws 1 \
    --da-window-steps 500 --methods ETKF EnKF --data-cache "$CACHE"
fi
echo "Done at $(date)"
