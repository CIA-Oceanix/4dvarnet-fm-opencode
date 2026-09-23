#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96-da-crps
#SBATCH --output=sbatch_logs/l96_da_crps_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_crps_%A_%a.err
# ETKF/EnKF reruns that record the ANALYSIS-ensemble CRPS (benchmark inflation
# S0 1.5 / S1 2.0, per-window fast_weights): task 0 regular 30-obs grid (P1
# cache, hash-guarded), task 1 canonical random layouts (regenerated from the
# same seeds; checked by content against the canonical test set).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=experiments/l96_datasets_obsj2_int100_nwin200.pt
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
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
