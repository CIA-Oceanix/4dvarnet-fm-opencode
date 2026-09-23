#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=l96-da-reg-s0
#SBATCH --output=sbatch_logs/l96_da_regular_s0_inf15_%j.out
#SBATCH --error=sbatch_logs/l96_da_regular_s0_inf15_%j.err
# Regular-grid (P1 30-obs) S0 ETKF/EnKF at the benchmark S0 inflation 1.5,
# corrected mode (per-window fast_weights). S1 (inflation 2.0) and
# Strong-4DVar (no inflation) already exist from the corrected 2.0 run.
# Refuses to run unless the test cache is the P1 cache (no regeneration).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=experiments/l96_datasets_obsj2_int100_nwin200.pt
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
python -u evaluate_all_l96.py --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
  --enkf-inflation 1.5 --etkf-inflation 1.5 --obs-j 2 --obs-interval 100 --suffix _s0c \
  --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}' \
  --skip-weak --skip-strong --da-fast-weights --cases s0
echo "Done at $(date)"
