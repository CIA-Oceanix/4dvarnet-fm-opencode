#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-s0b-obs30
#SBATCH --output=sbatch_logs/l96_da_s0b_obs30_%j.out
#SBATCH --error=sbatch_logs/l96_da_s0b_obs30_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 S0b DA Baselines (Obs30) | $(date)"
echo "  System: all-param + fast_weights randomization"
echo "  S0b: F,c1,h,hx,eps +-20%, fast_weights +-20%"
echo "  Partial obs: obs_j=2 (24D observed subspace)"
echo "  obs_interval: 100 (30 obs/window)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

python -u evaluate_all_l96.py \
  --da-window-steps 500 \
  --num-test-windows 200 \
  --t-max 3.0 \
  --enkf-inflation 2.0 \
  --etkf-inflation 2.0 \
  --obs-j 2 \
  --obs-interval 100 \
  --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "c1": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "h": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "eps": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}}' \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"
