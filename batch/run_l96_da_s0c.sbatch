#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-s0c
#SBATCH --output=sbatch_logs/l96_da_s0c_%j.out
#SBATCH --error=sbatch_logs/l96_da_s0c_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

OBS_INTERVAL="${OBS_INTERVAL:-100}"

echo "============================================"
echo " L96 S0c/S1c DA Baselines | $(date)"
echo "  System: all-param randomization EXCEPT h"
echo "  S0c: F,c1,hx,eps,fast_weights +-20%, h FIXED"
echo "  S1c: biased DA forward (non-h +10%), h unbiased"
echo "  Partial obs: obs_j=2 (24D observed subspace)"
echo "  obs_interval: ${OBS_INTERVAL}"
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
  --obs-interval "${OBS_INTERVAL}" \
  --suffix _s0c \
  --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}' \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"
