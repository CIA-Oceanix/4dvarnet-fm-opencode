#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-s0b-s1b
#SBATCH --output=sbatch_logs/l96_da_s0b_s1b_%j.out
#SBATCH --error=sbatch_logs/l96_da_s0b_s1b_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 S0b/S1b DA Baselines | $(date)"
echo "  System: all-5-param + fast_weights randomization"
echo "  S0b: F,c1,h,hx,eps +-20%, fast_weights +-20%"
echo "  S1b: same + biased DA fast_weights"
echo "  Partial obs: obs_j=2 (24D observed subspace)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

# Run S0b/S1b DA baselines with fast_weights randomization enabled.
# --randomize activates per-window fast_weights +-20% via the _draw_l96_params per-param path.
# --skip-weak avoids 4DVar on CPU (too slow for full 200-window sweep).
# The _fw cache suffix activates automatically when fast_weights is in the randomize dict.
python -u evaluate_all_l96.py \
  --da-window-steps 500 \
  --num-test-windows 200 \
  --t-max 3.0 \
  --enkf-inflation 2.0 \
  --etkf-inflation 2.0 \
  --obs-j 2 \
  --obs-interval 200 \
  --randomize '{"fast_weights": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "F": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "c1": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "h": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "eps": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}}' \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"
