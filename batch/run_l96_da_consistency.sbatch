#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-consistency
#SBATCH --output=sbatch_logs/l96_da_consistency_%j.out
#SBATCH --error=sbatch_logs/l96_da_consistency_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 DA Baseline Consistency Check | $(date)"
echo "  System: all-5-param randomization (F, c1, h, hx, eps)"
echo "  S0: +-20%, S1: +-20% + 10% bias"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

# Re-run L96 DA baselines on the all-5-param S0/S1 test sets, DWS=500, 200 windows.
# Use a distinct suffix so it does not collide with the existing F-only benchmarks.
python -u evaluate_all_l96.py \
  --da-window-steps 500 \
  --num-test-windows 200 \
  --t-max 3.0 \
  --enkf-inflation 2.0 \
  --etkf-inflation 2.0 \
  --suffix _all5params

echo ""
echo " Done at: $(date)"
echo "============================================"