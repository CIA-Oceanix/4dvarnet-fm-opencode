#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
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

# Observation interval in steps (default 200 = ~15 obs/window; 100 = 2x denser).
OBS_INTERVAL="${OBS_INTERVAL:-200}"

echo "============================================"
echo " L96 DA Baseline Consistency Check | $(date)"
echo "  System: all-5-param randomization (F, c1, h, hx, eps)"
echo "  S0: +-20%, S1: +-20% + 10% bias"
echo "  Partial obs: obs_j=2 (24D observed subspace)"
echo "  obs_interval: ${OBS_INTERVAL}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

# Re-run L96 DA baselines on the all-5-param S0/S1 test sets, DWS=500, 200 windows,
# with partial observations (obs_j=2 -> 24D observed subspace).
# obs_j<4 auto-appends the _obsj2 cache tag; obs_interval auto-appends the _int{N} tag.
python -u evaluate_all_l96.py \
  --da-window-steps 500 \
  --num-test-windows 200 \
  --t-max 3.0 \
  --enkf-inflation 2.0 \
  --etkf-inflation 2.0 \
  --obs-j 2 \
  --obs-interval "${OBS_INTERVAL}" \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"