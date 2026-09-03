#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-slowobs
#SBATCH --output=sbatch_logs/l96_da_slowobs_%j.out
#SBATCH --error=sbatch_logs/l96_da_slowobs_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 S0/S1 DA Baselines (slow-only obs) | $(date)"
echo "  obs_j=0 (8 slow X observed, no fast Y)"
echo "  S1 reduced dynamics kept at J=2 (24D state space)"
echo "  eval subspace = 24D obsj2 group (slow + first-2-fast)"
echo "  reuses canonical obsj2 trajectories via obsj0 cache"
echo "  methods: EnKF / ETKF / Strong-4DVar (skip Weak)"
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
  --obs-j 0 \
  --s1-j 2 \
  --eval-j 2 \
  --obs-interval 100 \
  --suffix _slowobs \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"
