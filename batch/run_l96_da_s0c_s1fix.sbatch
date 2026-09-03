#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-da-s0c-s1fix
#SBATCH --output=sbatch_logs/l96_da_s0c_s1fix_%j.out
#SBATCH --error=sbatch_logs/l96_da_s0c_s1fix_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 S0c/S1c DA Baselines (S1 corrupted-forcing fix) | $(date)"
echo "  Re-runs the canonical S0c obsj2 cache with cfg_s1 case=2 fix"
echo "  (S1 DA now feeds forcing_corrupted; S0 unchanged = forcing_true)"
echo "  Parallel output under _s1cfix suffix for validation before swap"
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
  --obs-j 2 \
  --s1-j 2 \
  --eval-j 2 \
  --obs-interval 100 \
  --suffix _s0c_s1cfix \
  --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}' \
  --skip-weak

echo ""
echo " Done at: $(date)"
echo "============================================"
