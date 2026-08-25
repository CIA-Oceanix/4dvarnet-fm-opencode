#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=sw-xkf-light
#SBATCH --output=sbatch_logs/sw_xkf_light_%j.out
#SBATCH --error=sbatch_logs/sw_xkf_light_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW xKF Light Smoke (5 windows, S0+S1) | $(date)"
echo "============================================"
echo "  da_window_steps=50  batch_size=1  inflation=1.1  Nx=64"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"

python evaluate_all_sw.py \
    --num-test-windows 5 \
    --da-window-steps 50 \
    --batch-size 1 \
    --enkf-inflation 1.1 \
    --etkf-inflation 1.1 \
    --loc-radius 3 \
    --methods EnKF ETKF

echo "============================================"
echo " Done: $(date)"
echo "============================================"