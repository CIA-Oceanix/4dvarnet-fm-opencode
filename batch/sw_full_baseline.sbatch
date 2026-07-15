#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=sw-full-baseline
#SBATCH --output=sbatch_logs/sw_full_baseline_%j.out
#SBATCH --error=sbatch_logs/sw_full_baseline_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Full SW baseline (200 windows, Strong-4DVar only) ==="
echo "Nx=64, da_window_steps=50, Strong-4DVar, S0+S1"

python evaluate_all_sw.py \
    --num-test-windows 200 \
    --da-window-steps 50 \
    --batch-size 20 \
    --methods Strong-4DVar \
    --Nx 64 \
    --Ny 64 \
    --output-dir outputs/sw_baselines

echo "=== Done ==="