#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=sw-4dvar-full
#SBATCH --output=sbatch_logs/sw_4dvar_full_%j.out
#SBATCH --error=sbatch_logs/sw_4dvar_full_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW Strong-4DVar Full (200 windows, S0+S1) | $(date)"
echo "============================================"
echo "  da_window_steps=10  opt_steps=200  lr=0.01  batch_size=10  methods=Strong-4DVar"

python evaluate_all_sw.py \
    --num-test-windows 200 \
    --da-window-steps 10 \
    --batch-size 10 \
    --methods Strong-4DVar

echo "============================================"
echo " Done: $(date)"
echo "============================================"