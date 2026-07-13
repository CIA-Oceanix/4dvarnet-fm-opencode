#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=sw-baseline-smoke
#SBATCH --output=sbatch_logs/sw_baseline_smoke_%j.out
#SBATCH --error=sbatch_logs/sw_baseline_smoke_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW Baseline Smoke Test (5 windows) | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"

python evaluate_all_sw.py \
    --num-test-windows 5 \
    --da-window-steps 500 \
    --batch-size 5 \
    --t-max 0.5

echo "============================================"
echo " Done: $(date)"
echo "============================================"
