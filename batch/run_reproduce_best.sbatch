#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=4dvarnet-reproduce-best
#SBATCH --output=sbatch_logs/reproduce_best_%j.out
#SBATCH --error=sbatch_logs/reproduce_best_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Reproduce Best (B1_small_unet) | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python run_experiments.py --experiment B1_small_unet

echo ""
echo "============================================"
echo " Done at: $(date)"
echo "============================================"
