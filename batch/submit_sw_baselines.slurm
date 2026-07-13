#!/bin/bash
#SBATCH --job-name=sw-baselines
#SBATCH --output=sw_baselines_%j.out
#SBATCH --error=sw_baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

set -euo pipefail
CONDA_ENV="${CONDA_ENV:-fdv}"
PYTHON="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV}/bin/python3"

echo "============================================"
echo " SW S0/S1 Baselines (200 windows) | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"

$PYTHON evaluate_all_sw.py \
    --num-test-windows 200 \
    --da-window-steps 500 \
    --batch-size 200 \
    --t-max 3.0 \
    --enkf-inflation 2.0 \
    --etkf-inflation 2.0

echo "============================================"
echo " Done: $(date)"
echo "============================================"
