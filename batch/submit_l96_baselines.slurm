#!/bin/bash
#SBATCH --job-name=l96-baselines
#SBATCH --output=l96_baselines_%j.out
#SBATCH --error=l96_baselines_%j.err
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
echo " L96 S0/S1 Baselines  |  $(date)"
echo "============================================"
echo "  Env:   ${CONDA_ENV}"
echo "  Python: ${PYTHON}"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

# Run L96 S0/S1 baseline evaluation
echo "Starting L96 baseline evaluation..."
$PYTHON -c "import sys; print('Python OK', flush=True)"
$PYTHON -c "import torch; print('torch', torch.__version__, flush=True)"
echo "  Running evaluate_all_l96.py..."
$PYTHON -u evaluate_all_l96.py \
    --da-window-steps 500 \
    --num-test-windows 3 \
    --batch-size 3 \
    --t-max 0.5 \
    --enkf-inflation 2.0 \
    --etkf-inflation 2.0

echo ""
echo "Finished at $(date)"
echo "============================================"