#!/bin/bash
#SBATCH --job-name=l96-validate-b2
#SBATCH --output=experiments/l96_validate_b2_%j.out
#SBATCH --error=experiments/l96_validate_b2_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fdv}"
PYTHON="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV}/bin/python3"

echo "============================================"
echo " L96 Validate b2 (200 windows) | $(date)"
echo "============================================"
echo "  S1: single-scale DA (NO=40, J=0)"
echo "  param_bias=0.0, forcing_state_bias=0.0"
echo "  Python: ${PYTHON}"
echo ""

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "b2-validate" \
    --param-bias 0.0 \
    --forcing-state-bias 0.0 \
    --obs-frac 1.0 \
    --num-windows 200 \
    --da-window-steps 500 \
    --s1-single-scale \
    --skip-weak

echo ""
echo "Finished at $(date)"
echo "============================================"
