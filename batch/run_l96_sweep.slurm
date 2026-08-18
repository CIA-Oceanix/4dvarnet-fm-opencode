#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/l96_sweep_%j.out
#SBATCH --error=experiments/l96_sweep_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fdv}"
PYTHON="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV}/bin/python3"

echo "============================================"
echo " L96 Sweep: ${LABEL}  |  $(date)"
echo "============================================"
echo "  param_bias=${PARAM_BIAS}, forcing_state_bias=${FSB}"
echo "  Python: ${PYTHON}"
echo ""

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "  Starting sweep..."
echo ""

$PYTHON -u evaluation/run_l96_sweep.py \
    --label "${LABEL}" \
    --param-bias "${PARAM_BIAS}" \
    --forcing-state-bias "${FSB}" \
    --obs-frac 1.0 \
    --num-windows "${NUM_WINDOWS:-20}" \
    --da-window-steps 500 \
    --skip-weak

echo ""
echo "Finished at $(date)"
echo "============================================"