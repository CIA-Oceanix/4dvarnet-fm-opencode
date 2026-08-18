#!/bin/bash
#SBATCH --job-name=l96-weak-tune
#SBATCH --output=experiments/l96_weak_tune_%j.out
#SBATCH --error=experiments/l96_weak_tune_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fdv}"
PYTHON="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV}/bin/python3"

echo "============================================"
echo " L96 Weak-4DVar Tune (b2: single-scale S1) | $(date)"
echo "============================================"
echo "  Python: ${PYTHON}"
echo ""

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

$PYTHON -u evaluation/tune_l96_weak4dvar.py

echo ""
echo "Finished at $(date)"
echo "============================================"