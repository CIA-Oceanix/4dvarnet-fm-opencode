#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-4dvar-cs34
#SBATCH --output=sbatch_logs/4dvar_cs3cs4_%j.out
#SBATCH --error=sbatch_logs/4dvar_cs3cs4_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVar (Weak+Strong) CS3/CS4 Evaluation | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "  DWS=50"
echo ""

python batch/run_4dvar_cs3cs4.py

echo ""
echo "  Done at: $(date)"
echo "============================================"
