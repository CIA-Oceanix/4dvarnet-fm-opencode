#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=4dvarnet-fm-baselines
#SBATCH --output=sbatch_logs/baselines_%j.out
#SBATCH --error=sbatch_logs/baselines_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Baselines  |  $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python eval_baselines.py hydra.run.dir=. hydra.output_subdir=null

echo ""
echo "  Done at: $(date)"
echo "============================================"
