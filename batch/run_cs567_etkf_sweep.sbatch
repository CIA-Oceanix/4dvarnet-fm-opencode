#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-cs567-etkf
#SBATCH --array=0-10
#SBATCH --output=sbatch_logs/cs567_etkf_%j_%a.out
#SBATCH --error=sbatch_logs/cs567_etkf_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

INFLATION_VALUES=(1.0 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.5 1.6 2.0)
INFLATION=${INFLATION_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " CS5/CS6/CS7 ETKF Inflation Sweep | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Inflation: ${INFLATION}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python batch/cs567_sweep.py --method etkf --inflation $INFLATION

echo ""
echo "  Done at: $(date)"
echo "============================================"
