#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-cs567-dws
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/cs567_dws_%j_%a.out
#SBATCH --error=sbatch_logs/cs567_dws_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

DWS_VALUES=(40 60 80 120)
DWS=${DWS_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " CS5/CS6/CS7 DWS Sweep | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / DWS: ${DWS}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python batch/cs567_sweep.py --dws $DWS

echo ""
echo "  Done at: $(date)"
echo "============================================"
