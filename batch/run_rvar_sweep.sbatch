#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-rvar-sweep
#SBATCH --array=0-5
#SBATCH --output=sbatch_logs/rvar_sweep_%j_%a.out
#SBATCH --error=sbatch_logs/rvar_sweep_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

RVAR_VALUES=(0.1 0.25 0.5 1.0 2.0 4.0)
RVAR=${RVAR_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " R_var Sensitivity Sweep | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / R_var: ${RVAR}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python batch/noise_density_sweep.py --rvar $RVAR

echo ""
echo "  Done at: $(date)"
echo "============================================"
