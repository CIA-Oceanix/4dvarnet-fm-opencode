#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-obsint-sweep
#SBATCH --array=0-4
#SBATCH --output=sbatch_logs/obsint_sweep_%j_%a.out
#SBATCH --error=sbatch_logs/obsint_sweep_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

OBS_INTERVAL_VALUES=(10 20 40 60 100)
OBS_INTERVAL=${OBS_INTERVAL_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " Observation Interval Sweep | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / obs_interval: ${OBS_INTERVAL}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python batch/noise_density_sweep.py --obs-interval $OBS_INTERVAL

echo ""
echo "  Done at: $(date)"
echo "============================================"
