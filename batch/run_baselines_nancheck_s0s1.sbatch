#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=bl-nan-s0s1
#SBATCH --output=sbatch_logs/bl_nan_s0s1_%j.out
#SBATCH --error=sbatch_logs/bl_nan_s0s1_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments

echo "============================================"
echo " Baseline NaN Check S0/S1 | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "  DWS=50, EnKF infl=2.0, ETKF infl=2.0"
echo "============================================"

python batch/_run_baselines_nancheck_s0s1.py

echo "============================================"
echo " Done at $(date)"
echo "============================================"