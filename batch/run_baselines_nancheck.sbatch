#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=baselines_nancheck
#SBATCH --output=sbatch_logs/baselines_nancheck_%j.out
#SBATCH --error=sbatch_logs/baselines_nancheck_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:${PATH}"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" sbatch_logs experiments reports/outputs

echo "============================================"
echo "  DA Baselines — NaN-check (v2)"
echo "  $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python batch/_run_baselines_nancheck.py

echo ""
echo "============================================"
echo "  Done at $(date)"
echo "============================================"