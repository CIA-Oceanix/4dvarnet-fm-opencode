#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=4dvarnet-tests
#SBATCH --output=sbatch_logs/tests_%j.out
#SBATCH --error=sbatch_logs/tests_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " 4DVarNet-FM Test Suite | $(date)"
echo "============================================"
echo ""

pytest tests/ -m "not slow" -v --tb=short

echo ""
echo "============================================"
echo " Tests completed at $(date)"
echo "============================================"
