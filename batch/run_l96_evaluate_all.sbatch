#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=l96-evaluate-all
#SBATCH --output=sbatch_logs/l96_evaluate_all_%j.out
#SBATCH --error=sbatch_logs/l96_evaluate_all_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 consolidated benchmark report | $(date)"
echo "  (CPU-only over cached artifacts)"
echo "============================================"
echo ""

python reports/l96/generate_l96_consolidated_report.py

echo ""
echo " Done at: $(date)"
echo "============================================"
