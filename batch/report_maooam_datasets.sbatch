#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=report-maooam
#SBATCH --output=sbatch_logs/report_maooam_%j.out
#SBATCH --error=sbatch_logs/report_maooam_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

mkdir -p sbatch_logs reports/outputs/figs/maooam_datasets

echo "============================================"
echo " MAOOAM dataset report                      | $(date)"
echo "============================================"

python3 reports/report_maooam_datasets.py

echo ""
echo "============================================"
echo " Done: $(date)"
echo "============================================"