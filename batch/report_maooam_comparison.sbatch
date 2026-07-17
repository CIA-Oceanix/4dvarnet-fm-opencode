#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=report_maooam_comparison
#SBATCH --output=sbatch_logs/report_maooam_comparison_%j.out
#SBATCH --error=sbatch_logs/report_maooam_comparison_%j.err

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode || exit 1

PYTHON=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

echo "=== MAOOAM Configuration Comparison Report ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

$PYTHON reports/report_maooam_comparison.py

echo ""
echo "=== Done ==="
date