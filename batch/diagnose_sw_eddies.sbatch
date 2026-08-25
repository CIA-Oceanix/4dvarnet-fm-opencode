#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=sw-eddy
#SBATCH --output=sbatch_logs/sw_eddy_%j.out
#SBATCH --error=sbatch_logs/sw_eddy_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "Starting SW eddy diagnostics at $(date)"
python reports/diagnose_sw_eddies.py
echo "Finished at $(date)"