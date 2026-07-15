#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=sw-256
#SBATCH --output=sbatch_logs/sw_256_%j.out
#SBATCH --error=sbatch_logs/sw_256_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "Starting SW 256x256 diagnostics at $(date)"
python reports/diagnose_sw_256.py
echo "Finished at $(date)"