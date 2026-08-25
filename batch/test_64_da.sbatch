#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=sw-64-da
#SBATCH --output=sbatch_logs/sw_64_da_%j.out
#SBATCH --error=sbatch_logs/sw_64_da_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "Starting DA test at $(date)"
python _tmp_test_64_da.py
echo "Finished at $(date)"