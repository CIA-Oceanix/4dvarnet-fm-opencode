#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=sw-new-da
#SBATCH --output=sbatch_logs/sw_new_da_%j.out
#SBATCH --error=sbatch_logs/sw_new_da_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Test DA with new SW parameters ==="
python _tmp_test_new_params_da.py
echo "=== Done ==="