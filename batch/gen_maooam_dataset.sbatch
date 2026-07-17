#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=gen-maooam
#SBATCH --output=sbatch_logs/gen_maooam_dataset_%j.out
#SBATCH --error=sbatch_logs/gen_maooam_dataset_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

mkdir -p sbatch_logs experiments

echo "============================================"
echo " MAOOAM dataset generation (nx=3 + nx=4)   | $(date)"
echo "============================================"

echo ""
echo "--- Dataset 1: nx=3 (state_dim=60) ---"
python3 batch/gen_maooam_dataset.py \
    --nx 3 \
    --n-trajs 10 \
    --seed-start 42 \
    --device cuda

echo ""
echo "--- Dataset 2: nx=4 (state_dim=104) ---"
python3 batch/gen_maooam_dataset.py \
    --nx 4 \
    --n-trajs 10 \
    --seed-start 42 \
    --device cuda

echo ""
echo "============================================"
echo " Done: $(date)"
echo "============================================"
