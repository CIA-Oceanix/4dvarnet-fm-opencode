#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=l96-valsets
#SBATCH --output=sbatch_logs/l96_make_valsets_%j.out
#SBATCH --error=sbatch_logs/l96_make_valsets_%j.err
# 50 validation windows per case (disjoint from train/val/test), regular and
# random-layout versions, for tuning SDA guidance / hybrid warm start / flow
# calibration without touching the test set.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
python scripts/make_l96_validation_sets.py --n-windows 50
