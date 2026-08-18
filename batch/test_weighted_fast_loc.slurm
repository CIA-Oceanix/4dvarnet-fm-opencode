#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_weighted_fast_loc_%j.out
#SBATCH --error=experiments/test_weighted_fast_loc_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

# Weighted fast coupling (0.1) + localization (per_member)
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "weighted_fast_loc" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 100 --inflation 1.1 --loc-radius 2.0 \
    --etkf-loc-mode per_member \
    --truth-fast-weights-unobserved 0.1 --skip-weak