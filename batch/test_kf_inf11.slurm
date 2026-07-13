#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_kf_inf11_%j.out
#SBATCH --error=experiments/test_kf_inf11_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "kf_inf11" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 --skip-weak