#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_pct_noise_%j.out
#SBATCH --error=experiments/test_pct_noise_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

# 1) Baseline: per-component 5% noise, same obs interval for S0 and S1
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "pct5_s0s1_eq" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --skip-weak

# 2) Duplicate of pct5_s0s1_eq for cross-check
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "pct5_s1obs10" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --skip-weak

# 3) Duplicate of pct5_s0s1_eq for cross-check
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "pct5_s1obs20" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --skip-weak