#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_pct5_strong_%j.out
#SBATCH --error=experiments/test_pct5_strong_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

LABEL=$1
ETKF_RIDGE=${2:-1.0}

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "${LABEL}" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --skip-const-bias --skip-weak \
    --etkf-ridge ${ETKF_RIDGE}