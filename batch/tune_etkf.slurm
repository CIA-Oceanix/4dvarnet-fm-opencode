#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/tune_etkf_%j.out
#SBATCH --error=experiments/tune_etkf_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

ADDITIVE_INFLATION=$1

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "tune_etkf_add${ADDITIVE_INFLATION}" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --skip-const-bias --skip-weak --skip-strong \
    --etkf-ridge 1.0 \
    --etkf-additive-inflation ${ADDITIVE_INFLATION}