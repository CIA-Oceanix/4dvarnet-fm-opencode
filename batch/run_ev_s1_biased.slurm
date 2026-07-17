#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/ev_s1_biased_%j.out
#SBATCH --error=experiments/ev_s1_biased_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "ev_s1_biased_f15_c115_ce08" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 200 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --etkf-ridge 1.0 --etkf-additive-inflation 0.05 \
    --dataset-cache experiments/dataset_ens200.pt \
    --s1-f-bias 0.15 --s1-c1-bias 0.15 --s1-coupling-exponent 0.8 \
    --skip-weak --skip-strong