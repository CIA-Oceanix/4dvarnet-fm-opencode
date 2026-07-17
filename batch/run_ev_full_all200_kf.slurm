#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/ev_full_all200_kf_%j.out
#SBATCH --error=experiments/ev_full_all200_kf_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "ev_full_all200_kf" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 200 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --r-var-pct 5 \
    --etkf-ridge 1.0 --etkf-additive-inflation 0.05 \
    --dataset-cache experiments/dataset_ens200.pt \
    --skip-weak --skip-strong