#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_s1_j2_obs5_%j.out
#SBATCH --error=experiments/test_s1_j2_obs5_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

echo "============================================"
echo " Test: S1 J=2, obs_interval=5  |  $(date)"
echo "============================================"
echo ""

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "test_s1_j2_obs5" \
    --param-bias 0.0 \
    --forcing-state-bias 0.0 \
    --num-windows 5 \
    --da-window-steps 500 \
    --obs-j 2 \
    --s1-j 2 \
    --obs-interval 5 \
    --skip-weak

echo ""
echo "============================================"
echo " Completed at $(date)"
echo "============================================"