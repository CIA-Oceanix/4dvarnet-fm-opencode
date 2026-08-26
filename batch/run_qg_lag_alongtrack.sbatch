#!/bin/bash
#SBATCH --job-name=qg_lag_test
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-0
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 60 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 0 \
    --method-list etkf \
    --init lagged \
    --geometry alongtrack \
    --scenarios test_s0,test_s1a \
    --outdir reports/outputs/qg_lag_alongtrack \
    --tag lag_alongtrack
