#!/bin/bash
#SBATCH --job-name=qg_lag_full
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2
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
    --init-lag-days-list 1.0,2.0,5.0 \
    --scenarios test_s0,test_s1a \
    --outdir reports/outputs/qg_lag_full \
    --tag lag_full_narrow
