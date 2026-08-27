#!/bin/bash
#SBATCH --job-name=qg_psi_obs
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 60 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --init-lag-days-list 0.5 \
    --obs-var psi \
    --scenarios test_s0,test_s1a \
    --outdir reports/outputs/qg_psi_obs_test \
    --tag psi_obs_cols4_lag0.5
